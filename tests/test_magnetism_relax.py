"""Tests for the collinear FM/FiM moment relaxation.

The end-to-end tests are skip-guarded on the real backbone, like the
classifier's own real-backbone test in ``tests/test_magnetism_mace_mlp.py``:
the ~80 MB checkpoint is not present in CI. The validity checks are tested
without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_ml.models.magnetism._mace_backbone import safe_load
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
    CONVERGED_AFTER_RETRY,
    INVALID,
    MAGNETIC_GAIN_SLACK_EV_PER_ATOM,
    MAX_EXCHANGE_EV_PER_UB2,
    OK,
    WARNING_NOT_CONVERGED,
    _attempt,
    _validity_reasons,
    domain_limits,
    relax,
)
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.seed_moments import (
    seed_moments_fm_fim,
)


def _real_checkpoint() -> Path:
    checkpoint = Path(
        "local_data/artifacts/sgk1t-48m42/mmace-matpes-pbe-pretrained.model"
    )
    if not checkpoint.is_file():
        pytest.skip("the mMACE backbone checkpoint is not present locally")
    return checkpoint


def test_relax_restores_torchs_global_default_dtype() -> None:
    """Regression test for goldilocks-ml#98.

    ``MagneticMACECalculator`` without ``default_dtype`` makes mace
    auto-detect the checkpoint's dtype and call the process-wide, unscoped
    ``torch.set_default_dtype(float64)`` -- confirmed to otherwise leak past
    ``relax()`` returning, which at the time broke every later float32
    prediction (e.g. the CGCNN ``is_metal`` classifier) in the same process.
    Those models now fix their own dtype (goldilocks-ml#110), but the global
    must still be restored. Checks both the global directly and that a real
    float32 embedding call still works right after ``relax()`` returns.
    """
    import torch

    from goldilocks_ml.models.magnetism._mace_backbone import embed_structures

    checkpoint = _real_checkpoint()
    assert torch.get_default_dtype() == torch.float32

    structure = Structure(
        Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    moments = np.zeros((2, 3))
    moments[:, 2] = 2.2
    relax(structure, moments, checkpoint=checkpoint, device="cpu")

    assert torch.get_default_dtype() == torch.float32

    silicon = Structure(
        Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    rows = embed_structures([silicon], checkpoint=checkpoint)
    assert np.isfinite(rows).all()


def test_relax_holds_moments_collinear_on_the_real_backbone() -> None:
    """A regression test for goldilocks-ml#92: ``ac8ff476`` had no gradient-
    masking mechanism at all, so relaxing against it either raised
    ``TypeError`` (with ``use_collinear``/``constrain_magnitude`` passed) or,
    had those kwargs been silently dropped instead, would have let the x/y
    components drift away from zero -- a collinear seed relaxing into a
    non-collinear result without anyone noticing. This checks both that it
    runs and that x/y stay exactly at zero.
    """
    checkpoint = _real_checkpoint()

    structure = Structure(
        Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    moments = np.zeros((2, 3))
    moments[:, 2] = 2.2

    result = relax(structure, moments, checkpoint=checkpoint, device="cpu")

    assert result.status == OK
    assert np.isfinite(result.energy_ev)
    assert result.moments[:, 0] == pytest.approx(0.0)
    assert result.moments[:, 1] == pytest.approx(0.0)
    assert (result.moments[:, 2] != 0.0).all()


def test_an_unfinished_relaxation_is_reported_as_not_converged() -> None:
    """mace returns no gradient history, so relax() reads the final magnetic
    forces itself. One SCF step from a far seed cannot converge, and must
    say so rather than pass as ``ok``.
    """
    checkpoint = _real_checkpoint()

    structure = Structure(
        Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    moments = np.zeros((2, 3))
    moments[:, 2] = 0.5

    result = _attempt(
        structure,
        moments,
        raw_model=safe_load(checkpoint),
        device="cpu",
        step_size=1e-4,
        n_scf_step=1,
    )

    assert result.status == WARNING_NOT_CONVERGED
    assert "final magnetic gradient norm" in result.reasons[0]


# Two Fe sites at 2 uB may gain at most this much per atom.
_BOUND_EV_PER_ATOM = MAX_EXCHANGE_EV_PER_UB2 * 4.0 + MAGNETIC_GAIN_SLACK_EV_PER_ATOM


def _gain_reasons(gain_ev_per_atom: float, zero_moment_energy_ev: float = -16.0):
    """Validity reasons for two Fe sites at 2 uB with the given gain."""
    moments = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    energy_ev = zero_moment_energy_ev + 2 * gain_ev_per_atom
    return _validity_reasons(
        energy_ev, zero_moment_energy_ev, moments, ["Fe", "Fe"], {"Fe": 5.0}
    )


def test_a_gain_inside_the_exchange_bound_is_valid() -> None:
    assert _gain_reasons(-0.9 * _BOUND_EV_PER_ATOM) == []
    # Ending above the zero-moment energy is a metastable state, not an error.
    assert _gain_reasons(0.3) == []


def test_a_gain_past_the_exchange_bound_is_invalid() -> None:
    reasons = _gain_reasons(-1.1 * _BOUND_EV_PER_ATOM)

    assert len(reasons) == 1
    assert "magnetic energy gain" in reasons[0]


def test_the_gain_is_skipped_without_a_finite_zero_moment_energy() -> None:
    moments = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

    reasons = _validity_reasons(
        -116.0, float("nan"), moments, ["Fe", "Fe"], {"Fe": 5.0}
    )

    assert reasons == []


def test_bcc_fe_from_its_seed_is_not_accepted_at_an_unphysical_moment() -> None:
    """goldilocks-ml#120: MatPES mp-13 (bcc Fe, DFT 2.25 uB) seeded at 3.6 uB.

    Without the gain check the retry stopped at 4.21 uB, 22.7 eV/atom below
    the zero-moment energy, and passed as converged_after_retry.
    """
    checkpoint = _real_checkpoint()
    structure = Structure(
        Lattice(
            [
                [2.3361, 0.0001, -0.8258],
                [-1.1681, 2.0230, -0.8261],
                [1.1701, 2.0273, 3.3103],
            ]
        ),
        ["Fe", "Fe"],
        [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    )
    seed = seed_moments_fm_fim(
        structure, max_moment_by_symbol=domain_limits(safe_load(checkpoint))
    )

    result = relax(structure, seed, checkpoint=checkpoint, device="cpu")

    accepted = result.status in (OK, CONVERGED_AFTER_RETRY)
    assert not (accepted and np.abs(result.moments[:, 2]).max() > 4.0)
    if result.status == INVALID:
        assert any("magnetic energy gain" in reason for reason in result.reasons)
