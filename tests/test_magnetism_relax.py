"""Tests for the collinear FM/FiM moment relaxation.

The descent and the validity checks are tested without the backbone, and
``relax()`` end to end on a small random-weight magnetic MACE. The tests on
the real backbone are skip-guarded, like the classifier's own in
``tests/test_magnetism_mace_mlp.py``: the ~80 MB checkpoint is not present
in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import tiny_magnetic_backbone
from pymatgen.core import Lattice, Structure

from goldilocks_ml.models.magnetism._mace_backbone import safe_load
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax import (
    relax as relax_module,
)
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
    GRADIENT_TOLERANCE,
    INVALID,
    MAGNETIC_GAIN_SLACK_EV_PER_ATOM,
    MAX_EXCHANGE_EV_PER_UB2,
    MAX_STEP_UB,
    OK,
    WARNING_NOT_CONVERGED,
    _descend,
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

    Before the gain check, the old LBFGS retry stopped at 4.21 uB, 22.7
    eV/atom below the zero-moment energy, and passed as converged.
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

    accepted = result.status == OK
    assert not (accepted and np.abs(result.moments[:, 2]).max() > 4.0)
    if result.status == INVALID:
        assert any("magnetic energy gain" in reason for reason in result.reasons)


def _double_well(z: np.ndarray) -> tuple[float, np.ndarray]:
    """Per site, a well near 1 uB, a barrier near 2 uB and a deeper well
    near 3 uB, with gradients as steep as Fe's on the backbone (tens of
    eV/uB). Without the step cap, the descent jumps the barrier."""
    energy = 50 * ((z - 1) ** 2 * (z - 3) ** 2 - 0.5 * (z - 1))
    gradient = 50 * (2 * (z - 1) * (z - 3) * (2 * z - 4) - 0.5)
    return float(energy.sum()), gradient


def _recording(evaluate):
    calls: list[tuple[np.ndarray, float]] = []

    def wrapped(z: np.ndarray) -> tuple[float, np.ndarray]:
        energy, gradient = evaluate(z)
        calls.append((np.array(z), energy))
        return energy, gradient

    return wrapped, calls


def test_the_descent_stays_in_the_basin_of_its_seed() -> None:
    """goldilocks-ml#143: LBFGS's line search jumped the barrier into the
    deeper, unphysical well. The capped, energy-decreasing descent cannot."""
    moments, energy, gradient = _descend(_double_well, np.array([0.2, 0.5, 0.8]))

    assert np.all(moments < 1.5)
    assert np.abs(gradient).max() < GRADIENT_TOLERANCE
    assert energy == pytest.approx(_double_well(moments)[0])


def test_every_accepted_step_is_capped_and_lowers_the_energy() -> None:
    evaluate, calls = _recording(_double_well)

    _, energy, _ = _descend(evaluate, np.array([0.2, 0.5, 0.8]))

    accepted_z, accepted_energy = calls[0]
    for z, trial_energy in calls[1:]:
        assert np.abs(z - accepted_z).max() <= MAX_STEP_UB + 1e-12
        if trial_energy < accepted_energy:
            accepted_z, accepted_energy = z, trial_energy
    assert energy == accepted_energy


def test_a_cell_and_its_supercell_relax_alike() -> None:
    """The step cap and the convergence test are per site, so replicating a
    cell does not change the path each site takes."""
    small, small_energy, _ = _descend(_double_well, np.full(2, 0.2))
    large, large_energy, _ = _descend(_double_well, np.full(16, 0.2))

    np.testing.assert_allclose(large, small[0], rtol=0, atol=1e-12)
    assert large_energy / 16 == pytest.approx(small_energy / 2, abs=1e-12)


def test_a_stiff_well_converges_to_the_tolerance() -> None:
    def stiff(z: np.ndarray) -> tuple[float, np.ndarray]:
        return float((100 * (z - 3.88) ** 2).sum()), 200 * (z - 3.88)

    moments, _, gradient = _descend(stiff, np.array([3.6, 3.6]))

    assert np.abs(gradient).max() < GRADIENT_TOLERANCE
    np.testing.assert_allclose(moments, 3.88, atol=1e-4)


@pytest.fixture
def tiny_relax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``relax()`` on the random-weight backbone instead of a checkpoint."""
    backbone = tiny_magnetic_backbone()
    monkeypatch.setattr(relax_module, "safe_load", lambda path, device: backbone)


def test_relax_moves_only_the_z_components(tiny_relax: None) -> None:
    import torch

    structure = Structure(
        Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    moments = np.array([[0.1, -0.2, 1.0], [0.0, 0.3, 1.2]])

    result = relax(structure, moments, checkpoint=Path("tiny"), device="cpu")

    np.testing.assert_array_equal(result.moments[:, :2], moments[:, :2])
    assert not np.array_equal(result.moments[:, 2], moments[:, 2])
    assert np.isfinite(result.energy_ev)
    assert torch.get_default_dtype() == torch.float32


def test_an_unfinished_relaxation_is_reported_as_not_converged(
    tiny_relax: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(relax_module, "MAX_STEPS", 0)
    structure = Structure(
        Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    moments = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    result = relax(structure, moments, checkpoint=Path("tiny"), device="cpu")

    assert result.status == WARNING_NOT_CONVERGED
    assert "largest magnetic gradient" in result.reasons[0]


def _bcc_fe() -> Structure:
    return Structure(
        Lattice.cubic(2.8665), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )


def test_fm_bcc_fe_relaxes_alike_in_a_cell_and_its_supercell() -> None:
    """goldilocks-ml#143: the 2-atom cell came back invalid at 4.21 uB while
    its 2x2x2 supercell converged at 3.88 uB, so goldilocks-core discarded
    the ferromagnetic candidate."""
    checkpoint = _real_checkpoint()
    limits = domain_limits(safe_load(checkpoint))
    cell, supercell = _bcc_fe(), _bcc_fe() * (2, 2, 2)

    small = relax(
        cell,
        seed_moments_fm_fim(cell, max_moment_by_symbol=limits),
        checkpoint=checkpoint,
    )
    large = relax(
        supercell,
        seed_moments_fm_fim(supercell, max_moment_by_symbol=limits),
        checkpoint=checkpoint,
    )

    assert small.status == OK
    assert large.status == OK
    np.testing.assert_allclose(small.moments[:, 2], 3.8797, atol=1e-3)
    assert small.energy_ev / 2 == pytest.approx(-10.86172, abs=1e-4)
    np.testing.assert_allclose(large.moments[:, 2], small.moments[0, 2], atol=1e-6)
    assert large.energy_ev / 16 == pytest.approx(small.energy_ev / 2, abs=1e-6)


def test_fm_bcc_fe_lies_below_its_b2_antiferromagnet() -> None:
    checkpoint = _real_checkpoint()
    structure = _bcc_fe()
    seed = seed_moments_fm_fim(
        structure, max_moment_by_symbol=domain_limits(safe_load(checkpoint))
    )
    antiparallel = seed * np.array([[1.0], [-1.0]])

    fm = relax(structure, seed, checkpoint=checkpoint)
    afm = relax(structure, antiparallel, checkpoint=checkpoint)

    assert fm.status == OK
    assert afm.status == OK
    assert np.sign(afm.moments[:, 2]).tolist() == [1.0, -1.0]
    assert fm.energy_ev < afm.energy_ev
