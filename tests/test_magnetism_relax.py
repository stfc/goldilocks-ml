"""End-to-end test for the collinear FM/FiM moment relaxation.

Deliberately mace-free except for this one skip-guarded test: the same
pattern ``tests/test_magnetism_mace_mlp.py`` uses for the classifier's own
real-backbone test, and for the same reason -- the ~80 MB checkpoint and the
manually-installed mace/e3nn/sphericart stack (see "Use the is_magnetic
classifier" in README.md) are not present in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
    OK,
    relax,
)


def _real_checkpoint() -> Path:
    checkpoint = Path(
        "local_data/artifacts/UNPUBLISHED-PENDING-LICENCE/"
        "mace_matpes_pbe_baseline_run-3.model"
    )
    if not checkpoint.is_file():
        pytest.skip("the mMACE backbone checkpoint is not present locally")
    try:
        import mace  # noqa: F401
    except ImportError:
        pytest.skip("mace is not installed")
    return checkpoint


def test_relax_restores_torchs_global_default_dtype() -> None:
    """Regression test for goldilocks-ml#98.

    ``MagneticMACECalculator`` without ``default_dtype`` makes mace
    auto-detect the checkpoint's dtype and call the process-wide, unscoped
    ``torch.set_default_dtype(float64)`` -- confirmed to otherwise leak past
    ``relax()`` returning and break every later float32 prediction (e.g. the
    CGCNN ``is_metal`` classifier) in the same process. Checks both the
    global directly and that a real float32 embedding call still works
    right after ``relax()`` returns.
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
