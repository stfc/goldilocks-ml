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


def test_relax_holds_moments_collinear_on_the_real_backbone() -> None:
    """A regression test for goldilocks-ml#92: ``ac8ff476`` had no gradient-
    masking mechanism at all, so relaxing against it either raised
    ``TypeError`` (with ``use_collinear``/``constrain_magnitude`` passed) or,
    had those kwargs been silently dropped instead, would have let the x/y
    components drift away from zero -- a collinear seed relaxing into a
    non-collinear result without anyone noticing. This checks both that it
    runs and that x/y stay exactly at zero.
    """
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
