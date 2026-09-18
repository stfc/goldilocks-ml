"""Tests for the FM/ferrimagnetic seed-moment guess.

Deliberately mace-free: this exercises only the oxidation-state guessing and
clamping logic, which is plain pymatgen, so it must not need the ``magnetism``
extra installed to run.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.seed_moments import (
    high_spin_moment,
    ionic_shell_occupancy,
    seed_moments_fm_fim,
)


def test_high_spin_moment_is_the_smaller_of_filled_and_empty_slots() -> None:
    assert high_spin_moment("d", 0) == pytest.approx(0.0)
    assert high_spin_moment("d", 5) == pytest.approx(5.0)
    assert high_spin_moment("d", 10) == pytest.approx(0.0)
    assert high_spin_moment("f", 7) == pytest.approx(7.0)


def test_high_spin_moment_rejects_a_shell_other_than_d_or_f() -> None:
    with pytest.raises(ValueError, match="shell must be"):
        high_spin_moment("p", 3)


def test_high_spin_moment_rejects_an_occupancy_outside_the_shell() -> None:
    with pytest.raises(ValueError, match="invalid occupancy"):
        high_spin_moment("d", 11)


def test_ionic_shell_occupancy_matches_textbook_fe_oxidation_states() -> None:
    # Fe2+ is the textbook high-spin d6 (4 unpaired electrons); Fe3+ is d5
    # (5 unpaired) -- both are standard enough to pin down as regression tests.
    assert ionic_shell_occupancy("Fe", 2.0) == ("d", pytest.approx(6.0))
    assert ionic_shell_occupancy("Fe", 3.0) == ("d", pytest.approx(5.0))


def test_ionic_shell_occupancy_is_none_without_a_d_or_f_shell() -> None:
    assert ionic_shell_occupancy("Na", 1.0) is None


def test_seed_moments_fm_fim_only_populates_z_and_only_for_known_elements() -> None:
    sites = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(Lattice.cubic(2.87), ["Fe", "Si"], sites)

    moments = seed_moments_fm_fim(structure, max_moment_by_symbol={"Fe": 5.0})

    assert moments.shape == (2, 3)
    # Silicon has no entry in the limit mapping, so it is left at zero rather
    # than guessed past a domain the caller never declared.
    assert moments[1] == pytest.approx([0.0, 0.0, 0.0])
    assert moments[0, 0] == pytest.approx(0.0)
    assert moments[0, 1] == pytest.approx(0.0)
    assert moments[0, 2] > 0.0


def test_seed_moments_fm_fim_never_exceeds_the_domain_safety_clamp() -> None:
    structure = Structure(Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]])
    tight_limit = 1.0

    moments = seed_moments_fm_fim(structure, max_moment_by_symbol={"Fe": tight_limit})

    assert 0.0 < moments[0, 2] <= 0.7 * tight_limit + 1e-9


def test_seed_moments_fm_fim_assigns_one_sign_per_element() -> None:
    """FM/ferrimagnetic by construction: one element never splits into two signs."""
    structure = Structure(
        Lattice.cubic(4.0),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    moments = seed_moments_fm_fim(structure, max_moment_by_symbol={"Fe": 5.0})

    assert np.sign(moments[0, 2]) == np.sign(moments[1, 2])
    assert moments[0, 2] == pytest.approx(moments[1, 2])
