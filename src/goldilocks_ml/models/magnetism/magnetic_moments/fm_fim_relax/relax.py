"""Relax a collinear initial moment guess on a frozen mMACE potential energy surface.

Atomic positions stay fixed; only the z component of each site's moment is
optimised, so a collinear seed stays collinear. This is not a structure
relaxation.

The optimiser is an L-BFGS descent on the backbone's energy, without a line
search. Each step moves no moment by more than ``MAX_STEP_UB`` and is
accepted only if the energy drops, so the descent cannot cross an energy
barrier: it ends in the basin its seed is in. That matters because the
backbone's energy surface breaks down inside its learned moment domain (for
Fe, above about 4.1 uB, where ``m_max`` is 5.15 uB), and a line search that
extrapolates can jump into that region (goldilocks-ml#143). It converges
when the largest per-atom |dE/dm| is below ``GRADIENT_TOLERANCE``, so a cell
and its supercell relax alike.

A relaxed result is ``invalid`` when its energy is non-finite or past
``MAX_ENERGY_PER_ATOM_EV``, when a moment leaves the backbone's learned domain,
or when it sits further below the same structure at zero moment than the
Stoner exchange energy allows (``MAX_EXCHANGE_EV_PER_UB2``, goldilocks-ml#120).

Ported from ``2-research/2-mace/1-magnetic-mace/scripts/run_all_cifs_magnetic_scf.py``
(``model_limits``, ``validity_checks``); the source's LBFGS loop and its
two-attempt retry policy were replaced by the descent above.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from goldilocks_ml.models.magnetism._mace_backbone import (
    MAGMOM_KEY,
    _atoms_to_batch,
    _default_dtype_float64,
    _zero_moment_forward,
    safe_load,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

# The descent: the largest change to any moment in one step (uB), the
# largest per-atom |dE/dm| (eV/uB) at which it stops as converged, the step
# length of a gradient step (uB^2/eV), how many past steps the L-BFGS
# direction uses, and how long it may run.
MAX_STEP_UB = 0.05
GRADIENT_TOLERANCE = 1e-2
INITIAL_STEP_LENGTH = 1e-2
LBFGS_HISTORY = 10
MAX_STEPS = 500
MAX_HALVINGS = 20
# A relaxed moment more than this multiple of the backbone's own learned
# per-element moment-magnitude domain is treated as unphysical.
DOMAIN_VIOLATION_FACTOR = 1.05
# `abs(energy) / natoms` above this is treated as a diverged relaxation
# rather than a real total energy.
MAX_ENERGY_PER_ATOM_EV = 100.0
# The most a relaxed configuration may sit below the same structure at zero
# moment: MAX_EXCHANGE_EV_PER_UB2 * sum(|m|^2) + MAGNETIC_GAIN_SLACK_EV_PER_ATOM
# * natoms. The first term is the Stoner exchange energy I/4 * m^2 with
# I = 1 eV, the top of the 3d range; the slack covers small moments. Past it,
# the backbone's energy surface has broken down -- see goldilocks-ml#120.
MAX_EXCHANGE_EV_PER_UB2 = 0.25
MAGNETIC_GAIN_SLACK_EV_PER_ATOM = 0.5

OK = "ok"
WARNING_NOT_CONVERGED = "warning_not_converged"
INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class MomentRelaxationResult:
    """One relaxed collinear moment configuration and how it got there."""

    moments: np.ndarray  # (N, 3), bohr magnetons
    energy_ev: float
    status: str
    reasons: tuple[str, ...] = ()


def domain_limits(raw_model: Any) -> dict[str, float]:
    """Return the backbone's learned per-element moment-magnitude domain.

    Keyed by chemical symbol -- the same shape
    :func:`~.seed_moments.seed_moments_fm_fim` expects for its
    ``max_moment_by_symbol`` argument, so a caller can seed and relax from
    the same loaded backbone.
    """
    from ase.data import chemical_symbols

    numbers = [int(z) for z in raw_model.atomic_numbers]
    limits = raw_model.m_max.detach().cpu().numpy()
    return {chemical_symbols[z]: float(m) for z, m in zip(numbers, limits, strict=True)}


def _validity_reasons(
    energy_ev: float,
    zero_moment_energy_ev: float,
    moments: np.ndarray,
    symbols: list[str],
    limits: dict[str, float],
) -> list[str]:
    """Return every reason a relaxed configuration is invalid, or ``[]``."""
    reasons = []
    natoms = len(moments)
    if not np.isfinite(energy_ev):
        reasons.append("non-finite energy")
    elif abs(energy_ev / natoms) > MAX_ENERGY_PER_ATOM_EV:
        reasons.append(
            f"unphysical |E/atom|={abs(energy_ev / natoms):.6g} "
            f"> {MAX_ENERGY_PER_ATOM_EV} eV"
        )
    elif np.isfinite(zero_moment_energy_ev) and np.isfinite(moments).all():
        gain = energy_ev - zero_moment_energy_ev
        bound = (
            MAX_EXCHANGE_EV_PER_UB2 * float(np.sum(np.square(moments)))
            + MAGNETIC_GAIN_SLACK_EV_PER_ATOM * natoms
        )
        if gain < -bound:
            reasons.append(
                f"magnetic energy gain {gain / natoms:.4g} eV/atom exceeds the "
                f"exchange bound of {bound / natoms:.4g} eV/atom"
            )
    if not np.isfinite(moments).all():
        reasons.append("non-finite magnetic moment")
    norms = np.linalg.norm(moments, axis=1)
    for index, (symbol, norm) in enumerate(zip(symbols, norms, strict=True)):
        limit = limits.get(symbol)
        exceeded = (
            limit is not None
            and np.isfinite(norm)
            and norm > DOMAIN_VIOLATION_FACTOR * limit
        )
        if exceeded:
            reasons.append(
                f"site {index} {symbol}: |m|={norm:.4g} > "
                f"{DOMAIN_VIOLATION_FACTOR}*m_max={DOMAIN_VIOLATION_FACTOR * limit:.4g}"
            )
    return reasons


def _zero_moment_energy(atoms: Any, *, raw_model: Any, device: str) -> float:
    """Return the energy of ``atoms`` at zero moment, from one forward pass."""
    return float(_zero_moment_forward(atoms, raw_model, device)["energy"].item())


def _descend(
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]], start: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Minimise ``evaluate`` from ``start`` by a capped, energy-decreasing L-BFGS.

    ``evaluate`` maps per-site z moments to the energy and its gradient. The
    step follows the L-BFGS direction over the last ``LBFGS_HISTORY`` steps
    (the gradient, scaled by ``INITIAL_STEP_LENGTH``, when there is none or
    it does not point downhill). It is scaled down so no site moves more
    than ``MAX_STEP_UB``, and halved until the energy drops. Returns the last
    accepted moments, energy and gradient.
    """
    moments = np.array(start, dtype=float)
    energy, gradient = evaluate(moments)
    steps: list[np.ndarray] = []
    changes: list[np.ndarray] = []
    for _ in range(MAX_STEPS):
        if np.abs(gradient).max() < GRADIENT_TOLERANCE:
            break
        step = -_inverse_hessian_times(gradient, steps, changes)
        if float(step @ gradient) >= 0:
            steps, changes = [], []
            step = -INITIAL_STEP_LENGTH * gradient
        largest = np.abs(step).max()
        if largest > MAX_STEP_UB:
            step *= MAX_STEP_UB / largest
        for _ in range(MAX_HALVINGS):
            trial_energy, trial_gradient = evaluate(moments + step)
            if np.isfinite(trial_energy) and trial_energy < energy:
                break
            step = step / 2
        else:
            break
        change = trial_gradient - gradient
        if float(step @ change) > 0:
            steps, changes = (
                [*steps, step][-LBFGS_HISTORY:],
                [*changes, change][-LBFGS_HISTORY:],
            )
        moments, energy, gradient = moments + step, trial_energy, trial_gradient
    return moments, energy, gradient


def _inverse_hessian_times(
    gradient: np.ndarray, steps: list[np.ndarray], changes: list[np.ndarray]
) -> np.ndarray:
    """The L-BFGS two-loop recursion: the inverse-Hessian estimate times
    ``gradient``, from past steps and the gradient changes they caused."""
    if not steps:
        return INITIAL_STEP_LENGTH * gradient
    q = gradient.copy()
    alphas = []
    for s, y in zip(reversed(steps), reversed(changes), strict=True):
        alpha = float(s @ q) / float(y @ s)
        q -= alpha * y
        alphas.append(alpha)
    r = q * float(steps[-1] @ changes[-1]) / float(changes[-1] @ changes[-1])
    for s, y, alpha in zip(steps, changes, reversed(alphas), strict=True):
        r += s * (alpha - float(y @ r) / float(y @ s))
    return r


def _energy_and_gradient(
    atoms: Any, moments: np.ndarray, raw_model: Any, device: str
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Return the backbone's energy (eV) and dE/dm_z (eV/uB) as a function of
    the per-site z moments, with x and y held at ``moments``'s."""
    import torch

    atoms = atoms.copy()
    atoms.arrays[MAGMOM_KEY] = np.array(moments, dtype=float)
    batch = _atoms_to_batch(atoms, raw_model, device)

    def evaluate(z: np.ndarray) -> tuple[float, np.ndarray]:
        with _default_dtype_float64(), torch.enable_grad():
            magmom = torch.as_tensor(moments, dtype=torch.float64, device=device)
            magmom = magmom.clone()
            magmom[:, 2] = torch.as_tensor(z, dtype=torch.float64, device=device)
            magmom.requires_grad_(True)
            output = raw_model(
                {**batch, "magmom": magmom},
                training=False,
                compute_force=False,
                compute_stress=False,
            )
            energy = output["energy"][0]
            (gradient,) = torch.autograd.grad(energy, magmom)
        return float(energy.detach()), gradient[:, 2].cpu().numpy()

    return evaluate


def relax(
    structure: Structure,
    initial_moments: np.ndarray,
    *,
    checkpoint: Path,
    device: str = "cpu",
) -> MomentRelaxationResult:
    """Relax one collinear initial moment guess on the backbone at ``checkpoint``.

    ``initial_moments`` is ``(N, 3)`` in bohr magnetons. Only the z components
    move. The result is ``ok`` when the descent converged, ``invalid`` when a
    validity check fails, and ``warning_not_converged`` otherwise.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    raw_model = safe_load(Path(checkpoint), device)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    initial = np.array(initial_moments, dtype=float, copy=True)

    evaluate = _energy_and_gradient(atoms, initial, raw_model, device)
    relaxed_z, energy_ev, gradient = _descend(evaluate, initial[:, 2])
    moments = initial.copy()
    moments[:, 2] = relaxed_z

    zero_moment_energy_ev = _zero_moment_energy(
        atoms, raw_model=raw_model, device=device
    )
    reasons = _validity_reasons(
        energy_ev,
        zero_moment_energy_ev,
        moments,
        atoms.get_chemical_symbols(),
        domain_limits(raw_model),
    )
    largest_gradient = float(np.abs(gradient).max())
    if reasons:
        status = INVALID
    elif not largest_gradient < GRADIENT_TOLERANCE:
        status = WARNING_NOT_CONVERGED
        reasons.append(
            f"largest magnetic gradient {largest_gradient:.6g} eV/uB "
            f"> {GRADIENT_TOLERANCE}"
        )
    else:
        status = OK
    return MomentRelaxationResult(
        moments=moments, energy_ev=energy_ev, status=status, reasons=tuple(reasons)
    )
