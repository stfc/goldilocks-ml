"""Relax a collinear initial moment guess on a frozen mMACE potential energy surface.

Ported from ``2-research/2-mace/1-magnetic-mace/scripts/run_all_cifs_magnetic_scf.py``
(``model_limits``, ``validity_checks``, the core of ``run_one``) and the
two-attempt retry policy in ``notebooks/01_basic_magnetic_scf.ipynb``'s batch
loop. Only atomic positions are fixed and only the per-atom moment vectors are
optimised (:class:`mace.modules.MagneticSCFMACE`, ``use_collinear=True``) --
this is not a structure relaxation, and this project's own tooling never
exercises the non-collinear path the backbone itself supports.

What is dropped on the port: the CSV/XYZ writing and the DFT-comparison-table
preparation in the source script are reporting glue for one ten-material
study, not general library code. What is new: the backbone is loaded through
:func:`goldilocks_ml.models.magnetism._mace_backbone.safe_load`, which the
source classifier code uses but this one script did not, for the same reason
the ported ``_mace_backbone`` module exists at all -- one loading path, not
three copies of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from goldilocks_ml.models.magnetism._mace_backbone import (
    MAGMOM_KEY,
    _default_dtype_float64,
    safe_load,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

# The source project's own retry policy: a conservative retry after a step
# size this large fails to converge cleanly (notebooks/01, cell 17).
DEFAULT_STEP_SIZE = 0.01
DEFAULT_N_SCF_STEP = 50
RETRY_STEP_SIZE = 0.001
RETRY_N_SCF_STEP = 200
# A gradient norm above this after the SCF loop stops is reported as
# unconverged rather than accepted silently.
GRADIENT_NORM_TOLERANCE = 1e-2
# A relaxed moment more than this multiple of the backbone's own learned
# per-element moment-magnitude domain is treated as unphysical.
DOMAIN_VIOLATION_FACTOR = 1.05
# `abs(energy) / natoms` above this is treated as a diverged relaxation
# rather than a real total energy.
MAX_ENERGY_PER_ATOM_EV = 100.0

OK = "ok"
CONVERGED_AFTER_RETRY = "converged_after_retry"
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


def _attempt(
    structure: Structure,
    initial_moments: np.ndarray,
    *,
    raw_model: Any,
    device: str,
    step_size: float,
    n_scf_step: int,
) -> MomentRelaxationResult:
    from mace.calculators.mace import MagneticMACECalculator
    from mace.modules import MagneticSCFMACE
    from pymatgen.io.ase import AseAtomsAdaptor

    scf_model = MagneticSCFMACE(
        raw_model,
        use_scf=True,
        use_collinear=True,
        n_scf_step=n_scf_step,
        scf_step_size=step_size,
        scf_tol=1e-4,
        scf_logging=False,
    )
    limits = domain_limits(raw_model)

    atoms = AseAtomsAdaptor.get_atoms(structure)
    symbols = atoms.get_chemical_symbols()
    atoms.arrays[MAGMOM_KEY] = np.array(initial_moments, dtype=float, copy=True)

    # The SCF model's last forward pass evaluates the returned moments. Its
    # magnetic forces give the final gradient, which the calculator does not
    # return.
    final_output: dict[str, Any] = {}
    hook = scf_model.register_forward_hook(
        lambda module, inputs, output: final_output.update(output)
    )
    # Runs under `_mace_lock`, one at a time with every other mace call in the
    # process -- see `_default_dtype_float64` and goldilocks-ml#98.
    try:
        with _default_dtype_float64():
            calculator = MagneticMACECalculator(
                models=[scf_model],
                device=device,
                default_dtype="float64",
                magmom_key=MAGMOM_KEY,
            )
            atoms.calc = calculator
            energy_ev = float(atoms.get_potential_energy())
    finally:
        hook.remove()
    moments = np.asarray(calculator.results["MACE_magmoms"], dtype=float).copy()
    reasons = _validity_reasons(energy_ev, moments, symbols, limits)
    # `use_collinear` zeroes the x/y gradient, so only z counts.
    magforces = final_output["magforces"].detach().cpu().numpy()
    final_grad = float(np.linalg.norm(magforces[:, 2]))

    if reasons:
        status = INVALID
    elif np.isfinite(final_grad) and final_grad > GRADIENT_NORM_TOLERANCE:
        status = WARNING_NOT_CONVERGED
        reasons.append(
            f"final magnetic gradient norm {final_grad:.6g} > {GRADIENT_NORM_TOLERANCE}"
        )
    else:
        status = OK
    return MomentRelaxationResult(
        moments=moments, energy_ev=energy_ev, status=status, reasons=tuple(reasons)
    )


def relax(
    structure: Structure,
    initial_moments: np.ndarray,
    *,
    checkpoint: Path,
    device: str = "cpu",
) -> MomentRelaxationResult:
    """Relax one collinear initial moment guess, retrying once at a smaller step.

    Mirrors the source project's own batch policy: an attempt at
    ``scf_step_size=0.01, n_scf_step=50`` that fails validity or the gradient
    norm floor is retried once at ``0.001, 200``. The retry's own outcome is
    returned whether or not it improves things -- this never tries a third
    configuration.
    """
    raw_model = safe_load(Path(checkpoint), device)

    first = _attempt(
        structure,
        initial_moments,
        raw_model=raw_model,
        device=device,
        step_size=DEFAULT_STEP_SIZE,
        n_scf_step=DEFAULT_N_SCF_STEP,
    )
    if first.status == OK:
        return first

    retry = _attempt(
        structure,
        initial_moments,
        raw_model=raw_model,
        device=device,
        step_size=RETRY_STEP_SIZE,
        n_scf_step=RETRY_N_SCF_STEP,
    )
    if retry.status == OK:
        return MomentRelaxationResult(
            moments=retry.moments,
            energy_ev=retry.energy_ev,
            status=CONVERGED_AFTER_RETRY,
            reasons=retry.reasons,
        )
    return retry
