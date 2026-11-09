"""Evaluate one structure's energy at fixed collinear moments, without SCF.

Writes ``scan.json`` into OUT_DIR, beside the outputs of ``run_pipeline.py``,
for ``compare.py`` to put side by side. Every site gets the same moment along
z; the grid runs past the element's ``m_max``.

    python scripts/mmace_parity/scan_moments.py \\
        structures.json BACKBONE OUT_DIR mp-13
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mace.calculators.mace import MagneticMACECalculator
from mace.modules import MagneticSCFMACE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from goldilocks_ml.models.magnetism._mace_backbone import (
    MAGMOM_KEY,
    _default_dtype_float64,
    safe_load,
)
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
    domain_limits,
)

GRID = [round(3.6 + 0.1 * step, 1) for step in range(19)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structures", type=Path)
    parser.add_argument("backbone", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("id")
    args = parser.parse_args()

    entries = {e["id"]: e for e in json.loads(args.structures.read_text())}
    structure = Structure.from_dict(entries[args.id]["structure"])
    raw_model = safe_load(args.backbone)
    one_step_model = MagneticSCFMACE(
        raw_model, n_scf_step=1, use_scf=False, scf_tol=1e-4, scf_logging=False
    )
    with _default_dtype_float64():
        calculator = MagneticMACECalculator(
            models=[one_step_model],
            device="cpu",
            default_dtype="float64",
            magmom_key=MAGMOM_KEY,
        )
    output: dict = {}
    calculator.models[0].register_forward_hook(lambda m, i, o: output.update(o))

    rows = []
    for value in GRID:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        moments = np.zeros((len(atoms), 3))
        moments[:, 2] = value
        atoms.arrays[MAGMOM_KEY] = moments
        with _default_dtype_float64():
            calculator.reset()
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
        magforce = float(output["magforces"].detach()[0, 2])
        rows.append({"moment": value, "energy_ev": energy, "magforce_z": magforce})

    limits = domain_limits(raw_model)
    m_max = {site.specie.symbol: limits[site.specie.symbol] for site in structure}
    (args.out_dir / "scan.json").write_text(
        json.dumps({"id": args.id, "m_max": m_max, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
