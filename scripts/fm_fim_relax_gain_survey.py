"""Relax MatPES structures and record each result's magnetic energy gain.

The gain is (E_relaxed - E_zero) / natoms, with E_zero the energy at zero
moments from one non-SCF forward pass. Each line of the output also carries
the relaxed and DFT site moments, so results that reproduce DFT can be told
apart. ``fm_fim_relax.relax``'s exchange bound on the gain
(``MAX_EXCHANGE_EV_PER_UB2``) was tested on this survey, run before that check
existed (goldilocks-ml#120).

It relaxes the ``relax`` structures of ``structures.json`` (from
``scripts/mmace_parity/select_structures.py``), 100 magnetic and 30
non-magnetic MatPES test-split structures of at most 24 atoms, drawn with
seed 120:

    uv run python scripts/fm_fim_relax_gain_survey.py \
        path/to/matpes_pbe_material_records.json.gz structures.json \
        BACKBONE gain.jsonl
"""

from __future__ import annotations

import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from goldilocks_ml.models.magnetism._mace_backbone import safe_load
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
    _zero_moment_energy,
    domain_limits,
    relax,
)
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.seed_moments import (
    seed_moments_fm_fim,
)

MAGNETIC_SAMPLE = 100
NON_MAGNETIC_SAMPLE = 30
SEED = 120
MAX_ATOMS = 24

records_path, structures_path, backbone, out = map(Path, sys.argv[1:5])
raw = safe_load(backbone)
limits = domain_limits(raw)


def label(m: float) -> str | None:
    return "non_magnetic" if m <= 0.05 else ("magnetic" if m >= 0.5 else None)


parity = [e for e in json.loads(structures_path.read_text()) if e.get("relax")]
parity_ids = {e["id"] for e in parity}
records = json.load(gzip.open(records_path))
test = sorted(
    (
        r
        for r in records
        if r["split"] == "test"
        and len(r["structure"]["sites"]) <= MAX_ATOMS
        and r["matpes_id"] not in parity_ids
    ),
    key=lambda r: r["matpes_id"],
)
rng = random.Random(SEED)
magnetic = rng.sample(
    [r for r in test if label(r["max_abs_magmom"]) == "magnetic"], MAGNETIC_SAMPLE
)
seedable = [
    r
    for r in test
    if label(r["max_abs_magmom"]) == "non_magnetic"
    and seed_moments_fm_fim(
        Structure.from_dict(r["structure"]), max_moment_by_symbol=limits
    )[:, 2].any()
]
non_magnetic = rng.sample(seedable, NON_MAGNETIC_SAMPLE)

entries = [(e["id"], "parity", e["structure"], e.get("seed")) for e in parity]
entries += [(r["matpes_id"], "magnetic", r["structure"], None) for r in magnetic]
entries += [
    (r["matpes_id"], "non_magnetic", r["structure"], None) for r in non_magnetic
]

with out.open("w") as handle:
    for key, group, structure_dict, given_seed in entries:
        structure = Structure.from_dict(structure_dict)
        seed = (
            np.array(given_seed, dtype=float)
            if given_seed is not None
            else seed_moments_fm_fim(structure, max_moment_by_symbol=limits)
        )
        started = time.perf_counter()
        result = relax(structure, seed, checkpoint=backbone)
        zero_energy = _zero_moment_energy(
            AseAtomsAdaptor.get_atoms(structure), raw_model=raw, device="cpu"
        )
        dft = [abs(float(site.properties.get("magmom", 0.0))) for site in structure]
        row = {
            "id": key,
            "group": group,
            "formula": structure.composition.reduced_formula,
            "natoms": len(structure),
            "status": result.status,
            "reasons": list(result.reasons),
            "energy_ev": result.energy_ev,
            "zero_energy_ev": zero_energy,
            "gain_ev_per_atom": (result.energy_ev - zero_energy) / len(structure),
            "relaxed_abs_mz": np.abs(result.moments[:, 2]).tolist(),
            "seed_mz": seed[:, 2].tolist(),
            "dft_abs_m": dft,
            "seconds": time.perf_counter() - started,
        }
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        print(
            f"{key} {group} {result.status} gain={row['gain_ev_per_atom']:.4g} "
            f"in {row['seconds']:.1f}s",
            flush=True,
        )
