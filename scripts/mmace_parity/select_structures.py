"""Pick the structures for the mMACE backbone parity report.

Reads the MatPES PBE records file the is_magnetic classifier was trained from
(one JSON object per material, with its split, max |site moment| and
structure) and writes the structures both environments run.

    uv run python scripts/mmace_parity/select_structures.py \
        path/to/matpes_pbe_material_records.json.gz structures.json
"""

from __future__ import annotations

import gzip
import json
import random
import sys

from pymatgen.core import Lattice, Structure

RELAX_SAMPLE = 14
SEED = 117

records = json.load(gzip.open(sys.argv[1]))
by_id = {r["matpes_id"]: r for r in records}


def label(max_abs_magmom: float) -> str | None:
    if max_abs_magmom <= 0.05:
        return "non_magnetic"
    if max_abs_magmom >= 0.5:
        return "magnetic"
    return None


entries = []

# The cases VENDORING_TODO.md recorded: bcc Fe, diamond Si and fcc Pt built by
# hand, and bcc Fe relaxed from a [0, 0, 2.2] seed.
fe_bcc = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
entries += [
    {
        "id": "hand:Fe-bcc",
        "group": "hand",
        "structure": fe_bcc.as_dict(),
        "embed": True,
        "relax": True,
        "seed": [[0, 0, 2.2], [0, 0, 2.2]],
    },
    {
        "id": "hand:Si-diamond",
        "group": "hand",
        "embed": True,
        "structure": Structure(
            Lattice([[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]]),
            ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        ).as_dict(),
    },
    {
        "id": "hand:Pt-fcc",
        "group": "hand",
        "embed": True,
        "structure": Structure(
            Lattice.cubic(3.92),
            ["Pt"] * 4,
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        ).as_dict(),
    },
]

# Well-known MatPES entries: bcc and fcc Fe, Ni, Co, Si, Pt, NiO, Fe2O3.
for matpes_id, relax_it in [
    ("mp-13", True),
    ("mp-150", True),
    ("mp-23", True),
    ("mp-54", True),
    ("mp-149", False),
    ("mp-126", False),
    ("mp-19009", True),
    ("mp-19770", True),
]:
    record = by_id[matpes_id]
    entries.append(
        {
            "id": matpes_id,
            "group": "named",
            "formula": record["formula"],
            "label": label(record["max_abs_magmom"]),
            "structure": record["structure"],
            "embed": True,
            "relax": relax_it,
        }
    )

# Every labelled structure in the test split, for embeddings and scores.
test = sorted(
    (r for r in records if r["split"] == "test" and label(r["max_abs_magmom"])),
    key=lambda r: r["matpes_id"],
)
relax_pool = [
    r
    for r in test
    if label(r["max_abs_magmom"]) == "magnetic" and len(r["structure"]["sites"]) <= 24
]
relax_ids = {
    r["matpes_id"] for r in random.Random(SEED).sample(relax_pool, RELAX_SAMPLE)
}
for record in test:
    entries.append(
        {
            "id": record["matpes_id"],
            "group": "test",
            "formula": record["formula"],
            "label": label(record["max_abs_magmom"]),
            "structure": record["structure"],
            "embed": True,
            "relax": record["matpes_id"] in relax_ids,
        }
    )

json.dump(entries, open(sys.argv[2], "w"))
print(
    len(entries), "entries;", sum(bool(e.get("relax")) for e in entries), "relax cases"
)
