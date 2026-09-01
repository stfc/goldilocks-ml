"""Convert the Matbench ``mp_is_metal`` dataset into a sealed snapshot.

The dataset is 106113 Materials Project structures labelled by whether DFT
gives them a zero band gap. Matminer distributes it with a published digest,
so the source is pinned without needing a Materials Project API key.

Sample ids are content digests of the structure, not row numbers: the same
crystal always gets the same id, and re-running this script on a re-downloaded
dataset produces the same snapshot. Groups are reduced formulae, so polymorphs
and differently sized cells of one composition cannot be split across the
train and test sets.

    uv run --extra models python scripts/matbench_to_snapshot.py \
        --output local_data/snapshots/mp-is-metal
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

DATASET = "matbench_mp_is_metal"
TARGET_CONTRACT = "goldilocks.is_metal.dft_band_gap_zero.v1"
TARGET_DEFINITION = (
    "True when the Materials Project DFT band gap is zero, taken from the "
    "Matbench v0.1 mp_is_metal task. 'metal' is the positive class."
)
METAL = "metal"
INSULATOR = "insulator"


def sample_id(cif: str) -> str:
    """Return a stable id derived from the structure's own CIF text."""
    return "mpm-" + hashlib.sha256(cif.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    """Write and seal the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-home",
        type=Path,
        default=Path("local_data/raw/matbench"),
        help="where matminer caches the downloaded dataset",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="take only the first N rows"
    )
    parser.add_argument("--snapshot-version", default="v1")
    arguments = parser.parse_args()

    from matminer.datasets import load_dataset
    from pymatgen.io.cif import CifWriter

    from goldilocks_ml.cli import seal

    frame = load_dataset(DATASET, data_home=str(arguments.data_home))
    if arguments.limit is not None:
        frame = frame.iloc[: arguments.limit]

    directory = arguments.output
    directory.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    duplicates = 0
    for structure, is_metal in zip(frame["structure"], frame["is_metal"], strict=True):
        cif = str(CifWriter(structure))
        identifier = sample_id(cif)
        if identifier in seen:
            # Content-addressed ids make duplicate crystals collide by
            # construction, which is the behaviour we want: keep one.
            duplicates += 1
            continue
        seen.add(identifier)
        (directory / f"{identifier}.cif").write_text(cif, encoding="utf-8")
        rows.append(
            (
                identifier,
                METAL if bool(is_metal) else INSULATOR,
                structure.composition.reduced_formula,
            )
        )

    with (directory / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    digest = seal(
        directory,
        record_id=DATASET,
        snapshot_version=arguments.snapshot_version,
        structure_suffix=".cif",
        target_name="is_metal",
        target_contract=TARGET_CONTRACT,
        target_definition=TARGET_DEFINITION,
        target_units=None,
    )["manifest_sha256"]

    labels = Counter(label for _, label, _ in rows)
    groups = len({group for _, _, group in rows})
    print(f"samples:    {len(rows)}")
    print(f"duplicates: {duplicates}")
    print(f"labels:     {dict(labels)}")
    print(f"groups:     {groups}")
    print(f"manifest:   {digest}")


if __name__ == "__main__":
    main()
