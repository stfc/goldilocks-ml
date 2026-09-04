"""Convert PSDI ``d5ds2-64f16`` into a sealed k-index snapshot.

The published record contains one CSV of converged labels and an archive with
both converged and unconverged structures.  This converter selects exactly the
labelled structures, groups them by reduced composition, and seals every output
file with SHA-256 before training can read it.

    uv run --extra models python scripts/psdi_kindex_to_snapshot.py \
        --source /path/to/downloaded-record \
        --output local_data/snapshots/kindex-d5ds2-64f16
"""

from __future__ import annotations

import argparse
import csv
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

from goldilocks_ml.cli import seal
from goldilocks_ml.hashing import sha256_file

RECORD_ID = "d5ds2-64f16"
SNAPSHOT_VERSION = "v1"
SUMMARY_FILE = "convergence_summary.csv"
ARCHIVE_FILE = "CIF_files.tar.gz"
SUMMARY_SHA256 = "83c1ca3afcd794e86f9b514d2425f1303f1d51f05ad3ee748f32e75bea0e8c5f"
ARCHIVE_SHA256 = "4f11f7c424e93af8c7074e55978eacd91ec618b79b09356f4586e279e79e1544"
TARGET_CONTRACT = "goldilocks.k_index.ladder_0based.max50.v1"
TARGET_DEFINITION = (
    "Zero-based rung on the structure-specific k-mesh ladder; rung 0 is the "
    "Gamma-only (1, 1, 1) mesh and change points were enumerated to 50 "
    "k-points per reciprocal-lattice axis."
)
REQUIRED_COLUMNS = ("source_db_id", "k_index", "k_dist_interval", "k_mesh")


def _verify_digest(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path.name} SHA-256 is {actual}; expected {expected}")


def _labels(path: Path) -> list[tuple[str, int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(REQUIRED_COLUMNS):
            raise ValueError(
                f"{path} columns are {reader.fieldnames!r}; expected "
                f"{list(REQUIRED_COLUMNS)!r}"
            )
        rows: list[tuple[str, int]] = []
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            sample_id = row["source_db_id"].strip()
            if (
                not sample_id
                or PurePosixPath(sample_id).name != sample_id
                or sample_id in {".", ".."}
            ):
                raise ValueError(f"{path}:{line} has unsafe source_db_id {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"{path}:{line} repeats source_db_id {sample_id!r}")
            seen.add(sample_id)
            try:
                target = int(row["k_index"])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line} has non-integer k_index {row['k_index']!r}"
                ) from error
            if target < 0:
                raise ValueError(f"{path}:{line} has negative k_index {target}")
            rows.append((sample_id, target))
    if not rows:
        raise ValueError(f"{path} contains no labels")
    return rows


def _archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"archive contains unsafe path {member.name!r}")
        if member.isdir() and path == PurePosixPath("CIF_files"):
            continue
        if (
            not member.isfile()
            or len(path.parts) != 2
            or path.parts[0] != "CIF_files"
            or path.suffix.lower() != ".cif"
        ):
            raise ValueError(f"archive contains unexpected member {member.name!r}")
        sample_id = path.stem
        if sample_id in members:
            raise ValueError(f"archive repeats structure {sample_id!r}")
        members[sample_id] = member
    return members


def convert(
    source: Path,
    output: Path,
    *,
    expected_summary_sha256: str = SUMMARY_SHA256,
    expected_archive_sha256: str = ARCHIVE_SHA256,
    record_id: str = RECORD_ID,
    snapshot_version: str = SNAPSHOT_VERSION,
) -> dict[str, object]:
    """Write the labelled flat snapshot and return conversion statistics."""
    from pymatgen.core import Structure

    source = source.resolve()
    output = output.resolve()
    summary = source / SUMMARY_FILE
    archive_path = source / ARCHIVE_FILE
    _verify_digest(summary, expected_summary_sha256)
    _verify_digest(archive_path, expected_archive_sha256)
    rows = _labels(summary)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty snapshot {output}")
    output.mkdir(parents=True, exist_ok=True)

    id_rows: list[tuple[str, int, str]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = _archive_members(archive)
        missing = [sample_id for sample_id, _ in rows if sample_id not in members]
        if missing:
            raise ValueError(
                f"archive lacks {len(missing)} labelled structures, starting with "
                f"{missing[0]}"
            )
        for sample_id, target in rows:
            extracted = archive.extractfile(members[sample_id])
            if extracted is None:
                raise ValueError(f"could not read structure {sample_id!r}")
            payload = extracted.read()
            structure = Structure.from_str(payload.decode("utf-8"), fmt="cif")
            group = structure.composition.reduced_formula
            (output / f"{sample_id}.cif").write_bytes(payload)
            id_rows.append((sample_id, target, group))

    with (output / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(id_rows)

    sealed = seal(
        output,
        record_id=record_id,
        snapshot_version=snapshot_version,
        structure_suffix=".cif",
        target_name="k_index",
        target_contract=TARGET_CONTRACT,
        target_definition=TARGET_DEFINITION,
        target_units=None,
    )
    return {
        "samples": len(id_rows),
        "groups": len({group for _, _, group in id_rows}),
        "labels": dict(sorted(Counter(target for _, target, _ in id_rows).items())),
        "manifest_sha256": sealed["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = convert(arguments.source, arguments.output)
    print(f"samples:  {result['samples']}")
    print(f"groups:   {result['groups']}")
    print(f"labels:   {result['labels']}")
    print(f"manifest: {result['manifest_sha256']}")


if __name__ == "__main__":
    main()
