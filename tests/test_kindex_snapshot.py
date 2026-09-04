"""Tests for conversion of the published k-index dataset into a snapshot."""

from __future__ import annotations

import csv
import io
import json
import runpy
import tarfile
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from goldilocks_ml.hashing import sha256_file

SCRIPT = Path(__file__).parents[1] / "scripts/psdi_kindex_to_snapshot.py"
SCRIPT_NAMESPACE = runpy.run_path(str(SCRIPT))
ARCHIVE_FILE = SCRIPT_NAMESPACE["ARCHIVE_FILE"]
SUMMARY_FILE = SCRIPT_NAMESPACE["SUMMARY_FILE"]
TARGET_CONTRACT = SCRIPT_NAMESPACE["TARGET_CONTRACT"]
_archive_members = SCRIPT_NAMESPACE["_archive_members"]
convert = SCRIPT_NAMESPACE["convert"]


def _source(directory: Path) -> tuple[str, str]:
    directory.mkdir()
    rows = [
        ("mc3d-a", 0, "[1.0, 0.5)", "(1, 1, 1)"),
        ("mc3d-b", 4, "[0.2, 0.1)", "(4, 4, 4)"),
    ]
    with (directory / SUMMARY_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_db_id", "k_index", "k_dist_interval", "k_mesh"))
        writer.writerows(rows)

    structures = {
        "mc3d-a": Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]]),
        "mc3d-b": Structure(Lattice.cubic(4.0), ["Ge"], [[0, 0, 0]]),
        "unconverged": Structure(Lattice.cubic(5.0), ["C"], [[0, 0, 0]]),
    }
    staging = directory / "staging"
    staging.mkdir()
    for sample_id, structure in structures.items():
        (staging / f"{sample_id}.cif").write_text(
            str(CifWriter(structure)), encoding="utf-8"
        )
    with tarfile.open(directory / ARCHIVE_FILE, "w:gz") as archive:
        for path in sorted(staging.iterdir()):
            archive.add(path, arcname=f"CIF_files/{path.name}")
    return (
        sha256_file(directory / SUMMARY_FILE),
        sha256_file(directory / ARCHIVE_FILE),
    )


def test_converter_keeps_only_labelled_structures_and_seals_groups(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    summary_digest, archive_digest = _source(source)
    output = tmp_path / "snapshot"

    result = convert(
        source,
        output,
        expected_summary_sha256=summary_digest,
        expected_archive_sha256=archive_digest,
        record_id="fixture-kindex",
    )

    assert result["samples"] == 2
    assert result["groups"] == 2
    assert result["labels"] == {0: 1, 4: 1}
    assert not (output / "unconverged.cif").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["record_id"] == "fixture-kindex"
    assert manifest["target"]["contract"] == TARGET_CONTRACT
    assert len(manifest["files"]) == 3
    assert (output / "id_prop.csv").read_text(encoding="utf-8").splitlines() == [
        "mc3d-a,0,Si",
        "mc3d-b,4,Ge",
    ]


def test_archive_validation_rejects_parent_traversal(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("../outside.cif")
        payload = b"unsafe"
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(path, "r:gz") as archive:
        with pytest.raises(ValueError, match="unsafe path"):
            _archive_members(archive)
