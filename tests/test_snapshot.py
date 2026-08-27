"""Tests for immutable dataset snapshot verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    build_snapshot,
    classification_document,
    make_rows,
    pin,
    regression_document,
    write_protocol,
)

from goldilocks_ml.core.hashing import sha256_file
from goldilocks_ml.core.protocol import load_protocol
from goldilocks_ml.core.snapshot import Snapshot, load_snapshot


def _load(
    tmp_path: Path,
    snapshot: Path,
    *,
    classification: bool = False,
    **overrides: Any,
) -> Snapshot:
    document = (
        classification_document(**overrides)
        if classification
        else regression_document(**overrides)
    )
    protocol = load_protocol(write_protocol(tmp_path / "protocol.toml", document))
    return load_snapshot(snapshot, protocol)


def test_load_snapshot_returns_verified_samples(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)

    snapshot = _load(tmp_path, snapshot_dir)

    assert len(snapshot.samples) == 24
    assert snapshot.samples[0].sample_id == "syn-000"
    assert snapshot.samples[0].group == "grp-00"
    assert snapshot.samples[0].structure_path is None
    assert snapshot.capabilities == frozenset({"features", "groups"})


def test_structures_become_a_capability(tmp_path: Path, snapshot_dir: Path) -> None:
    build_snapshot(snapshot_dir, structures=True)

    snapshot = _load(tmp_path, snapshot_dir, dataset={"requires": ["structures"]})

    assert "structures" in snapshot.capabilities
    path = snapshot.samples[0].structure_path
    assert path is not None and path.name == "syn-000.cif"


def test_a_protocol_cannot_require_a_missing_capability(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir, structures=False)

    with pytest.raises(
        ValueError, match="capabilities this snapshot lacks: structures"
    ):
        _load(tmp_path, snapshot_dir, dataset={"requires": ["structures"]})


def test_group_splitting_needs_the_third_column(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir, groups=False)

    with pytest.raises(ValueError, match="needs a third column in id_prop.csv"):
        _load(tmp_path, snapshot_dir, split={"method": "group"})


def test_consecutive_integer_sample_ids_are_rejected(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    rows = [{**row, "sample_id": str(index)} for index, row in enumerate(make_rows())]

    with pytest.raises(ValueError, match="indistinguishable from dataframe row"):
        build_snapshot(snapshot_dir, rows)


def test_genuine_numeric_ids_survive_when_they_are_not_a_range(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    rows = [
        {**row, "sample_id": str(1000 + index * 7)}
        for index, row in enumerate(make_rows())
    ]
    build_snapshot(snapshot_dir, rows)

    snapshot = _load(tmp_path, snapshot_dir)

    assert snapshot.samples[0].sample_id == "1000"


def test_a_pinned_protocol_rejects_a_different_snapshot(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    document = pin(regression_document(), "b" * 64)
    protocol = load_protocol(write_protocol(tmp_path / "protocol.toml", document))

    with pytest.raises(ValueError, match="protocol pins"):
        load_snapshot(snapshot_dir, protocol)


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ({"record_id": "other"}, "protocol requires 'other'"),
        ({"snapshot_version": "v2"}, "protocol requires 'v2'"),
    ],
)
def test_a_pinned_protocol_rejects_a_different_identity(
    tmp_path: Path, snapshot_dir: Path, identity: dict[str, str], message: str
) -> None:
    digest = build_snapshot(snapshot_dir)
    document = pin(regression_document(), digest, **identity)
    protocol = load_protocol(write_protocol(tmp_path / "protocol.toml", document))

    with pytest.raises(ValueError, match=message):
        load_snapshot(snapshot_dir, protocol)


def test_an_unpinned_protocol_accepts_any_conforming_snapshot(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir, record_id="somebody-elses-data", snapshot_version="v9")

    snapshot = _load(tmp_path, snapshot_dir)

    assert snapshot.record_id == "somebody-elses-data"
    assert snapshot.identity()["manifest_sha256"] == sha256_file(
        snapshot_dir / "manifest.json"
    )


def test_load_snapshot_rejects_a_modified_data_file(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    path = snapshot_dir / "id_prop.csv"
    # Same length, so the digest check is what has to catch this.
    path.write_bytes(path.read_bytes().replace(b"syn-000", b"syn-999"))

    with pytest.raises(ValueError, match="id_prop.csv SHA-256 is"):
        _load(tmp_path, snapshot_dir)


def test_load_snapshot_rejects_a_resized_data_file(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    path = snapshot_dir / "id_prop.csv"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="id_prop.csv has .* bytes; expected"):
        _load(tmp_path, snapshot_dir)


def test_load_snapshot_rejects_a_missing_manifest(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    (snapshot_dir / "manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="goldilocks-train seal"):
        _load(tmp_path, snapshot_dir)


def test_load_snapshot_rejects_an_unsupported_manifest_version(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    path = snapshot_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 2
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="manifest schema_version must be 1"):
        _load(tmp_path, snapshot_dir)


def test_load_snapshot_rejects_a_manifest_file_without_a_digest(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    path = snapshot_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["sha256"] = "short"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="invalid SHA-256 for snapshot file"):
        _load(tmp_path, snapshot_dir)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.__setitem__(3, {**rows[0]}), "repeats a sample id"),
        (
            lambda rows: rows[5].__setitem__("value", "not-a-number"),
            "non-numeric target",
        ),
        (lambda rows: rows[5].__setitem__("value", float("inf")), "non-finite target"),
        (lambda rows: rows[5].__setitem__("sample_id", "  "), "empty sample id"),
        (lambda rows: rows[5].__setitem__("group", " "), "has an empty group"),
    ],
)
def test_load_snapshot_rejects_unusable_rows(
    tmp_path: Path, snapshot_dir: Path, mutate: Any, message: str
) -> None:
    rows = make_rows()
    mutate(rows)
    try:
        build_snapshot(snapshot_dir, rows)
    except ValueError as error:
        assert message in str(error)
        return

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, snapshot_dir)


def test_classification_needs_two_classes(tmp_path: Path, snapshot_dir: Path) -> None:
    rows = [{**row, "label": "metal"} for row in make_rows()]
    build_snapshot(snapshot_dir, rows, target="label")

    with pytest.raises(ValueError, match="at least two classes"):
        _load(tmp_path, snapshot_dir, classification=True)


def test_an_unedited_positive_label_is_caught_before_training(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir, target="label")

    with pytest.raises(ValueError, match="positive_label 'superconductor' is absent"):
        _load(
            tmp_path,
            snapshot_dir,
            classification=True,
            evaluation={"positive_label": "superconductor"},
        )
