"""Tests for immutable dataset snapshot verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    COLUMNS,
    build_snapshot,
    make_rows,
    regression_document,
    write_protocol,
)

from goldilocks_ml.datasets import load_snapshot
from goldilocks_ml.protocol import load_protocol


def _load(tmp_path: Path, snapshot: Path, digest: str, **overrides: Any):
    protocol = load_protocol(
        write_protocol(
            tmp_path / "protocol.toml", regression_document(digest, **overrides)
        )
    )
    return load_snapshot(snapshot, protocol)


def test_load_snapshot_returns_verified_samples(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    digest = build_snapshot(snapshot_dir)

    snapshot = _load(tmp_path, snapshot_dir, digest)

    assert len(snapshot.samples) == 24
    assert snapshot.manifest_sha256 == digest
    assert snapshot.samples[0].sample_id == "s000"
    assert snapshot.samples[0].features == (-2.0, -1.0, 0.0)
    assert snapshot.identity()["sample_count"] == 24


def test_load_snapshot_rejects_manifest_digest_mismatch(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)

    with pytest.raises(ValueError, match="protocol pins"):
        _load(tmp_path, snapshot_dir, "b" * 64)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"record_id": "other"}, "protocol requires 'other'"),
        ({"snapshot_version": "v2"}, "protocol requires 'v2'"),
    ],
)
def test_load_snapshot_rejects_identity_mismatch(
    tmp_path: Path, snapshot_dir: Path, overrides: dict[str, str], message: str
) -> None:
    digest = build_snapshot(snapshot_dir)
    # The pinned digest still matches, so only the declared identity differs.
    protocol_overrides = {"dataset": {**overrides, "manifest_sha256": digest}}

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, snapshot_dir, digest, **protocol_overrides)


def test_load_snapshot_rejects_modified_data_file(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    digest = build_snapshot(snapshot_dir)
    data = snapshot_dir / "data.csv"
    # Same length, so the digest check is what has to catch this.
    data.write_bytes(data.read_bytes().replace(b"s000", b"s999"))

    with pytest.raises(ValueError, match="data.csv SHA-256 is"):
        _load(tmp_path, snapshot_dir, digest)


def test_load_snapshot_rejects_resized_data_file(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    digest = build_snapshot(snapshot_dir)
    data = snapshot_dir / "data.csv"
    data.write_bytes(data.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="data.csv has .* bytes; expected"):
        _load(tmp_path, snapshot_dir, digest)


def test_load_snapshot_rejects_missing_file(tmp_path: Path, snapshot_dir: Path) -> None:
    digest = build_snapshot(snapshot_dir)
    (snapshot_dir / "data.csv").unlink()

    with pytest.raises(FileNotFoundError):
        _load(tmp_path, snapshot_dir, digest)


def test_load_snapshot_rejects_missing_manifest(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    snapshot_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        _load(tmp_path, snapshot_dir, "c" * 64)


def test_load_snapshot_rejects_unsupported_manifest_version(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    digest = build_snapshot(snapshot_dir, manifest_overrides={"schema_version": 2})

    with pytest.raises(ValueError, match="manifest schema_version must be 1"):
        _load(tmp_path, snapshot_dir, digest)


def test_load_snapshot_rejects_missing_columns(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    digest = build_snapshot(snapshot_dir)

    with pytest.raises(ValueError, match="missing column\\(s\\): soap_0"):
        _load(
            tmp_path,
            snapshot_dir,
            digest,
            features={"columns": ["x1", "soap_0"]},
        )


def test_load_snapshot_rejects_duplicate_sample_ids(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    rows = make_rows()
    rows[3]["sample_id"] = rows[0]["sample_id"]
    digest = build_snapshot(snapshot_dir, rows)

    with pytest.raises(ValueError, match="duplicate sample id: s000"):
        _load(tmp_path, snapshot_dir, digest)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("sample_id", "  ", "empty sample_id"),
        ("target_value", "", "empty target_value"),
        ("target_value", "not-a-number", "non-numeric target_value"),
        ("target_value", "inf", "non-finite target_value"),
        ("x2", "n/a", "non-numeric x2"),
    ],
)
def test_load_snapshot_rejects_unusable_values(
    tmp_path: Path, snapshot_dir: Path, column: str, value: str, message: str
) -> None:
    rows = make_rows()
    rows[5][column] = value
    digest = build_snapshot(snapshot_dir, rows)

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, snapshot_dir, digest)


def test_load_snapshot_rejects_empty_group_value(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    rows = make_rows()
    rows[2]["structure_group_id"] = ""
    digest = build_snapshot(snapshot_dir, rows)

    with pytest.raises(ValueError, match="empty structure_group_id"):
        _load(
            tmp_path,
            snapshot_dir,
            digest,
            split={"method": "group", "group_column": "structure_group_id"},
        )


def test_load_snapshot_requires_two_classes(tmp_path: Path, snapshot_dir: Path) -> None:
    rows = [{**row, "target_class": "metal"} for row in make_rows()]
    digest = build_snapshot(snapshot_dir, rows)
    protocol = load_protocol(
        write_protocol(
            tmp_path / "protocol.toml",
            regression_document(
                digest,
                task="classification",
                trainer="logistic_regression",
                dataset={"target": "target_class"},
                evaluation={
                    "primary_metric": "accuracy",
                    "metrics": ["accuracy"],
                    "baseline": "train_majority",
                },
            ),
        )
    )

    with pytest.raises(ValueError, match="at least two classes"):
        load_snapshot(snapshot_dir, protocol)


def test_load_snapshot_rejects_data_file_without_rows(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    digest = build_snapshot(snapshot_dir, [])

    with pytest.raises(ValueError, match="no data rows"):
        _load(tmp_path, snapshot_dir, digest)


def test_load_snapshot_rejects_manifest_file_entry_without_digest(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = "short"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    from goldilocks_ml.hashing import sha256_file

    with pytest.raises(ValueError, match="invalid SHA-256 for snapshot file"):
        _load(tmp_path, snapshot_dir, sha256_file(manifest_path))


def test_snapshot_columns_are_read_from_the_manifest_data_file() -> None:
    assert COLUMNS[0] == "sample_id"
