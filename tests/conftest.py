"""Offline fixtures for the shared training protocol suite."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.runs import dumps_toml

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic-tabular"
PROTOCOL_ROOT = Path(__file__).parents[1] / "protocols" / "synthetic"

COLUMNS = (
    "sample_id",
    "structure_group_id",
    "x1",
    "x2",
    "x3",
    "target_value",
    "target_class",
)


def make_rows(count: int = 24, groups: int = 8) -> list[dict[str, str]]:
    """Build a small, exactly reproducible table with a known linear target."""
    rows: list[dict[str, str]] = []
    for index in range(count):
        x1 = (index % 5) - 2.0
        x2 = (index % 3) - 1.0
        x3 = (index % 7) / 7.0
        rows.append(
            {
                "sample_id": f"s{index:03d}",
                "structure_group_id": f"g{index % groups:02d}",
                "x1": f"{x1:.6f}",
                "x2": f"{x2:.6f}",
                "x3": f"{x3:.6f}",
                "target_value": f"{2.0 * x1 - x2 + 0.5 * x3 + 3.0:.6f}",
                "target_class": "metal" if x1 >= 0 else "insulator",
            }
        )
    return rows


def build_snapshot(
    directory: Path,
    rows: Sequence[dict[str, str]] | None = None,
    *,
    record_id: str = "synthetic",
    snapshot_version: str = "v1",
    columns: Sequence[str] = COLUMNS,
    manifest_overrides: dict[str, Any] | None = None,
) -> str:
    """Write a snapshot directory and return its manifest digest."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(make_rows()) if rows is None else list(rows)
    data_path = directory / "data.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_id": record_id,
        "snapshot_version": snapshot_version,
        "data_file": "data.csv",
        "files": [
            {
                "name": "data.csv",
                "size_bytes": data_path.stat().st_size,
                "sha256": sha256_file(data_path),
            }
        ],
    }
    manifest.update(manifest_overrides or {})
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(manifest_path)


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``base`` with ``overrides`` applied."""
    result = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in base.items()
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def regression_document(manifest_sha256: str, **overrides: Any) -> dict[str, Any]:
    """Return the baseline synthetic regression protocol document."""
    base = {
        "schema_version": 1,
        "id": "synthetic-regression-v1",
        "task": "regression",
        "trainer": "linear_regression",
        "dataset": {
            "record_id": "synthetic",
            "snapshot_version": "v1",
            "manifest_sha256": manifest_sha256,
            "sample_id": "sample_id",
            "target": "target_value",
        },
        "split": {
            "method": "random",
            "train": 0.5,
            "validation": 0.2,
            "calibration": 0.1,
            "test": 0.2,
            "seed": 7,
        },
        "features": {"schema": "synthetic_xyz", "columns": ["x1", "x2", "x3"]},
        "model": {"seed": 7, "parameters": {"l2": 1e-9}},
        "evaluation": {
            "primary_metric": "mae",
            "metrics": ["mae", "rmse", "r2"],
            "baseline": "train_median",
        },
    }
    return merge(base, overrides)


def classification_document(manifest_sha256: str, **overrides: Any) -> dict[str, Any]:
    """Return the baseline synthetic classification protocol document."""
    base = {
        "schema_version": 1,
        "id": "synthetic-classification-v1",
        "task": "classification",
        "trainer": "logistic_regression",
        "dataset": {
            "record_id": "synthetic",
            "snapshot_version": "v1",
            "manifest_sha256": manifest_sha256,
            "sample_id": "sample_id",
            "target": "target_class",
        },
        "split": {
            "method": "random",
            "train": 0.5,
            "validation": 0.2,
            "calibration": 0.1,
            "test": 0.2,
            "seed": 7,
        },
        "features": {"schema": "synthetic_xyz", "columns": ["x1", "x2", "x3"]},
        "model": {"seed": 7, "parameters": {"iterations": 200}},
        "evaluation": {
            "primary_metric": "mcc",
            "metrics": ["accuracy", "mcc", "roc_auc"],
            "baseline": "train_majority",
            "positive_label": "metal",
        },
    }
    return merge(base, overrides)


def write_protocol(path: Path, document: dict[str, Any]) -> Path:
    """Serialise a protocol document to TOML on disk."""
    path.write_text(dumps_toml(document), encoding="utf-8")
    return path


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    """Return a fresh snapshot directory path."""
    return tmp_path / "snapshot"


@pytest.fixture
def regression_setup(tmp_path: Path, snapshot_dir: Path) -> tuple[Path, Path]:
    """Return a valid regression protocol path and its snapshot directory."""
    digest = build_snapshot(snapshot_dir)
    protocol = write_protocol(tmp_path / "protocol.toml", regression_document(digest))
    return protocol, snapshot_dir


@pytest.fixture
def classification_setup(tmp_path: Path, snapshot_dir: Path) -> tuple[Path, Path]:
    """Return a valid classification protocol path and its snapshot directory."""
    digest = build_snapshot(snapshot_dir)
    protocol = write_protocol(
        tmp_path / "protocol.toml", classification_document(digest)
    )
    return protocol, snapshot_dir
