"""Offline fixtures for the shared training protocol suite."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from goldilocks_ml.cli import seal
from goldilocks_ml.runs import dumps_toml

FIXTURES = Path(__file__).parent / "fixtures"
KDIST_SNAPSHOT = FIXTURES / "kdist"
METALLIC_SNAPSHOT = FIXTURES / "metallic"
PROTOCOLS = Path(__file__).parents[1] / "protocols" / "synthetic"
PACKAGE = Path(__file__).parents[1] / "src" / "goldilocks_ml"

FEATURE_COLUMNS = ("x1", "x2", "x3")


def make_rows(count: int = 24, groups: int = 8) -> list[dict[str, Any]]:
    """Build a small, exactly reproducible table with a known linear target."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        x1 = (index % 5) - 2.0
        x2 = (index % 3) - 1.0
        x3 = (index % 7) / 7.0
        rows.append(
            {
                "sample_id": f"syn-{index:03d}",
                "group": f"grp-{index % groups:02d}",
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "value": 2.0 * x1 - x2 + 0.5 * x3 + 3.0,
                "label": "metal" if x1 >= 0 else "insulator",
            }
        )
    return rows


def build_snapshot(
    directory: Path,
    rows: Sequence[dict[str, Any]] | None = None,
    *,
    record_id: str = "synthetic",
    snapshot_version: str = "v1",
    target: str = "value",
    groups: bool = True,
    features: bool = True,
    structures: bool = False,
) -> str:
    """Write a snapshot directory, seal it, and return its manifest digest."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(make_rows()) if rows is None else list(rows)

    with (directory / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            value = row[target]
            record = [
                row["sample_id"],
                f"{value:.6f}" if isinstance(value, float) else value,
            ]
            if groups:
                record.append(row["group"])
            writer.writerow(record)

    if features:
        with (directory / "features.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", *FEATURE_COLUMNS])
            for row in rows:
                writer.writerow(
                    [row["sample_id"], *(f"{row[c]:.6f}" for c in FEATURE_COLUMNS)]
                )

    if structures:
        for row in rows:
            (directory / f"{row['sample_id']}.cif").write_text(
                f"# synthetic structure for {row['sample_id']}\n", encoding="utf-8"
            )

    return seal(
        directory,
        record_id=record_id,
        snapshot_version=snapshot_version,
        structure_suffix=".cif",
        target_name=target,
        target_contract=f"synthetic.{target}.v1",
        target_definition=f"Synthetic test target {target}.",
        target_units="arbitrary" if target == "value" else None,
    )["manifest_sha256"]


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


def regression_document(**overrides: Any) -> dict[str, Any]:
    """Return the baseline synthetic regression protocol document."""
    base = {
        "schema_version": 1,
        "id": "synthetic-regression-v1",
        "task": "regression",
        "trainer": "linear_regression",
        "dataset": {
            "target": "value",
            "target_contract": "synthetic.value.v1",
            "target_units": "arbitrary",
            "requires": ["features"],
        },
        "split": {
            "method": "random",
            "train": 0.5,
            "validation": 0.2,
            "calibration": 0.1,
            "test": 0.2,
            "seed": 7,
        },
        "features": {"schema": "tabular", "parameters": {}},
        "model": {"seed": 7, "parameters": {"l2": 1e-9}},
        "evaluation": {
            "primary_metric": "mae",
            "metrics": ["mae", "rmse", "r2"],
            "baseline": "train_median",
        },
    }
    return merge(base, overrides)


def classification_document(**overrides: Any) -> dict[str, Any]:
    """Return the baseline synthetic classification protocol document."""
    base = {
        "schema_version": 1,
        "id": "synthetic-classification-v1",
        "task": "classification",
        "trainer": "logistic_regression",
        "dataset": {
            "target": "label",
            "target_contract": "synthetic.label.v1",
            "requires": ["features"],
        },
        "split": {
            "method": "random",
            "train": 0.5,
            "validation": 0.2,
            "calibration": 0.1,
            "test": 0.2,
            "seed": 7,
        },
        "features": {"schema": "tabular", "parameters": {}},
        "model": {"seed": 7, "parameters": {"iterations": 200}},
        "evaluation": {
            "primary_metric": "mcc",
            "metrics": ["accuracy", "mcc", "roc_auc"],
            "baseline": "train_majority",
            "positive_label": "metal",
        },
    }
    return merge(base, overrides)


def pin(document: dict[str, Any], digest: str, **identity: str) -> dict[str, Any]:
    """Return the document with a snapshot pinned into its dataset section."""
    return merge(
        document,
        {
            "dataset": {
                "record_id": identity.get("record_id", "synthetic"),
                "snapshot_version": identity.get("snapshot_version", "v1"),
                "manifest_sha256": digest,
            }
        },
    )


def write_protocol(path: Path, document: dict[str, Any]) -> Path:
    """Serialise a protocol document to TOML on disk."""
    path.write_text(dumps_toml(document), encoding="utf-8")
    return path


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    """Return a fresh snapshot directory path."""
    return tmp_path / "snapshot"
