"""Tests for strict training protocol loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import (
    PACKAGE,
    PROTOCOLS,
    classification_document,
    regression_document,
    write_protocol,
)

from goldilocks_ml.protocol import load_protocol

DIGEST = "a" * 64


def _regression(tmp_path: Path, **overrides: Any) -> Path:
    return write_protocol(tmp_path / "protocol.toml", regression_document(**overrides))


def _classification(tmp_path: Path, **overrides: Any) -> Path:
    return write_protocol(
        tmp_path / "protocol.toml", classification_document(**overrides)
    )


def test_load_protocol_returns_validated_configuration(tmp_path: Path) -> None:
    protocol = load_protocol(_regression(tmp_path))

    assert protocol.id == "synthetic-regression-v1"
    assert protocol.task == "regression"
    assert protocol.trainer == "linear_regression"
    assert protocol.features.schema == "tabular"
    assert protocol.evaluation.metrics == ("mae", "rmse", "r2")


def test_a_template_pins_no_snapshot(tmp_path: Path) -> None:
    protocol = load_protocol(_regression(tmp_path))

    assert protocol.dataset.pinned is None
    assert protocol.dataset.requires == ("features",)


def test_a_pinned_protocol_carries_the_whole_identity(tmp_path: Path) -> None:
    protocol = load_protocol(
        _regression(
            tmp_path,
            dataset={
                "record_id": "synthetic",
                "snapshot_version": "v1",
                "manifest_sha256": DIGEST,
            },
        )
    )

    assert protocol.dataset.pinned is not None
    assert protocol.dataset.pinned.record_id == "synthetic"
    assert protocol.dataset.pinned.manifest_sha256 == DIGEST


def test_partial_pinning_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing snapshot_version, manifest_sha256"):
        load_protocol(_regression(tmp_path, dataset={"record_id": "synthetic"}))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version must be 1"),
        ({"task": "clustering"}, "task must be regression or classification"),
        ({"id": ""}, "protocol.id must be a non-empty string"),
        ({"trainer": " "}, "protocol.trainer must be a non-empty string"),
    ],
)
def test_load_protocol_rejects_invalid_root(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_protocol(_regression(tmp_path, **overrides))


def test_load_protocol_rejects_unknown_root_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown protocol field\\(s\\): mlflow_uri"):
        load_protocol(_regression(tmp_path, mlflow_uri="http://tracking"))


@pytest.mark.parametrize(
    ("section", "overrides", "message"),
    [
        ("dataset", {"target": ""}, "dataset.target"),
        ("dataset", {"requires": ["lmdb"]}, "unknown dataset.requires value"),
        ("dataset", {"requires": ["features", "features"]}, "must be unique"),
        ("dataset", {"manifest_sha256": "abc"}, "needs every one of"),
        ("dataset", {"notebook": "run.ipynb"}, "unknown dataset field"),
        ("split", {"train": 1.0}, "less than 1"),
        ("split", {"validation": 0.3}, "must sum to 1"),
        ("split", {"seed": -1}, "split.seed"),
        ("split", {"method": "kfold"}, "random or group"),
        ("split", {"group_column": "x"}, "unknown split field"),
        ("features", {"parameters": 3}, "features.parameters must be a TOML table"),
        ("features", {"columns": ["x"]}, "unknown features field"),
        ("model", {"parameters": 3}, "model.parameters must be a TOML table"),
        ("evaluation", {"metrics": []}, "non-empty string array"),
        ("evaluation", {"metrics": ["mae", "mae"]}, "must be unique"),
        ("evaluation", {"metrics": ["mae", "top_k"]}, "unsupported regression metric"),
        ("evaluation", {"primary_metric": "rmse", "metrics": ["mae"]}, "listed in"),
        ("evaluation", {"baseline": "zero"}, "baseline must be train_median"),
    ],
)
def test_load_protocol_rejects_invalid_sections(
    tmp_path: Path, section: str, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_protocol(_regression(tmp_path, **{section: overrides}))


def test_stratify_is_rejected_for_regression(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid for classification"):
        load_protocol(_regression(tmp_path, split={"stratify": True}))


def test_train_and_test_splits_must_be_non_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        load_protocol(
            _regression(
                tmp_path,
                split={
                    "train": 0.8,
                    "validation": 0.1,
                    "calibration": 0.1,
                    "test": 0.0,
                },
            )
        )


def test_feature_dependencies_are_pinned_by_digest(tmp_path: Path) -> None:
    protocol = load_protocol(
        _regression(
            tmp_path,
            features={
                "depends_on": {
                    "metallicity": {
                        "record_id": "ptc95-vbq12",
                        "file": "is_metal.ckpt",
                        "sha256": DIGEST,
                    }
                }
            },
        )
    )

    assert len(protocol.features.depends_on) == 1
    dependency = protocol.features.depends_on[0]
    assert dependency.name == "metallicity"
    assert dependency.file == "is_metal.ckpt"
    assert dependency.sha256 == DIGEST


@pytest.mark.parametrize(
    ("dependency", "message"),
    [
        ({"record_id": "r", "file": "f", "sha256": "short"}, "lowercase SHA-256"),
        ({"record_id": "r", "file": "a/b", "sha256": DIGEST}, "must be a basename"),
        ({"record_id": "", "file": "f", "sha256": DIGEST}, "record_id"),
        ({"record_id": "r", "file": "f", "sha256": DIGEST, "url": "x"}, "unknown"),
    ],
)
def test_invalid_feature_dependencies_are_rejected(
    tmp_path: Path, dependency: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_protocol(
            _regression(tmp_path, features={"depends_on": {"metallicity": dependency}})
        )


def test_classification_accepts_ranking_metrics_and_threshold(tmp_path: Path) -> None:
    protocol = load_protocol(
        _classification(
            tmp_path,
            evaluation={
                "metrics": ["accuracy", "mcc", "roc_auc", "pr_auc"],
                "threshold_metric": "mcc",
            },
        )
    )

    assert protocol.evaluation.threshold_metric == "mcc"
    assert protocol.evaluation.positive_label == "metal"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"threshold_metric": "f1"}, "must be listed in metrics"),
        ({"threshold_metric": "roc_auc"}, "threshold-dependent"),
        ({"positive_label": ""}, "positive_label must be a non-empty string"),
        ({"baseline": "train_median"}, "baseline must be train_majority"),
        ({"metrics": ["mae"]}, "unsupported classification metric"),
    ],
)
def test_load_protocol_rejects_invalid_classification_evaluation(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_protocol(_classification(tmp_path, evaluation=overrides))


def test_regression_rejects_classification_only_evaluation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive_label is only valid"):
        load_protocol(_regression(tmp_path, evaluation={"positive_label": "metal"}))


@pytest.mark.parametrize(
    "path",
    sorted(PROTOCOLS.rglob("*.toml")) + sorted(PACKAGE.rglob("protocol.toml")),
    ids=lambda path: "/".join(path.parts[-3:]),
)
def test_every_committed_protocol_loads(path: Path) -> None:
    protocol = load_protocol(path)

    assert protocol.schema_version == 1
    assert protocol.features.schema
