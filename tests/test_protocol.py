"""Tests for strict training protocol loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import (
    PROTOCOL_ROOT,
    classification_document,
    regression_document,
    write_protocol,
)

from goldilocks_ml.protocol import load_protocol

DIGEST = "a" * 64


def _regression(tmp_path: Path, **overrides: Any) -> Path:
    return write_protocol(
        tmp_path / "protocol.toml", regression_document(DIGEST, **overrides)
    )


def _classification(tmp_path: Path, **overrides: Any) -> Path:
    return write_protocol(
        tmp_path / "protocol.toml", classification_document(DIGEST, **overrides)
    )


def test_load_protocol_returns_validated_configuration(tmp_path: Path) -> None:
    protocol = load_protocol(_regression(tmp_path))

    assert protocol.id == "synthetic-regression-v1"
    assert protocol.task == "regression"
    assert protocol.trainer == "linear_regression"
    assert protocol.evaluation.metrics == ("mae", "rmse", "r2")
    assert protocol.features.columns == ("x1", "x2", "x3")


def test_required_columns_covers_every_column_read(tmp_path: Path) -> None:
    path = _regression(
        tmp_path, split={"method": "group", "group_column": "structure_group_id"}
    )

    protocol = load_protocol(path)

    assert protocol.required_columns == (
        "sample_id",
        "target_value",
        "structure_group_id",
        "x1",
        "x2",
        "x3",
    )


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
        ("dataset", {"manifest_sha256": "abc"}, "lowercase SHA-256"),
        ("dataset", {"record_id": ""}, "dataset.record_id"),
        ("dataset", {"notebook": "run.ipynb"}, "unknown dataset field"),
        ("split", {"train": 1.0}, "less than 1"),
        ("split", {"validation": 0.3}, "must sum to 1"),
        ("split", {"seed": -1}, "split.seed"),
        ("split", {"method": "kfold"}, "random or group"),
        ("split", {"method": "group"}, "group_column is required"),
        ("features", {"columns": ["x1", "x1"]}, "features.columns must be unique"),
        ("features", {"columns": [""]}, "array of non-empty strings"),
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


def test_group_column_is_rejected_for_random_splitting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid for group splitting"):
        load_protocol(_regression(tmp_path, split={"group_column": "structure"}))


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
    "name", ["tabular_regression.toml", "tabular_classification.toml"]
)
def test_committed_synthetic_protocols_load(name: str) -> None:
    protocol = load_protocol(PROTOCOL_ROOT / name)

    assert protocol.split.method == "group"
    assert protocol.dataset.record_id == "synthetic-tabular"
