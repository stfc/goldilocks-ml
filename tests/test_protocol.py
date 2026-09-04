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

    assert protocol.id == "synthetic.value.linear.synthetic.v1"
    assert protocol.task == "regression"
    assert protocol.trainer == "linear_regression"
    assert protocol.features.schema == "tabular"
    assert protocol.dataset.target_contract == "synthetic.value.v1"
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
        ("dataset", {"target_contract": ""}, "dataset.target_contract"),
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


def test_a_release_name_parses_into_its_parts(tmp_path: Path) -> None:
    protocol = load_protocol(_regression(tmp_path))

    assert protocol.release.parameter == "synthetic"
    assert protocol.release.quantity == "value"
    assert protocol.release.family == "linear"
    assert protocol.release.dataset == "synthetic"
    assert protocol.release.version == 1
    assert protocol.release.runtime == "synthetic.value.linear"
    assert str(protocol.release) == protocol.id


@pytest.mark.parametrize(
    "protocol_id",
    [
        "synthetic-regression-v1",
        "synthetic.value.linear.v1",
        "synthetic.value.linear.synthetic.synthetic.v1",
        "synthetic.value.linear.synthetic",
        "synthetic.value.linear.synthetic.v0",
        "Synthetic.Value.Linear.Synthetic.v1",
    ],
)
def test_load_protocol_rejects_a_name_that_is_not_a_release(
    tmp_path: Path, protocol_id: str
) -> None:
    with pytest.raises(ValueError, match="must name a release"):
        load_protocol(_regression(tmp_path, id=protocol_id))


def test_a_release_name_may_not_contradict_the_pinned_dataset(
    tmp_path: Path,
) -> None:
    document = regression_document(
        id="synthetic.value.linear.elsewhere.v1",
        dataset={
            "record_id": "synthetic",
            "snapshot_version": "v1",
            "manifest_sha256": DIGEST,
        },
    )

    with pytest.raises(ValueError, match="names dataset 'elsewhere'"):
        load_protocol(write_protocol(tmp_path / "protocol.toml", document))


def test_a_hyphenated_record_id_is_spelled_with_underscores(tmp_path: Path) -> None:
    document = regression_document(
        id="synthetic.value.linear.two_words.v1",
        dataset={
            "record_id": "two-words",
            "snapshot_version": "v1",
            "manifest_sha256": DIGEST,
        },
    )

    protocol = load_protocol(write_protocol(tmp_path / "protocol.toml", document))

    assert protocol.dataset.pinned.record_id == "two-words"


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
    assert protocol.evaluation.min_recall is None


def test_classification_accepts_a_recall_floor(tmp_path: Path) -> None:
    protocol = load_protocol(
        _classification(
            tmp_path,
            evaluation={
                "metrics": ["accuracy", "recall", "mcc"],
                "threshold_metric": "mcc",
                "min_recall": 0.97,
            },
        )
    )

    assert protocol.evaluation.min_recall == 0.97


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"threshold_metric": "f1"}, "must be listed in metrics"),
        ({"threshold_metric": "roc_auc"}, "threshold-dependent"),
        ({"positive_label": ""}, "positive_label must be a non-empty string"),
        (
            {
                "metrics": ["accuracy", "recall", "mcc"],
                "threshold_metric": "mcc",
                "min_recall": 0.0,
            },
            r"min_recall must lie in \(0, 1\]",
        ),
        (
            {
                "metrics": ["accuracy", "recall", "mcc"],
                "threshold_metric": "mcc",
                "min_recall": "high",
            },
            "min_recall must be a number",
        ),
        (
            {"metrics": ["accuracy", "mcc"], "min_recall": 0.97},
            "min_recall requires evaluation.threshold_metric",
        ),
        (
            {
                "metrics": ["accuracy", "mcc"],
                "threshold_metric": "mcc",
                "min_recall": 0.97,
            },
            "min_recall requires recall in metrics",
        ),
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


def test_regression_accepts_the_rounded_decision_metrics(tmp_path: Path) -> None:
    protocol = load_protocol(
        _regression(
            tmp_path,
            evaluation={
                "primary_metric": "mae",
                "metrics": ["mae", "rounded_accuracy", "underprediction_rate"],
                "coverage_bins": [6, 11],
            },
        )
    )

    assert protocol.evaluation.coverage_bins == (6.0, 11.0)
    assert "underprediction_rate" in protocol.evaluation.metrics


def test_classification_rejects_the_rounded_decision_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported classification metric"):
        load_protocol(
            _classification(
                tmp_path,
                evaluation={"primary_metric": "mcc", "metrics": ["mcc", "within_one"]},
            )
        )


def test_coverage_bins_are_rejected_for_classification(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid for regression"):
        load_protocol(_classification(tmp_path, evaluation={"coverage_bins": [0.5]}))


@pytest.mark.parametrize(
    ("bins", "message"),
    [
        ([], "non-empty array"),
        ([11, 6], "must increase"),
        ([6, 6], "must increase"),
        (["6"], "must contain numbers"),
        ([True], "must contain numbers"),
    ],
)
def test_coverage_bins_are_validated(tmp_path: Path, bins: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_protocol(_regression(tmp_path, evaluation={"coverage_bins": bins}))


def _decision(**evaluation: Any) -> dict[str, Any]:
    base = {
        "primary_metric": "mean_excess",
        "metrics": ["mean_excess", "underprediction_rate", "mae"],
        "decision_metric": "mean_excess",
        "max_underprediction": 0.05,
    }
    base.update(evaluation)
    return base


def test_a_regression_protocol_can_state_the_error_it_refuses(tmp_path: Path) -> None:
    protocol = load_protocol(_regression(tmp_path, evaluation=_decision()))

    assert protocol.evaluation.decision_metric == "mean_excess"
    assert protocol.evaluation.max_underprediction == pytest.approx(0.05)


def test_a_decision_metric_is_rejected_for_classification(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid for regression"):
        load_protocol(
            _classification(tmp_path, evaluation={"decision_metric": "mean_excess"})
        )


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        (_decision(decision_metric="rmse"), "must be listed in metrics"),
        (
            _decision(
                metrics=["rmse", "underprediction_rate", "mean_excess"],
                decision_metric="rmse",
            ),
            "must be one of",
        ),
        (
            {
                "primary_metric": "mae",
                "metrics": ["mae"],
                "max_underprediction": 0.05,
            },
            "requires evaluation.decision_metric",
        ),
        (_decision(max_underprediction=1.0), "must lie in"),
        (_decision(max_underprediction="0.05"), "must be a number"),
        (
            _decision(metrics=["mean_excess", "mae"]),
            "requires underprediction_rate",
        ),
    ],
)
def test_a_decision_rule_is_validated(
    tmp_path: Path, evaluation: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_protocol(_regression(tmp_path, evaluation=evaluation))


def test_decision_bands_need_a_floor_to_honour(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires evaluation.max_underprediction"):
        load_protocol(
            _regression(
                tmp_path,
                evaluation={
                    "primary_metric": "mean_excess",
                    "metrics": ["mean_excess"],
                    "decision_bands": [6],
                },
            )
        )


def test_decision_bands_are_validated(tmp_path: Path) -> None:
    protocol = load_protocol(
        _regression(tmp_path, evaluation=_decision(decision_bands=[6, 11]))
    )
    assert protocol.evaluation.decision_bands == (6.0, 11.0)

    with pytest.raises(ValueError, match="must increase"):
        load_protocol(
            _regression(tmp_path, evaluation=_decision(decision_bands=[11, 6]))
        )
