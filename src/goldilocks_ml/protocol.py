"""Load and validate versioned machine-learning training protocols."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from goldilocks_ml.hashing import is_sha256

Task = Literal["regression", "classification"]
SplitMethod = Literal["random", "group"]

TASKS: frozenset[str] = frozenset({"regression", "classification"})
SPLIT_METHODS: frozenset[str] = frozenset({"random", "group"})
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "calibration", "test")

REGRESSION_METRICS: frozenset[str] = frozenset({"mae", "rmse", "r2"})
CLASSIFICATION_METRICS: frozenset[str] = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    }
)
BASELINES: dict[str, str] = {
    "regression": "train_median",
    "classification": "train_majority",
}


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Identity and columns required from an immutable dataset snapshot."""

    record_id: str
    snapshot_version: str
    manifest_sha256: str
    sample_id: str
    target: str


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Deterministic dataset partitioning configuration."""

    method: SplitMethod
    train: float
    validation: float
    calibration: float
    test: float
    seed: int
    group_column: str | None = None
    stratify: bool = False

    @property
    def ratios(self) -> tuple[tuple[str, float], ...]:
        """Return split names and ratios in assignment order."""
        return (
            ("train", self.train),
            ("validation", self.validation),
            ("calibration", self.calibration),
            ("test", self.test),
        )


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Versioned feature contract selected by a trainer.

    ``schema`` names a reviewed contract. ``columns`` is set only by contracts
    that read model inputs straight from snapshot columns; contracts that derive
    features from structures resolve them inside their own trainer.
    """

    schema: str
    columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Built-in trainer configuration.

    Trainer-specific settings live in ``parameters`` so that the surrounding
    protocol schema can reject unknown fields without knowing every trainer.
    """

    seed: int
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Metrics and baseline required by a protocol."""

    primary_metric: str
    metrics: tuple[str, ...]
    baseline: str
    threshold_metric: str | None = None
    positive_label: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingProtocol:
    """Validated configuration for one reproducible training workflow."""

    schema_version: int
    id: str
    task: Task
    trainer: str
    dataset: DatasetSpec
    split: SplitSpec
    features: FeatureSpec
    model: ModelSpec
    evaluation: EvaluationSpec
    source: Path

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return every snapshot column this protocol reads."""
        columns = [self.dataset.sample_id, self.dataset.target]
        if self.split.group_column is not None:
            columns.append(self.split.group_column)
        columns.extend(self.features.columns)
        seen: dict[str, None] = {}
        for column in columns:
            seen.setdefault(column, None)
        return tuple(seen)


_ROOT_KEYS = {
    "schema_version",
    "id",
    "task",
    "trainer",
    "dataset",
    "split",
    "features",
    "model",
    "evaluation",
}
_DATASET_KEYS = {
    "record_id",
    "snapshot_version",
    "manifest_sha256",
    "sample_id",
    "target",
}
_SPLIT_KEYS = {
    "method",
    "group_column",
    "stratify",
    "train",
    "validation",
    "calibration",
    "test",
    "seed",
}
_FEATURE_KEYS = {"schema", "columns"}
_MODEL_KEYS = {"seed", "parameters"}
_EVALUATION_KEYS = {
    "primary_metric",
    "metrics",
    "baseline",
    "threshold_metric",
    "positive_label",
}


def _table(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")


def _string(table: dict[str, Any], field: str, section: str) -> str:
    value = table.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{field} must be a non-empty string")
    return value


def _integer(table: dict[str, Any], field: str, section: str) -> int:
    value = table.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{section}.{field} must be a non-negative integer")
    return value


def _ratio(table: dict[str, Any], field: str) -> float:
    value = table.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"split.{field} must be a number")
    result = float(value)
    if result < 0 or result >= 1:
        raise ValueError(f"split.{field} must be at least 0 and less than 1")
    return result


def _load_dataset(root: dict[str, Any]) -> DatasetSpec:
    table = _table(root.get("dataset"), "dataset")
    _reject_unknown(table, _DATASET_KEYS, "dataset")
    manifest_sha256 = _string(table, "manifest_sha256", "dataset")
    if not is_sha256(manifest_sha256):
        raise ValueError("dataset.manifest_sha256 must be a lowercase SHA-256 digest")
    return DatasetSpec(
        record_id=_string(table, "record_id", "dataset"),
        snapshot_version=_string(table, "snapshot_version", "dataset"),
        manifest_sha256=manifest_sha256,
        sample_id=_string(table, "sample_id", "dataset"),
        target=_string(table, "target", "dataset"),
    )


def _load_split(root: dict[str, Any], task: str) -> SplitSpec:
    table = _table(root.get("split"), "split")
    _reject_unknown(table, _SPLIT_KEYS, "split")
    method = _string(table, "method", "split")
    if method not in SPLIT_METHODS:
        raise ValueError("split.method must be random or group")
    group_column = table.get("group_column")
    if group_column is not None and (
        not isinstance(group_column, str) or not group_column.strip()
    ):
        raise ValueError("split.group_column must be a non-empty string")
    if method == "group" and group_column is None:
        raise ValueError("split.group_column is required for group splitting")
    if method == "random" and group_column is not None:
        raise ValueError("split.group_column is only valid for group splitting")
    stratify = table.get("stratify", False)
    if not isinstance(stratify, bool):
        raise ValueError("split.stratify must be a boolean")
    if stratify and task != "classification":
        raise ValueError("split.stratify is only valid for classification")
    split = SplitSpec(
        method=cast(SplitMethod, method),
        group_column=group_column,
        stratify=stratify,
        train=_ratio(table, "train"),
        validation=_ratio(table, "validation"),
        calibration=_ratio(table, "calibration"),
        test=_ratio(table, "test"),
        seed=_integer(table, "seed", "split"),
    )
    if abs(sum(value for _, value in split.ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    if split.train == 0 or split.test == 0:
        raise ValueError("split.train and split.test must be greater than 0")
    return split


def _load_features(root: dict[str, Any]) -> FeatureSpec:
    table = _table(root.get("features"), "features")
    _reject_unknown(table, _FEATURE_KEYS, "features")
    raw_columns = table.get("columns", [])
    if not isinstance(raw_columns, list) or any(
        not isinstance(column, str) or not column.strip() for column in raw_columns
    ):
        raise ValueError("features.columns must be an array of non-empty strings")
    columns = tuple(raw_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("features.columns must be unique")
    return FeatureSpec(schema=_string(table, "schema", "features"), columns=columns)


def _load_model(root: dict[str, Any]) -> ModelSpec:
    table = _table(root.get("model"), "model")
    _reject_unknown(table, _MODEL_KEYS, "model")
    parameters = table.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("model.parameters must be a TOML table")
    return ModelSpec(seed=_integer(table, "seed", "model"), parameters=parameters)


def _load_evaluation(root: dict[str, Any], task: str) -> EvaluationSpec:
    table = _table(root.get("evaluation"), "evaluation")
    _reject_unknown(table, _EVALUATION_KEYS, "evaluation")
    raw_metrics = table.get("metrics")
    if (
        not isinstance(raw_metrics, list)
        or not raw_metrics
        or any(not isinstance(metric, str) or not metric for metric in raw_metrics)
    ):
        raise ValueError("evaluation.metrics must be a non-empty string array")
    metrics = tuple(raw_metrics)
    if len(metrics) != len(set(metrics)):
        raise ValueError("evaluation.metrics must be unique")
    supported = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
    unsupported = sorted(set(metrics) - supported)
    if unsupported:
        raise ValueError(f"unsupported {task} metric(s): {', '.join(unsupported)}")
    primary_metric = _string(table, "primary_metric", "evaluation")
    if primary_metric not in metrics:
        raise ValueError("evaluation.primary_metric must be listed in metrics")
    baseline = _string(table, "baseline", "evaluation")
    expected_baseline = BASELINES[task]
    if baseline != expected_baseline:
        raise ValueError(f"{task} baseline must be {expected_baseline}")
    threshold_metric = table.get("threshold_metric")
    if threshold_metric is not None:
        if task != "classification":
            raise ValueError(
                "evaluation.threshold_metric is only valid for classification"
            )
        if not isinstance(threshold_metric, str) or threshold_metric not in metrics:
            raise ValueError("evaluation.threshold_metric must be listed in metrics")
        if threshold_metric in {"roc_auc", "pr_auc"}:
            raise ValueError(
                "evaluation.threshold_metric must be a threshold-dependent metric"
            )
    positive_label = table.get("positive_label")
    if positive_label is not None:
        if task != "classification":
            raise ValueError(
                "evaluation.positive_label is only valid for classification"
            )
        if not isinstance(positive_label, str) or not positive_label.strip():
            raise ValueError("evaluation.positive_label must be a non-empty string")
    return EvaluationSpec(
        primary_metric=primary_metric,
        metrics=metrics,
        baseline=baseline,
        threshold_metric=threshold_metric,
        positive_label=positive_label,
    )


def load_protocol(path: Path) -> TrainingProtocol:
    """Load a TOML training protocol and reject invalid or unknown fields."""
    path = path.resolve()
    with path.open("rb") as handle:
        root = tomllib.load(handle)
    _reject_unknown(root, _ROOT_KEYS, "protocol")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    protocol_id = _string(root, "id", "protocol")
    task = _string(root, "task", "protocol")
    if task not in TASKS:
        raise ValueError("protocol.task must be regression or classification")
    trainer = _string(root, "trainer", "protocol")

    return TrainingProtocol(
        schema_version=1,
        id=protocol_id,
        task=cast(Task, task),
        trainer=trainer,
        dataset=_load_dataset(root),
        split=_load_split(root, task),
        features=_load_features(root),
        model=_load_model(root),
        evaluation=_load_evaluation(root, task),
        source=path,
    )
