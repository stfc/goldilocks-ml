"""Load and validate versioned machine-learning training protocols."""

from __future__ import annotations

import re
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
CAPABILITIES: frozenset[str] = frozenset({"structures", "features", "groups"})

REGRESSION_METRICS: frozenset[str] = frozenset({"mae", "rmse", "r2"})
# Reported alongside the requested metrics when a trainer predicts intervals.
INTERVAL_METRICS: frozenset[str] = frozenset(
    {"interval_coverage", "mean_interval_width", "pinball_loss"}
)
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


# A release name is five parts: which setting it advises, what its number is,
# what was fitted, which data it learned from, and which attempt this is. The
# first three are the serving runtime; a consumer that can read one release of
# a runtime can read every release of it.
_SEGMENT = r"[a-z][a-z0-9_]*"
_RELEASE_PATTERN = re.compile(
    rf"^(?P<parameter>{_SEGMENT})\.(?P<quantity>{_SEGMENT})\."
    rf"(?P<family>{_SEGMENT})\.(?P<dataset>{_SEGMENT})\.v(?P<version>[1-9][0-9]*)$"
)


@dataclass(frozen=True, slots=True)
class ReleaseName:
    """One trained model's name, parsed into the parts that mean something."""

    parameter: str
    quantity: str
    family: str
    dataset: str
    version: int

    @property
    def runtime(self) -> str:
        """Return the serving runtime this release must be read back through."""
        return f"{self.parameter}.{self.quantity}.{self.family}"

    def __str__(self) -> str:
        return f"{self.runtime}.{self.dataset}.v{self.version}"


def parse_release_name(value: str) -> ReleaseName:
    """Parse a release name, rejecting anything that is not five parts."""
    match = _RELEASE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            f"protocol.id {value!r} must name a release as "
            "<parameter>.<quantity>.<family>.<dataset>.v<n>, lowercase, "
            "for example 'k_points.k_distance.qrf.goldilocks_kdist_ultra.v1'"
        )
    return ReleaseName(
        parameter=match["parameter"],
        quantity=match["quantity"],
        family=match["family"],
        dataset=match["dataset"],
        version=int(match["version"]),
    )


def dataset_segment(record_id: str) -> str:
    """Return the release-name spelling of a snapshot record id."""
    return record_id.replace("-", "_").lower()


@dataclass(frozen=True, slots=True)
class PinnedSnapshot:
    """One exact snapshot a protocol reproduces rather than merely accepts."""

    record_id: str
    snapshot_version: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """What a protocol needs from a snapshot, and optionally which snapshot.

    A template leaves ``pinned`` unset so it runs against any conforming
    snapshot; the run bundle still records the snapshot's real digest.
    """

    target: str
    target_contract: str
    requires: tuple[str, ...]
    target_units: str | None = None
    pinned: PinnedSnapshot | None = None


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Deterministic dataset partitioning configuration."""

    method: SplitMethod
    train: float
    validation: float
    calibration: float
    test: float
    seed: int
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
class ArtifactDependency:
    """A released model artifact a feature contract needs, pinned by digest.

    The k-distance feature contract embeds the metallicity model's learned
    representation, so the exact checkpoint is part of the feature definition.
    """

    name: str
    record_id: str
    file: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The feature contract a protocol selects, and how it is configured.

    ``parameters`` is validated by the contract, not by this schema, for the
    same reason ``model.parameters`` is validated by the trainer.
    """

    schema: str
    parameters: dict[str, Any]
    depends_on: tuple[ArtifactDependency, ...] = ()


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
    min_recall: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingProtocol:
    """Validated configuration for one reproducible training workflow."""

    schema_version: int
    id: str
    release: ReleaseName
    task: Task
    trainer: str
    dataset: DatasetSpec
    split: SplitSpec
    features: FeatureSpec
    model: ModelSpec
    evaluation: EvaluationSpec
    source: Path


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
    "target",
    "target_contract",
    "target_units",
    "requires",
    "record_id",
    "snapshot_version",
    "manifest_sha256",
}
_PIN_KEYS = ("record_id", "snapshot_version", "manifest_sha256")
_SPLIT_KEYS = {
    "method",
    "stratify",
    "train",
    "validation",
    "calibration",
    "test",
    "seed",
}
_FEATURE_KEYS = {"schema", "parameters", "depends_on"}
_DEPENDENCY_KEYS = {"record_id", "file", "sha256"}
_MODEL_KEYS = {"seed", "parameters"}
_EVALUATION_KEYS = {
    "primary_metric",
    "metrics",
    "baseline",
    "threshold_metric",
    "positive_label",
    "min_recall",
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

    raw_requires = table.get("requires", [])
    if not isinstance(raw_requires, list) or any(
        not isinstance(item, str) for item in raw_requires
    ):
        raise ValueError("dataset.requires must be an array of strings")
    requires = tuple(raw_requires)
    if len(requires) != len(set(requires)):
        raise ValueError("dataset.requires must be unique")
    unknown = sorted(set(requires) - CAPABILITIES)
    if unknown:
        raise ValueError(
            f"unknown dataset.requires value(s): {', '.join(unknown)}; "
            f"known: {', '.join(sorted(CAPABILITIES))}"
        )

    target_units = table.get("target_units")
    if target_units is not None and (
        not isinstance(target_units, str) or not target_units.strip()
    ):
        raise ValueError("dataset.target_units must be a non-empty string")

    present = [field for field in _PIN_KEYS if field in table]
    if present and len(present) != len(_PIN_KEYS):
        missing = [field for field in _PIN_KEYS if field not in table]
        raise ValueError(
            "pinning a snapshot needs every one of "
            f"{', '.join(_PIN_KEYS)}; missing {', '.join(missing)}"
        )
    pinned = None
    if present:
        manifest_sha256 = _string(table, "manifest_sha256", "dataset")
        if not is_sha256(manifest_sha256):
            raise ValueError(
                "dataset.manifest_sha256 must be a lowercase SHA-256 digest"
            )
        pinned = PinnedSnapshot(
            record_id=_string(table, "record_id", "dataset"),
            snapshot_version=_string(table, "snapshot_version", "dataset"),
            manifest_sha256=manifest_sha256,
        )

    return DatasetSpec(
        target=_string(table, "target", "dataset"),
        target_contract=_string(table, "target_contract", "dataset"),
        requires=requires,
        target_units=target_units,
        pinned=pinned,
    )


def _load_split(root: dict[str, Any], task: str) -> SplitSpec:
    table = _table(root.get("split"), "split")
    _reject_unknown(table, _SPLIT_KEYS, "split")
    method = _string(table, "method", "split")
    if method not in SPLIT_METHODS:
        raise ValueError("split.method must be random or group")
    stratify = table.get("stratify", False)
    if not isinstance(stratify, bool):
        raise ValueError("split.stratify must be a boolean")
    if stratify and task != "classification":
        raise ValueError("split.stratify is only valid for classification")
    split = SplitSpec(
        method=cast(SplitMethod, method),
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


def _load_dependency(name: str, value: object) -> ArtifactDependency:
    table = _table(value, f"features.depends_on.{name}")
    _reject_unknown(table, _DEPENDENCY_KEYS, f"features.depends_on.{name}")
    section = f"features.depends_on.{name}"
    sha256 = _string(table, "sha256", section)
    if not is_sha256(sha256):
        raise ValueError(f"{section}.sha256 must be a lowercase SHA-256 digest")
    file_name = _string(table, "file", section)
    if Path(file_name).name != file_name:
        raise ValueError(f"{section}.file must be a basename")
    return ArtifactDependency(
        name=name,
        record_id=_string(table, "record_id", section),
        file=file_name,
        sha256=sha256,
    )


def _load_features(root: dict[str, Any]) -> FeatureSpec:
    table = _table(root.get("features"), "features")
    _reject_unknown(table, _FEATURE_KEYS, "features")
    parameters = table.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("features.parameters must be a TOML table")
    raw_dependencies = table.get("depends_on", {})
    if not isinstance(raw_dependencies, dict):
        raise ValueError("features.depends_on must be a TOML table")
    dependencies = tuple(
        _load_dependency(name, value)
        for name, value in sorted(raw_dependencies.items())
    )
    return FeatureSpec(
        schema=_string(table, "schema", "features"),
        parameters=parameters,
        depends_on=dependencies,
    )


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
    min_recall = table.get("min_recall")
    if min_recall is not None:
        # A recall floor states the error the protocol refuses to make. The
        # threshold metric then chooses among the thresholds that honour it.
        if threshold_metric is None:
            raise ValueError(
                "evaluation.min_recall requires evaluation.threshold_metric"
            )
        if isinstance(min_recall, bool) or not isinstance(min_recall, int | float):
            raise ValueError("evaluation.min_recall must be a number")
        min_recall = float(min_recall)
        if not 0.0 < min_recall <= 1.0:
            raise ValueError("evaluation.min_recall must lie in (0, 1]")
        if "recall" not in metrics:
            raise ValueError("evaluation.min_recall requires recall in metrics")
    return EvaluationSpec(
        primary_metric=primary_metric,
        metrics=metrics,
        baseline=baseline,
        threshold_metric=threshold_metric,
        positive_label=positive_label,
        min_recall=min_recall,
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
    release = parse_release_name(protocol_id)
    task = _string(root, "task", "protocol")
    if task not in TASKS:
        raise ValueError("protocol.task must be regression or classification")
    trainer = _string(root, "trainer", "protocol")

    dataset = _load_dataset(root)
    if dataset.pinned is not None:
        expected = dataset_segment(dataset.pinned.record_id)
        if release.dataset != expected:
            # The name says which data this learned from, so it cannot be
            # allowed to say something the pin contradicts.
            raise ValueError(
                f"protocol.id names dataset {release.dataset!r} but "
                f"dataset.record_id is {dataset.pinned.record_id!r}, which "
                f"belongs in the name as {expected!r}"
            )

    return TrainingProtocol(
        schema_version=1,
        id=protocol_id,
        release=release,
        task=cast(Task, task),
        trainer=trainer,
        dataset=dataset,
        split=_load_split(root, task),
        features=_load_features(root),
        model=_load_model(root),
        evaluation=_load_evaluation(root, task),
        source=path,
    )
