"""The trainer and feature-contract interfaces, and the registries that find them.

A protocol names its trainer and feature contract; this module resolves the
names. Each model registers its own, so that adding one never edits the shared
pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from goldilocks_ml.protocol import TrainingProtocol
    from goldilocks_ml.snapshot import Sample, Snapshot


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Model inputs for a whole snapshot, keyed by stable sample id."""

    columns: tuple[str, ...]
    rows: Mapping[str, tuple[float, ...]]

    def matrix(self, samples: Sequence[Sample]) -> list[tuple[float, ...]]:
        """Return feature rows aligned to ``samples``, in their order."""
        try:
            return [self.rows[sample.sample_id] for sample in samples]
        except KeyError as error:
            raise ValueError(
                f"the feature contract produced no row for {error.args[0]}"
            ) from error

    def subset(self, samples: Sequence[Sample]) -> FeatureMatrix:
        """Return only the rows named by a sequence of samples."""
        return FeatureMatrix(
            columns=self.columns,
            rows={sample.sample_id: self.rows[sample.sample_id] for sample in samples},
        )

    def validate(self, snapshot: Snapshot) -> None:
        """Reject a contract that skipped samples or changed its own width."""
        missing = sorted(set(snapshot.sample_ids) - set(self.rows))
        if missing:
            raise ValueError(
                f"the feature contract produced no row for {len(missing)} sample(s), "
                f"starting with {missing[0]}"
            )
        width = len(self.columns)
        for sample_id, values in self.rows.items():
            if len(values) != width:
                raise ValueError(
                    f"{sample_id} has {len(values)} features; expected {width}"
                )


@dataclass(frozen=True, slots=True)
class TrainingPartition:
    """Samples and features from one explicitly named non-test split."""

    samples: tuple[Sample, ...]
    features: FeatureMatrix


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Train, validation, and calibration data; test data is never exposed."""

    train: TrainingPartition
    validation: TrainingPartition | None
    calibration: TrainingPartition | None
    artifacts: Mapping[str, Path]
    output_dir: Path


@runtime_checkable
class FittedModel(Protocol):
    """A trained model that can predict, describe itself, and be serialised."""

    def predict(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[float]:
        """Return regression values or positive-class scores, one per sample."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return the JSON-serialisable record of what was fitted."""
        ...

    def save(self, directory: Path) -> None:
        """Write the model artifacts into an existing directory."""
        ...


# Feature extraction must be stateless across samples. Any fitted preprocessing
# belongs inside the trainer, where split boundaries are explicit.
Trainer = Callable[["TrainingProtocol", TrainingContext], FittedModel]
FeatureContract = Callable[
    ["TrainingProtocol", "Snapshot", Mapping[str, Path]], FeatureMatrix
]

_TRAINERS: dict[str, Trainer] = {}
_FEATURES: dict[str, FeatureContract] = {}


def register_trainer(name: str, trainer: Trainer) -> None:
    """Register a trainer under the stable name a protocol selects."""
    if name in _TRAINERS:
        raise ValueError(f"trainer {name} is already registered")
    _TRAINERS[name] = trainer


def register_feature_contract(name: str, contract: FeatureContract) -> None:
    """Register a feature contract under the stable name a protocol selects."""
    if name in _FEATURES:
        raise ValueError(f"feature contract {name} is already registered")
    _FEATURES[name] = contract


def get_trainer(name: str) -> Trainer:
    """Return the trainer a protocol selected."""
    try:
        return _TRAINERS[name]
    except KeyError:
        known = ", ".join(sorted(_TRAINERS)) or "none"
        raise ValueError(f"unknown trainer {name!r}; registered: {known}") from None


def get_feature_contract(name: str) -> FeatureContract:
    """Return the feature contract a protocol selected."""
    try:
        return _FEATURES[name]
    except KeyError:
        known = ", ".join(sorted(_FEATURES)) or "none"
        raise ValueError(
            f"unknown feature contract {name!r}; registered: {known}"
        ) from None


def trainer_names() -> tuple[str, ...]:
    """Return every registered trainer name."""
    return tuple(sorted(_TRAINERS))


def feature_contract_names() -> tuple[str, ...]:
    """Return every registered feature contract name."""
    return tuple(sorted(_FEATURES))
