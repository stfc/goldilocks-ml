"""The trainer interface and the CPU-only trainers used by CI fixtures.

Model-specific trainers register themselves here; the shared pipeline never
imports a scientific task module directly. Every trainer receives the training
split alone, so learned preprocessing cannot see validation, calibration, or
test data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from goldilocks_ml.datasets import Sample
from goldilocks_ml.protocol import TrainingProtocol


@runtime_checkable
class FittedModel(Protocol):
    """A trained model that can predict, describe itself, and be serialised."""

    def predict(self, samples: Sequence[Sample]) -> list[float]:
        """Return regression values or positive-class scores, one per sample."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return the JSON-serialisable record of what was fitted."""
        ...

    def save(self, directory: Path) -> None:
        """Write the model artifacts into an existing directory."""
        ...


Trainer = Callable[[TrainingProtocol, Sequence[Sample]], FittedModel]

_TRAINERS: dict[str, Trainer] = {}


def register_trainer(name: str, trainer: Trainer) -> None:
    """Register a built-in trainer under a stable protocol name."""
    if name in _TRAINERS:
        raise ValueError(f"trainer {name} is already registered")
    _TRAINERS[name] = trainer


def get_trainer(name: str) -> Trainer:
    """Return the trainer a protocol selected."""
    try:
        return _TRAINERS[name]
    except KeyError:
        known = ", ".join(sorted(_TRAINERS)) or "none"
        raise ValueError(f"unknown trainer {name!r}; registered: {known}") from None


def trainer_names() -> tuple[str, ...]:
    """Return every registered trainer name."""
    return tuple(sorted(_TRAINERS))


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Per-feature centring and scaling fitted on the training split only."""

    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, samples: Sequence[Sample]) -> Standardizer:
        """Fit centring and scaling from training samples."""
        if not samples:
            raise ValueError("cannot fit preprocessing on an empty split")
        width = len(samples[0].features)
        means: list[float] = []
        scales: list[float] = []
        for index in range(width):
            column = [sample.features[index] for sample in samples]
            mean = sum(column) / len(column)
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            deviation = math.sqrt(variance)
            means.append(mean)
            scales.append(deviation if deviation > 1e-12 else 1.0)
        return cls(means=tuple(means), scales=tuple(scales))

    def apply(self, sample: Sample) -> list[float]:
        """Standardise one sample's features."""
        return [
            (value - mean) / scale
            for value, mean, scale in zip(
                sample.features, self.means, self.scales, strict=True
            )
        ]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small symmetric positive-definite system by Gaussian elimination."""
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("the feature matrix is singular; check the feature set")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return [augmented[index][size] for index in range(size)]


@dataclass(frozen=True, slots=True)
class LinearRegressionModel:
    """Ridge-regularised least squares over standardised features."""

    standardizer: Standardizer
    weights: tuple[float, ...]
    intercept: float
    l2: float
    seed: int
    feature_columns: tuple[str, ...]

    def predict(self, samples: Sequence[Sample]) -> list[float]:
        """Return one predicted value per sample."""
        return [
            self.intercept
            + sum(
                weight * value
                for weight, value in zip(
                    self.weights, self.standardizer.apply(sample), strict=True
                )
            )
            for sample in samples
        ]

    def describe(self) -> dict[str, Any]:
        """Return the fitted coefficients and hyperparameters."""
        return {
            "trainer": "linear_regression",
            "task": "regression",
            "seed": self.seed,
            "deterministic": True,
            "hyperparameters": {"l2": self.l2},
            "feature_columns": list(self.feature_columns),
            "standardizer": {
                "means": list(self.standardizer.means),
                "scales": list(self.standardizer.scales),
            },
            "intercept": self.intercept,
            "weights": list(self.weights),
        }

    def save(self, directory: Path) -> None:
        """Write the model as a single JSON document."""
        path = directory / "model.json"
        path.write_text(
            json.dumps(self.describe(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _fit_linear_regression(
    protocol: TrainingProtocol, samples: Sequence[Sample]
) -> FittedModel:
    l2 = float(protocol.model.parameters.get("l2", 1e-6))
    if l2 < 0:
        raise ValueError("model.parameters.l2 must not be negative")
    if not protocol.features.columns:
        raise ValueError("linear_regression needs features.columns")
    standardizer = Standardizer.fit(samples)
    design = [standardizer.apply(sample) for sample in samples]
    targets = [float(sample.target) for sample in samples]
    intercept = sum(targets) / len(targets)
    centred = [value - intercept for value in targets]
    width = len(protocol.features.columns)
    gram = [
        [
            sum(row[i] * row[j] for row in design) + (l2 if i == j else 0.0)
            for j in range(width)
        ]
        for i in range(width)
    ]
    moment = [
        sum(row[i] * value for row, value in zip(design, centred, strict=True))
        for i in range(width)
    ]
    weights = _solve(gram, moment)
    return LinearRegressionModel(
        standardizer=standardizer,
        weights=tuple(weights),
        intercept=intercept,
        l2=l2,
        seed=protocol.model.seed,
        feature_columns=protocol.features.columns,
    )


@dataclass(frozen=True, slots=True)
class LogisticRegressionModel:
    """Batch gradient-descent logistic regression over standardised features."""

    standardizer: Standardizer
    weights: tuple[float, ...]
    intercept: float
    l2: float
    learning_rate: float
    iterations: int
    positive_label: str
    seed: int
    feature_columns: tuple[str, ...]

    def predict(self, samples: Sequence[Sample]) -> list[float]:
        """Return the positive-class probability for each sample."""
        scores: list[float] = []
        for sample in samples:
            linear = self.intercept + sum(
                weight * value
                for weight, value in zip(
                    self.weights, self.standardizer.apply(sample), strict=True
                )
            )
            scores.append(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, linear)))))
        return scores

    def describe(self) -> dict[str, Any]:
        """Return the fitted coefficients and hyperparameters."""
        return {
            "trainer": "logistic_regression",
            "task": "classification",
            "seed": self.seed,
            "deterministic": True,
            "hyperparameters": {
                "l2": self.l2,
                "learning_rate": self.learning_rate,
                "iterations": self.iterations,
            },
            "positive_label": self.positive_label,
            "feature_columns": list(self.feature_columns),
            "standardizer": {
                "means": list(self.standardizer.means),
                "scales": list(self.standardizer.scales),
            },
            "intercept": self.intercept,
            "weights": list(self.weights),
        }

    def save(self, directory: Path) -> None:
        """Write the model as a single JSON document."""
        path = directory / "model.json"
        path.write_text(
            json.dumps(self.describe(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _fit_logistic_regression(
    protocol: TrainingProtocol, samples: Sequence[Sample]
) -> FittedModel:
    parameters = protocol.model.parameters
    l2 = float(parameters.get("l2", 1e-4))
    learning_rate = float(parameters.get("learning_rate", 0.1))
    iterations = int(parameters.get("iterations", 500))
    if l2 < 0:
        raise ValueError("model.parameters.l2 must not be negative")
    if learning_rate <= 0:
        raise ValueError("model.parameters.learning_rate must be positive")
    if iterations <= 0:
        raise ValueError("model.parameters.iterations must be positive")
    if not protocol.features.columns:
        raise ValueError("logistic_regression needs features.columns")

    labels = sorted({str(sample.target) for sample in samples})
    if len(labels) != 2:
        raise ValueError("logistic_regression needs exactly two training classes")
    positive = protocol.evaluation.positive_label or labels[-1]
    if positive not in labels:
        raise ValueError(f"positive label {positive!r} is absent from the train split")

    standardizer = Standardizer.fit(samples)
    design = [standardizer.apply(sample) for sample in samples]
    outcomes = [1.0 if str(sample.target) == positive else 0.0 for sample in samples]
    width = len(protocol.features.columns)
    weights = [0.0] * width
    intercept = 0.0
    count = len(samples)
    for _ in range(iterations):
        gradient = [0.0] * width
        intercept_gradient = 0.0
        for row, outcome in zip(design, outcomes, strict=True):
            linear = intercept + sum(
                weight * value for weight, value in zip(weights, row, strict=True)
            )
            probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, linear))))
            error = probability - outcome
            intercept_gradient += error
            for index, value in enumerate(row):
                gradient[index] += error * value
        intercept -= learning_rate * intercept_gradient / count
        weights = [
            weight - learning_rate * (gradient[index] / count + l2 * weight)
            for index, weight in enumerate(weights)
        ]

    return LogisticRegressionModel(
        standardizer=standardizer,
        weights=tuple(weights),
        intercept=intercept,
        l2=l2,
        learning_rate=learning_rate,
        iterations=iterations,
        positive_label=positive,
        seed=protocol.model.seed,
        feature_columns=protocol.features.columns,
    )


register_trainer("linear_regression", _fit_linear_regression)
register_trainer("logistic_regression", _fit_logistic_regression)
