"""Dependency-free baselines over precomputed features.

These are honest reference models, not scientific ones. They train on a CPU in
milliseconds and need nothing beyond the standard library, so CI can exercise
the complete protocol workflow without a GPU, private data, or network access.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goldilocks_ml.core.protocol import TrainingProtocol
from goldilocks_ml.core.registry import (
    FeatureMatrix,
    FittedModel,
    TrainingContext,
    register_trainer,
)
from goldilocks_ml.core.snapshot import Sample


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Per-feature centring and scaling fitted on the training split only."""

    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> Standardizer:
        """Fit centring and scaling from training feature rows."""
        if not rows:
            raise ValueError("cannot fit preprocessing on an empty split")
        width = len(rows[0])
        means: list[float] = []
        scales: list[float] = []
        for index in range(width):
            column = [row[index] for row in rows]
            mean = sum(column) / len(column)
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            deviation = math.sqrt(variance)
            means.append(mean)
            scales.append(deviation if deviation > 1e-12 else 1.0)
        return cls(means=tuple(means), scales=tuple(scales))

    def apply(self, row: Sequence[float]) -> list[float]:
        """Standardise one feature row."""
        return [
            (value - mean) / scale
            for value, mean, scale in zip(row, self.means, self.scales, strict=True)
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
class _Linear:
    """Ridge-regularised least squares over standardised features."""

    features: FeatureMatrix
    standardizer: Standardizer
    weights: tuple[float, ...]
    intercept: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def _linear(self, sample: Sample) -> float:
        row = self.standardizer.apply(self.features.rows[sample.sample_id])
        return self.intercept + sum(
            weight * value for weight, value in zip(self.weights, row, strict=True)
        )

    def predict(self, samples: Sequence[Sample]) -> list[float]:
        """Return one predicted value per sample."""
        return [self._linear(sample) for sample in samples]

    def describe(self) -> dict[str, Any]:
        """Return the fitted coefficients and hyperparameters."""
        return {
            "trainer": "linear_regression",
            "task": "regression",
            "seed": self.seed,
            "deterministic": True,
            "hyperparameters": self.hyperparameters,
            "feature_columns": list(self.features.columns),
            "standardizer": {
                "means": list(self.standardizer.means),
                "scales": list(self.standardizer.scales),
            },
            "intercept": self.intercept,
            "weights": list(self.weights),
        }

    def save(self, directory: Path) -> None:
        """Write the model as a single JSON document."""
        (directory / "model.json").write_text(
            json.dumps(self.describe(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class _Logistic(_Linear):
    """Batch gradient-descent logistic regression over standardised features."""

    positive_label: str = ""

    def predict(self, samples: Sequence[Sample]) -> list[float]:
        """Return the positive-class probability for each sample."""
        return [
            1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self._linear(sample)))))
            for sample in samples
        ]

    def describe(self) -> dict[str, Any]:
        """Return the fitted coefficients and hyperparameters."""
        # @dataclass(slots=True) rebuilds the class, so zero-argument super()
        # cannot find its own cell; call the base implementation explicitly.
        return {
            **_Linear.describe(self),
            "trainer": "logistic_regression",
            "task": "classification",
            "positive_label": self.positive_label,
        }


def _rows(
    context: TrainingContext, samples: Sequence[Sample]
) -> list[tuple[float, ...]]:
    rows = context.features.matrix(samples)
    if not rows or not rows[0]:
        raise ValueError("the feature contract produced no feature columns")
    return rows


def fit_linear_regression(
    protocol: TrainingProtocol,
    samples: Sequence[Sample],
    context: TrainingContext,
) -> FittedModel:
    """Fit ridge-regularised least squares on the training split."""
    l2 = float(protocol.model.parameters.get("l2", 1e-6))
    if l2 < 0:
        raise ValueError("model.parameters.l2 must not be negative")

    rows = _rows(context, samples)
    standardizer = Standardizer.fit(rows)
    design = [standardizer.apply(row) for row in rows]
    targets = [float(sample.target) for sample in samples]
    intercept = sum(targets) / len(targets)
    centred = [value - intercept for value in targets]
    width = len(rows[0])
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
    return _Linear(
        features=context.features,
        standardizer=standardizer,
        weights=tuple(_solve(gram, moment)),
        intercept=intercept,
        hyperparameters={"l2": l2},
        seed=protocol.model.seed,
    )


def fit_logistic_regression(
    protocol: TrainingProtocol,
    samples: Sequence[Sample],
    context: TrainingContext,
) -> FittedModel:
    """Fit logistic regression by batch gradient descent on the training split."""
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

    labels = sorted({str(sample.target) for sample in samples})
    if len(labels) != 2:
        raise ValueError("logistic_regression needs exactly two training classes")
    positive = protocol.evaluation.positive_label or labels[-1]
    if positive not in labels:
        raise ValueError(f"positive label {positive!r} is absent from the train split")

    rows = _rows(context, samples)
    standardizer = Standardizer.fit(rows)
    design = [standardizer.apply(row) for row in rows]
    outcomes = [1.0 if str(sample.target) == positive else 0.0 for sample in samples]
    width = len(rows[0])
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

    return _Logistic(
        features=context.features,
        standardizer=standardizer,
        weights=tuple(weights),
        intercept=intercept,
        hyperparameters={
            "l2": l2,
            "learning_rate": learning_rate,
            "iterations": iterations,
        },
        seed=protocol.model.seed,
        positive_label=positive,
    )


register_trainer("linear_regression", fit_linear_regression)
register_trainer("logistic_regression", fit_logistic_regression)
