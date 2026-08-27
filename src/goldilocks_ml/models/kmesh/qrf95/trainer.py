"""Seeded quantile-random-forest training for QRF95-compatible artifacts."""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn_quantile import RandomForestQuantileRegressor

from goldilocks_ml.protocol import TrainingProtocol
from goldilocks_ml.registry import (
    FeatureMatrix,
    FittedModel,
    TrainingContext,
    register_trainer,
)
from goldilocks_ml.snapshot import Sample

TRAINER = "quantile_random_forest"
MODEL_FILE = "QRF95.pkl"
CALIBRATION_FILE = "calibration.json"
MODEL_RECORD_FILE = "model.json"


def _prediction_matrix(
    estimator: RandomForestQuantileRegressor,
    rows: Sequence[Sequence[float]],
) -> np.ndarray:
    values = np.asarray(estimator.predict(np.asarray(rows, dtype=float)), dtype=float)
    if values.ndim == 1:
        values = values.reshape(3, -1)
    if values.shape[0] != 3 and values.shape[1] == 3:
        values = values.T
    if values.shape != (3, len(rows)):
        raise ValueError(
            f"quantile forest returned shape {values.shape}; expected (3, {len(rows)})"
        )
    if not np.isfinite(values).all():
        raise ValueError("quantile forest returned non-finite predictions")
    if np.any(values[0] > values[1]) or np.any(values[1] > values[2]):
        raise ValueError("quantile forest returned unordered predictions")
    return values


def conformal_correction(
    truth: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    coverage: float,
) -> float:
    """Return the finite-sample split-conformal interval correction."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    if not truth or len(truth) != len(lower) or len(truth) != len(upper):
        raise ValueError("calibration truth and interval arrays must be non-empty")
    scores = sorted(
        max(float(low) - float(actual), float(actual) - float(high))
        for actual, low, high in zip(truth, lower, upper, strict=True)
    )
    rank = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    return scores[rank - 1]


@dataclass(frozen=True, slots=True)
class QRF95Model:
    """A fitted quantile forest plus its separately recorded calibration."""

    estimator: RandomForestQuantileRegressor
    quantiles: tuple[float, float, float]
    correction: float
    coverage: float
    seed: int
    target_name: str
    target_contract: str
    target_units: str | None
    feature_schema: str
    feature_columns: tuple[str, ...]
    hyperparameters: dict[str, Any]
    calibration_count: int

    def predict_quantiles(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[tuple[float, float, float]]:
        """Return calibrated lower, median, and upper predictions."""
        if features.columns != self.feature_columns:
            raise ValueError("prediction feature columns differ from the fitted model")
        raw = _prediction_matrix(self.estimator, features.matrix(samples))
        result = [
            (
                float(lower - self.correction),
                float(median),
                float(upper + self.correction),
            )
            for lower, median, upper in zip(*raw, strict=True)
        ]
        for lower, median, upper in result:
            if not lower <= median <= upper:
                raise ValueError("calibrated QRF95 quantiles are not ordered")
        return result

    def predict(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[float]:
        """Return median k-distance predictions."""
        return [median for _, median, _ in self.predict_quantiles(samples, features)]

    def describe(self) -> dict[str, Any]:
        """Return the estimator, feature, and calibration provenance."""
        return {
            "trainer": TRAINER,
            "task": "regression",
            "seed": self.seed,
            "deterministic": True,
            "hyperparameters": self.hyperparameters,
            "quantiles": list(self.quantiles),
            "target": {
                "name": self.target_name,
                "contract": self.target_contract,
                "units": self.target_units,
            },
            "feature_schema": self.feature_schema,
            "feature_columns": list(self.feature_columns),
            "calibration": {
                "method": "split_conformal_quantile_regression",
                "coverage": self.coverage,
                "correction": self.correction,
                "sample_count": self.calibration_count,
                "applied_as": "lower - correction, upper + correction",
            },
            "artifacts": {
                "estimator": MODEL_FILE,
                "calibration": CALIBRATION_FILE,
            },
        }

    def save(self, directory: Path) -> None:
        """Write the Core-compatible estimator and separate calibration record."""
        with (directory / MODEL_FILE).open("wb") as handle:
            pickle.dump(self.estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        calibration = self.describe()["calibration"]
        (directory / CALIBRATION_FILE).write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / MODEL_RECORD_FILE).write_text(
            json.dumps(self.describe(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _parameters(protocol: TrainingProtocol) -> dict[str, Any]:
    values = protocol.model.parameters
    unknown = sorted(set(values) - {"n_estimators", "quantiles", "n_jobs"})
    if unknown:
        raise ValueError(f"unknown quantile forest parameter(s): {', '.join(unknown)}")

    n_estimators = values.get("n_estimators", 100)
    if (
        not isinstance(n_estimators, int)
        or isinstance(n_estimators, bool)
        or n_estimators <= 0
    ):
        raise ValueError("model.parameters.n_estimators must be a positive integer")

    raw_quantiles = values.get("quantiles", [0.05, 0.5, 0.95])
    if not isinstance(raw_quantiles, list) or len(raw_quantiles) != 3:
        raise ValueError("model.parameters.quantiles must contain three numbers")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in raw_quantiles
    ):
        raise ValueError("model.parameters.quantiles must contain three numbers")
    quantiles = tuple(float(value) for value in raw_quantiles)
    if not 0 < quantiles[0] < quantiles[1] < quantiles[2] < 1:
        raise ValueError(
            "model.parameters.quantiles must be strictly ordered in (0, 1)"
        )
    if quantiles[1] != 0.5 or not math.isclose(
        quantiles[0], 1.0 - quantiles[2], abs_tol=1e-12
    ):
        raise ValueError("QRF95 quantiles must be symmetric around the 0.5 median")

    n_jobs = values.get("n_jobs")
    if n_jobs is not None and (
        not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0
    ):
        raise ValueError("model.parameters.n_jobs must be a non-zero integer")
    return {
        "n_estimators": n_estimators,
        "quantiles": quantiles,
        "n_jobs": n_jobs,
    }


def fit(protocol: TrainingProtocol, context: TrainingContext) -> FittedModel:
    """Fit on train, then calibrate intervals on calibration only."""
    if protocol.task != "regression":
        raise ValueError("quantile_random_forest requires a regression protocol")
    if context.calibration is None or not context.calibration.samples:
        raise ValueError("quantile_random_forest requires a calibration split")
    parameters = _parameters(protocol)
    quantiles = parameters["quantiles"]
    estimator = RandomForestQuantileRegressor(
        n_estimators=parameters["n_estimators"],
        q=list(quantiles),
        random_state=protocol.model.seed,
        n_jobs=parameters["n_jobs"],
    )
    train_rows = context.train.features.matrix(context.train.samples)
    train_targets = [float(sample.target) for sample in context.train.samples]
    estimator.fit(np.asarray(train_rows, dtype=float), np.asarray(train_targets))

    calibration = context.calibration
    raw = _prediction_matrix(
        estimator, calibration.features.matrix(calibration.samples)
    )
    coverage = quantiles[2] - quantiles[0]
    correction = conformal_correction(
        [float(sample.target) for sample in calibration.samples],
        raw[0],
        raw[2],
        coverage=coverage,
    )
    return QRF95Model(
        estimator=estimator,
        quantiles=quantiles,
        correction=correction,
        coverage=coverage,
        seed=protocol.model.seed,
        target_name=protocol.dataset.target,
        target_contract=protocol.dataset.target_contract,
        target_units=protocol.dataset.target_units,
        feature_schema=protocol.features.schema,
        feature_columns=context.train.features.columns,
        hyperparameters={
            "n_estimators": parameters["n_estimators"],
            "quantiles": list(quantiles),
            "n_jobs": parameters["n_jobs"],
        },
        calibration_count=len(calibration.samples),
    )


register_trainer(TRAINER, fit)
