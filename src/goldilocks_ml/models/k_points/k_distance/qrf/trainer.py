"""Seeded quantile-random-forest training for QRF95-compatible artifacts."""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn_quantile import RandomForestQuantileRegressor

from goldilocks_ml.evaluation import (
    pinball_loss,
    select_band_offsets,
    select_decision_level,
)
from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.protocol import TrainingProtocol
from goldilocks_ml.registry import (
    FeatureMatrix,
    FittedModel,
    TrainingContext,
    TrainingPartition,
    register_trainer,
)
from goldilocks_ml.snapshot import Sample

TRAINER = "quantile_random_forest"
# The serving runtime that reads these artifacts back. It is versioned
# separately from the trainer: the same fitting algorithm can produce a
# model with a different feature schema, calibration, or output contract.
RUNTIME = "k_points.k_distance.qrf"
RUNTIME_VERSION = 1
KINDEX_RUNTIME = "k_points.k_index.qrf"
KINDEX_RUNTIME_VERSION = 1
RECORD_SCHEMA_VERSION = 1
CALIBRATION_METHOD = "split_conformal_quantile_regression"
ENDPOINT_ADJUSTMENT = "clamped_to_include_median"
MODEL_FILE = "QRF95.pkl"
KINDEX_MODEL_FILE = "k_index_qrf.pkl"
CALIBRATION_FILE = "calibration.json"
MODEL_RECORD_FILE = "model.json"
# Relative slack when checking that fitted quantiles come back in order. Two
# adjacent levels that resolve to the same training label can differ by a few
# parts in ten thousand of accumulated float32 error in the weighted quantile;
# a genuinely inverted pair is off by a meaningful fraction of the target.
ORDER_TOLERANCE = 1e-3
RUNTIME_SETTINGS = {
    RUNTIME: (RUNTIME_VERSION, MODEL_FILE),
    KINDEX_RUNTIME: (KINDEX_RUNTIME_VERSION, KINDEX_MODEL_FILE),
}


def prediction_matrix(
    estimator: RandomForestQuantileRegressor,
    rows: Sequence[Sequence[float]],
    levels: int = 3,
) -> np.ndarray:
    """Return one row of predictions per fitted quantile level, in q order."""
    values = np.asarray(estimator.predict(np.asarray(rows, dtype=float)), dtype=float)
    if values.ndim == 1:
        values = values.reshape(levels, -1)
    if values.shape[0] != levels and values.shape[1] == levels:
        values = values.T
    if values.shape != (levels, len(rows)):
        raise ValueError(
            f"quantile forest returned shape {values.shape}; "
            f"expected ({levels}, {len(rows)})"
        )
    if not np.isfinite(values).all():
        raise ValueError("quantile forest returned non-finite predictions")
    # The forest returns float32, and two adjacent levels can land on the same
    # training label yet disagree in its last digits. That is noise, not an
    # inverted quantile.
    tolerance = ORDER_TOLERANCE * max(1.0, float(np.abs(values).max()))
    if any(
        np.any(values[index] > values[index + 1] + tolerance)
        for index in range(len(values) - 1)
    ):
        raise ValueError("quantile forest returned unordered predictions")
    return values


def publish(value: float, decision: dict[str, Any] | None) -> float:
    """Apply a decision rule's rounding and band offset to one raw quantile.

    Shared by the trainer and every serving runtime, so the number a run scores
    and the number a consumer receives are produced by the same three lines.
    """
    if not decision:
        return value
    published = value
    if decision.get("rounding") == "half_up":
        published = float(math.floor(published + 0.5))
    for band in decision.get("bands") or []:
        upper = band.get("upper")
        if upper is None or published < float(upper):
            return published + float(band["offset"])
    return published


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


def calibrate_interval(
    lower: float, median: float, upper: float, correction: float
) -> tuple[float, float, float]:
    """Apply the conformal correction to one interval, then clamp its endpoints.

    Conformal quantile regression calibrates the outer pair only. It makes no
    claim about the median, which the forest estimates independently, so a
    negative correction can narrow the interval past the median, and where the
    raw interval is narrower than twice the correction it can invert it
    outright.

    Clamping each endpoint to the median settles both: the lower bound may only
    move down to reach it and the upper bound may only move up. This is not a
    sort -- an inverted pair collapses onto the median rather than swapping
    ends -- and the median itself never moves.

    Coverage cannot fall. Where the calibrated pair is ordered the clamped
    interval contains it, and where it is inverted it covers nothing to begin
    with. Measured on held-out data the cost is nil: test coverage moves from
    89.4% to 89.5% and the mean interval width does not move at all, while the
    guarantee that a consumer can read the median as a point inside its own
    interval becomes unconditional.
    """
    low, high = lower - correction, upper + correction
    return min(low, median), median, max(high, median)


@dataclass(frozen=True, slots=True)
class QuantileRandomForestModel:
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
    feature_parameters: dict[str, Any]
    requires_artifacts: tuple[dict[str, str], ...]
    hyperparameters: dict[str, Any]
    calibration_count: int
    calibration_mean_width: float
    selection: dict[str, Any]
    runtime: str
    runtime_version: int
    model_file: str
    levels: tuple[float, ...]
    decision: dict[str, Any] | None

    def predict_quantiles(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[tuple[float, float, float]]:
        """Return the calibrated interval around the value this model publishes.

        The middle element is what the model actually recommends, and it is the
        median only where no decision rule was declared. Where one was, it is
        the quantile that rule chose, and the interval stays what it always
        was: a diagnostic about the spread, not the answer.
        """
        if features.columns != self.feature_columns:
            raise ValueError("prediction feature columns differ from the fitted model")
        raw = prediction_matrix(
            self.estimator, features.matrix(samples), len(self.levels)
        )
        low, mid, high = (self.levels.index(level) for level in self.quantiles)
        published = (
            self.levels.index(float(self.decision["level"]))
            if self.decision is not None
            else mid
        )
        rows: list[tuple[float, float, float]] = []
        for index in range(len(samples)):
            lower, _, upper = calibrate_interval(
                float(raw[low, index]),
                float(raw[mid, index]),
                float(raw[high, index]),
                self.correction,
            )
            rows.append(
                (lower, publish(float(raw[published, index]), self.decision), upper)
            )
        return rows

    def predict(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[float]:
        """Return the value this model publishes for each sample."""
        return [value for _, value, _ in self.predict_quantiles(samples, features)]

    def describe(self) -> dict[str, Any]:
        """Return the estimator, feature, and calibration provenance."""
        return {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "runtime": {"id": self.runtime, "version": self.runtime_version},
            "trainer": TRAINER,
            "task": "regression",
            "seed": self.seed,
            "deterministic": True,
            "hyperparameters": self.hyperparameters,
            "quantiles": list(self.quantiles),
            # Every level the estimator was fitted with, in the order it
            # returns them. A serving runtime needs this to find the column its
            # decision rule names among the ones it does not publish.
            "levels": list(self.levels),
            "target": {
                "name": self.target_name,
                "contract": self.target_contract,
                "units": self.target_units,
            },
            "feature_schema": self.feature_schema,
            "feature_columns": list(self.feature_columns),
            "feature_parameters": self.feature_parameters,
            "requires_artifacts": [dict(item) for item in self.requires_artifacts],
            "selection": self.selection,
            "calibration": {
                "method": CALIBRATION_METHOD,
                "coverage": self.coverage,
                "correction": self.correction,
                "sample_count": self.calibration_count,
                "mean_interval_width": self.calibration_mean_width,
                "applied_as": "lower - correction, upper + correction",
                "endpoint_adjustment": ENDPOINT_ADJUSTMENT,
                "median_adjusted": False,
                "notes": (
                    "Calibration adjusts the outer quantiles only. A negative "
                    "correction can narrow an interval past the median, so each "
                    "endpoint is then clamped to it: lower = min(lower, median), "
                    "upper = max(upper, median). The median is never moved, and "
                    "the reported interval always contains it."
                ),
            },
            "decision": dict(self.decision) if self.decision else None,
            "artifacts": {
                "estimator": self.model_file,
                "calibration": CALIBRATION_FILE,
            },
        }

    def save(self, directory: Path) -> None:
        """Write the estimator, its digest, and a separate calibration record."""
        with (directory / self.model_file).open("wb") as handle:
            pickle.dump(self.estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Loading unpickles this file, which executes code, so the record must
        # pin what it is allowed to unpickle.
        record = self.describe()
        record["artifacts"]["estimator_sha256"] = sha256_file(
            directory / self.model_file
        )
        calibration = record["calibration"]
        (directory / CALIBRATION_FILE).write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / MODEL_RECORD_FILE).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _parameters(protocol: TrainingProtocol) -> dict[str, Any]:
    values = protocol.model.parameters
    unknown = sorted(
        set(values)
        - {"n_estimators", "quantiles", "n_jobs", "search", "decision_levels"}
    )
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
        raise ValueError(
            "quantile forest quantiles must be symmetric around the 0.5 median"
        )

    n_jobs = values.get("n_jobs")
    if n_jobs is not None and (
        not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0
    ):
        raise ValueError("model.parameters.n_jobs must be a non-zero integer")
    raw_levels = values.get("decision_levels")
    if raw_levels is None:
        decision_levels: tuple[float, ...] = ()
    else:
        # The candidates a decision rule may choose between. They are fitted
        # alongside the interval quantiles, so choosing one costs no extra fit.
        if not isinstance(raw_levels, list) or not raw_levels:
            raise ValueError(
                "model.parameters.decision_levels must be a non-empty list"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < float(value) < 1
            for value in raw_levels
        ):
            raise ValueError(
                "model.parameters.decision_levels must lie strictly in (0, 1)"
            )
        decision_levels = tuple(sorted({float(value) for value in raw_levels}))
    return {
        "n_estimators": n_estimators,
        "quantiles": quantiles,
        "n_jobs": n_jobs,
        "decision_levels": decision_levels,
        "levels": tuple(sorted(set(quantiles) | set(decision_levels))),
        "search": _search_grid(values.get("search", {})),
    }


def _search_grid(table: object) -> list[dict[str, Any]]:
    """Expand the validation search table into candidate settings.

    An absent or empty table yields one candidate with the estimator defaults,
    so a protocol that does not want a search does not pay for one.
    """
    if not isinstance(table, dict):
        raise ValueError("model.parameters.search must be a TOML table")
    unknown = sorted(set(table) - set(SEARCHABLE))
    if unknown:
        raise ValueError(
            f"unsearchable parameter(s): {', '.join(unknown)}; "
            f"searchable: {', '.join(sorted(SEARCHABLE))}"
        )
    axes: dict[str, list[Any]] = {}
    for name, values in table.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"model.parameters.search.{name} must be a non-empty list")
        for value in values:
            SEARCHABLE[name](name, value)
        axes[name] = values
    if not axes:
        return [{}]
    names = sorted(axes)
    return [
        dict(zip(names, combination, strict=True))
        for combination in product(*(axes[name] for name in names))
    ]


def _check_leaf(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"search.{name} values must be positive integers")


def _check_features(name: str, value: Any) -> None:
    if isinstance(value, str):
        if value not in {"sqrt", "log2"}:
            raise ValueError(f"search.{name} strings must be 'sqrt' or 'log2'")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"search.{name} values must be numbers or 'sqrt'/'log2'")
    if not 0 < float(value) <= 1:
        raise ValueError(f"search.{name} fractions must lie in (0, 1]")


SEARCHABLE = {
    "min_samples_leaf": _check_leaf,
    "max_features": _check_features,
}


def _build(parameters: dict[str, Any], seed: int, candidate: dict[str, Any]):
    return RandomForestQuantileRegressor(
        n_estimators=parameters["n_estimators"],
        q=list(parameters["levels"]),
        random_state=seed,
        n_jobs=parameters["n_jobs"],
        **candidate,
    )


def _validation_score(
    estimator: RandomForestQuantileRegressor,
    partition: TrainingPartition,
    quantiles: tuple[float, float, float],
    levels: tuple[float, ...],
) -> float:
    """Return mean pinball loss over the three interval quantiles."""
    raw = prediction_matrix(
        estimator, partition.features.matrix(partition.samples), len(levels)
    )
    truth = [float(sample.target) for sample in partition.samples]
    return sum(
        pinball_loss(truth, list(raw[levels.index(level)]), level)
        for level in quantiles
    ) / len(quantiles)


def fit(protocol: TrainingProtocol, context: TrainingContext) -> FittedModel:
    """Select on validation, fit on train, calibrate on calibration."""
    if protocol.task != "regression":
        raise ValueError("quantile_random_forest requires a regression protocol")
    if context.calibration is None or not context.calibration.samples:
        raise ValueError("quantile_random_forest requires a calibration split")
    parameters = _parameters(protocol)
    try:
        runtime_version, model_file = RUNTIME_SETTINGS[protocol.release.runtime]
    except KeyError:
        supported = ", ".join(sorted(RUNTIME_SETTINGS))
        raise ValueError(
            f"quantile_random_forest cannot produce runtime "
            f"{protocol.release.runtime!r}; supported: {supported}"
        ) from None
    quantiles = parameters["quantiles"]
    candidates = parameters["search"]
    if len(candidates) > 1 and (
        context.validation is None or not context.validation.samples
    ):
        raise ValueError(
            "searching hyperparameters requires a non-empty validation split"
        )

    train_rows = np.asarray(
        context.train.features.matrix(context.train.samples), dtype=float
    )
    train_targets = np.asarray(
        [float(sample.target) for sample in context.train.samples]
    )

    # Every candidate is fitted on train alone and scored on validation alone;
    # calibration and test are untouched until the winner is chosen.
    trials: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], Any] | None = None
    for candidate in candidates:
        estimator = _build(parameters, protocol.model.seed, candidate)
        estimator.fit(train_rows, train_targets)
        if context.validation is None or not context.validation.samples:
            score = float("nan")
        else:
            score = _validation_score(
                estimator, context.validation, quantiles, parameters["levels"]
            )
        trials.append({"parameters": dict(candidate), "validation_pinball_loss": score})
        if best is None or score < best[0]:
            best = (score, candidate, estimator)
    assert best is not None
    selected_score, selected, estimator = best

    levels = parameters["levels"]
    low, mid, high = (levels.index(level) for level in quantiles)
    decision = _decision(protocol, estimator, context.validation, levels)

    calibration = context.calibration
    raw = prediction_matrix(
        estimator, calibration.features.matrix(calibration.samples), len(levels)
    )
    coverage = quantiles[2] - quantiles[0]
    correction = conformal_correction(
        [float(sample.target) for sample in calibration.samples],
        raw[low],
        raw[high],
        coverage=coverage,
    )
    calibrated = [
        calibrate_interval(
            float(raw[low, index]),
            float(raw[mid, index]),
            float(raw[high, index]),
            correction,
        )
        for index in range(len(calibration.samples))
    ]
    mean_width = sum(high - low for low, _, high in calibrated) / len(calibrated)

    return QuantileRandomForestModel(
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
        feature_parameters=dict(protocol.features.parameters),
        requires_artifacts=tuple(
            {
                "name": dependency.name,
                "record_id": dependency.record_id,
                "file": dependency.file,
                "sha256": dependency.sha256,
            }
            for dependency in protocol.features.depends_on
        ),
        hyperparameters={
            "n_estimators": parameters["n_estimators"],
            "quantiles": list(quantiles),
            "n_jobs": parameters["n_jobs"],
            **selected,
        },
        calibration_count=len(calibration.samples),
        calibration_mean_width=mean_width,
        selection={
            "criterion": "mean pinball loss over the three quantiles",
            "selected_on": "validation",
            "selected": dict(selected),
            "validation_pinball_loss": selected_score,
            "trials": trials,
        },
        runtime=protocol.release.runtime,
        runtime_version=runtime_version,
        model_file=model_file,
        levels=levels,
        decision=decision,
    )


def _decision(
    protocol: TrainingProtocol,
    estimator: RandomForestQuantileRegressor,
    validation: TrainingPartition | None,
    levels: tuple[float, ...],
) -> dict[str, Any] | None:
    """Choose which fitted quantile this model publishes as its one value.

    Without a declared rule the model publishes its median, which is what every
    artifact fitted before this existed does. That is a choice too; declaring
    one makes it a measured choice instead of a default.
    """
    metric = protocol.evaluation.decision_metric
    if metric is None:
        return None
    if validation is None or not validation.samples:
        raise ValueError("a decision rule must be chosen on a non-empty validation set")
    candidates = protocol.model.parameters.get("decision_levels")
    if not candidates:
        raise ValueError(
            "evaluation.decision_metric requires model.parameters.decision_levels"
        )
    raw = prediction_matrix(
        estimator, validation.features.matrix(validation.samples), len(levels)
    )
    truth = [float(sample.target) for sample in validation.samples]
    chosen = select_decision_level(
        truth,
        {level: list(raw[levels.index(level)]) for level in sorted(set(candidates))},
        metric=metric,
        max_underprediction=protocol.evaluation.max_underprediction,
    )
    decision = {**chosen, "selected_on": "validation"}
    edges = protocol.evaluation.decision_bands
    if edges is not None:
        # A whole-step lift per band, so the floor is honoured inside the part
        # of the range where the model is weakest and not only on average.
        # Integer offsets only make sense on an integer grid, so this is also
        # where the published value becomes a whole step.
        decision["bands"] = select_band_offsets(
            truth,
            list(raw[levels.index(float(chosen["level"]))]),
            edges,
            max_underprediction=protocol.evaluation.max_underprediction,
        )
        decision["rounding"] = "half_up"
    return decision


register_trainer(TRAINER, fit)
