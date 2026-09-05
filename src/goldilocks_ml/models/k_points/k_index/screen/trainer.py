"""Fit the dense-mesh screening classifier.

A screening model's decision is not a threshold. A classifier that serves a
person asks "is this one positive"; this one serves a budget, and the question
is "which N of these should we spend machine time on". Ranking is the whole
output, so what the record carries is the ranking's precision and recall at the
budgets a campaign might actually choose, not an operating point pretending to
be the only one.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.protocol import TrainingProtocol
from goldilocks_ml.registry import (
    FeatureMatrix,
    FittedModel,
    TrainingContext,
    register_trainer,
)
from goldilocks_ml.snapshot import Sample

TRAINER = "dense_mesh_screen"
RUNTIME = "k_points.k_index.screen"
RUNTIME_VERSION = 1
RECORD_SCHEMA_VERSION = 1
MODEL_FILE = "k_index_screen.pkl"
RANKING_FILE = "ranking.json"

# The budgets the record reports precision and recall at, as fractions of the
# ranked pool rather than counts. A campaign ranks a pool whose size it chooses
# and takes the top N of it; a count measured on a validation split of a
# different size does not transfer, and saturates once N exceeds the split.
# A fraction does: taking 2000 of 13801 candidates is taking the top 14.5%.
REPORTED_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.25)


def ranking_quality(
    truth: Sequence[bool], scores: Sequence[float], fractions: Sequence[float]
) -> list[dict[str, Any]]:
    """Return precision and recall when the top fraction by score is taken."""
    if len(truth) != len(scores):
        raise ValueError("ranking quality needs one score per sample")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("ranking fractions must lie in (0, 1]")
    order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
    positives = sum(1 for value in truth if value)
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        taken = order[: max(1, round(fraction * len(order)))]
        hits = sum(1 for index in taken if truth[index])
        rows.append(
            {
                "fraction": float(fraction),
                "taken": len(taken),
                "hits": hits,
                "precision": hits / len(taken),
                "recall": hits / positives if positives else 0.0,
                # What the ranking multiplies the base rate by. A campaign
                # comparing against sampling at random reads this column.
                "enrichment": (hits / len(taken)) / (positives / len(truth))
                if positives
                else 0.0,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class DenseMeshScreen:
    """A fitted screening classifier and the record describing it."""

    estimator: Any
    classes: dict[str, str]
    decision: dict[str, Any]
    target_name: str
    target_contract: str
    threshold: float
    feature_schema: str
    feature_columns: tuple[str, ...]
    feature_parameters: Mapping[str, Any]
    hyperparameters: Mapping[str, Any]
    seed: int

    def _scores(self, features: FeatureMatrix, samples: Sequence[Sample]):
        rows = features.matrix(samples)
        return [float(value) for value in self.estimator.predict_proba(rows)[:, 1]]

    def predict(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[float]:
        """Return the positive-class score for each sample."""
        return self._scores(features, samples)

    def describe(self) -> dict[str, Any]:
        """Return the record written to ``model.json``."""
        return {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "task": "classification",
            "trainer": TRAINER,
            "runtime": {"id": RUNTIME, "version": RUNTIME_VERSION},
            "deterministic": True,
            "seed": self.seed,
            "classes": dict(self.classes),
            "decision": dict(self.decision),
            "target": {
                "name": self.target_name,
                "contract": self.target_contract,
                "units": None,
            },
            "derived_from": {
                "target": "k_index",
                "rule": "at_or_above",
                "threshold": self.threshold,
            },
            "feature_schema": self.feature_schema,
            "feature_columns": list(self.feature_columns),
            "feature_parameters": dict(self.feature_parameters),
            "hyperparameters": dict(self.hyperparameters),
            "requires_artifacts": [],
            "artifacts": {"estimator": MODEL_FILE},
        }

    def save(self, directory: Path) -> None:
        """Write the estimator, its digest, and the ranking record."""
        with (directory / MODEL_FILE).open("wb") as handle:
            pickle.dump(self.estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        record = self.describe()
        record["artifacts"]["estimator_sha256"] = sha256_file(directory / MODEL_FILE)
        (directory / RANKING_FILE).write_text(
            json.dumps(record["decision"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "model.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def fit(protocol: TrainingProtocol, context: TrainingContext) -> FittedModel:
    """Fit on train, measure the ranking on validation."""
    from sklearn.ensemble import RandomForestClassifier

    if protocol.task != "classification":
        raise ValueError(f"{TRAINER} requires a classification protocol")
    derive = protocol.dataset.derive
    if derive is None:
        raise ValueError(
            f"{TRAINER} requires dataset.derive; the screen's classes are defined "
            "by the rung they cut the recorded target at, and a record that does "
            "not name that rung does not say what it screens for"
        )

    parameters = dict(protocol.model.parameters)
    unknown = sorted(set(parameters) - {"n_estimators", "min_samples_leaf", "n_jobs"})
    if unknown:
        raise ValueError(f"unknown model parameter(s): {', '.join(unknown)}")
    n_estimators = int(parameters.get("n_estimators", 500))
    min_samples_leaf = int(parameters.get("min_samples_leaf", 1))
    n_jobs = int(parameters.get("n_jobs", -1))

    positive = derive.positive
    train_samples = context.train.samples
    labels = {str(sample.target) for sample in train_samples}
    if labels != {derive.positive, derive.negative}:
        raise ValueError(
            f"the train split carries classes {sorted(labels)}; the protocol "
            f"derives {sorted([derive.positive, derive.negative])}"
        )

    rows = context.train.features.matrix(train_samples)
    outcomes = [str(sample.target) == positive for sample in train_samples]
    # The positive class is under a tenth of the data by construction -- that
    # scarcity is the reason the campaign exists -- so the fit is balanced,
    # or the forest learns to answer "sparse" and score nothing.
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=protocol.model.seed,
        class_weight="balanced",
    )
    estimator.fit(rows, outcomes)

    if context.validation is None:
        raise ValueError(f"{TRAINER} needs a validation split to measure its ranking")
    validation = context.validation
    scores = [
        float(value)
        for value in estimator.predict_proba(
            validation.features.matrix(validation.samples)
        )[:, 1]
    ]
    truth = [str(sample.target) == positive for sample in validation.samples]
    decision = {
        "rule": "ranking",
        "selected_on": "validation",
        "positive_rate": sum(truth) / len(truth),
        "fractions": ranking_quality(truth, scores, REPORTED_FRACTIONS),
    }

    return DenseMeshScreen(
        estimator=estimator,
        classes={"positive": derive.positive, "negative": derive.negative},
        decision=decision,
        target_name="k_index_dense",
        target_contract=derive.contract,
        threshold=derive.threshold,
        feature_schema=protocol.features.schema,
        feature_columns=context.train.features.columns,
        feature_parameters=dict(protocol.features.parameters),
        hyperparameters={
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": "balanced",
        },
        seed=protocol.model.seed,
    )


register_trainer(TRAINER, fit)
