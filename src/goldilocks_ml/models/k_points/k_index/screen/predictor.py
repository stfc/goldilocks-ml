"""Serve the dense-mesh screen for candidate structures.

A screening model is read as a ranking, so what it returns is the score
alongside the class a plain cut would give it. The class is the lesser half:
a campaign sorts by ``score`` and takes as many as its machine time allows,
and the record's measured precision at that fraction is what says how many of
them will be worth having.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.inference import ModelPrediction, contract_for
from goldilocks_ml.models.k_points.k_index.screen.trainer import (
    RUNTIME,
    RUNTIME_VERSION,
)
from goldilocks_ml.registry import register_predictor

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

# The cut used when a caller asks for a class rather than a score. It is not
# an operating point the model was tuned for -- the record carries no single
# threshold, because a campaign's budget is what decides where the line falls.
NOMINAL_CUT = 0.5


@dataclass(frozen=True, slots=True)
class DenseMeshScreenPredictor:
    """A verified screening classifier and its ranking record."""

    estimator: Any
    record: Mapping[str, Any]
    model_id: str
    threshold: float

    def predict(self, structure: Structure) -> ModelPrediction:
        """Return the screen's verdict for one structure."""
        return self.predict_batch([structure])[0]

    def predict_batch(self, structures: Sequence[Structure]) -> list[ModelPrediction]:
        """Return one verdict per structure, featurising them together."""
        from goldilocks_ml.models.k_points.k_index.qrf import features

        if not structures:
            return []
        rows = features.feature_rows(structures)
        scores = [float(value) for value in self.estimator.predict_proba(rows)[:, 1]]
        target_contract = self.record["target"]["contract"]
        contract = contract_for(target_contract)
        classes = self.record["classes"]
        decision = self.record["decision"]

        predictions: list[ModelPrediction] = []
        for score in scores:
            is_dense = score >= NOMINAL_CUT
            predictions.append(
                ModelPrediction(
                    parameter=contract.parameter,
                    quantity=contract.quantity,
                    value=is_dense,
                    target_contract=target_contract,
                    model_id=self.model_id,
                    # A ranking score is not a coverage guarantee, and nothing
                    # here proves it is calibrated as a probability.
                    confidence=None,
                    details={
                        "score": score,
                        "label": classes["positive"]
                        if is_dense
                        else classes["negative"],
                        "nominal_cut": NOMINAL_CUT,
                        "rung_threshold": self.threshold,
                        "rule": decision["rule"],
                        # What the ranking was measured to deliver, so a caller
                        # sizing a campaign does not have to re-derive it.
                        "measured_fractions": decision["fractions"],
                    },
                )
            )
            contract.check_value(is_dense)
        return predictions


def load(
    record: Mapping[str, Any], directory: Path, artifacts: Mapping[str, Path]
) -> DenseMeshScreenPredictor:
    """Load an integrity-checked screen without trusting its pickle."""
    from goldilocks_ml.models.k_points.k_index.qrf import features

    if artifacts:
        raise ValueError("the dense-mesh screen has no artifact dependencies")
    version = record.get("runtime", {}).get("version")
    if version != RUNTIME_VERSION:
        raise ValueError(
            f"this artifact declares {RUNTIME} runtime version {version!r}; "
            f"this build implements version {RUNTIME_VERSION}"
        )
    schema = record["feature_schema"]
    if schema != features.SCHEMA:
        raise ValueError(
            f"this artifact was built against feature contract {schema!r}, but "
            f"this build provides {features.SCHEMA!r}"
        )
    recorded = tuple(record["feature_columns"])
    if recorded != features.column_names():
        raise ValueError("the recorded CSLR columns differ from this build's contract")

    # The rung the screen cuts at is what the model means. A record that does
    # not carry it describes an unknown question, and its scores cannot be
    # compared with anything.
    derived = record.get("derived_from") or {}
    if derived.get("rule") != "at_or_above" or derived.get("threshold") is None:
        raise ValueError(
            "this artifact does not record the rung it screens at; a screening "
            "model's threshold is part of what its scores mean"
        )
    decision = record.get("decision") or {}
    if decision.get("rule") != "ranking":
        raise ValueError(
            f"this build serves the 'ranking' decision rule; the artifact "
            f"records {decision.get('rule')!r}"
        )
    if not decision.get("fractions"):
        raise ValueError(
            "the record carries no measured ranking quality; a screen without "
            "it cannot tell a campaign what its budget buys"
        )

    estimator_file = record["artifacts"]["estimator"]
    estimator_path = Path(directory) / estimator_file
    pinned = record["artifacts"].get("estimator_sha256")
    if not pinned:
        raise ValueError(
            f"the record does not pin a SHA-256 for {estimator_file}; refusing "
            "to unpickle an unverified estimator"
        )
    digest = sha256_file(estimator_path)
    if digest != pinned:
        raise ValueError(
            f"{estimator_file} has SHA-256 {digest}; its record pins {pinned}"
        )
    with estimator_path.open("rb") as handle:
        estimator = pickle.load(handle)

    expected_width = len(recorded)
    actual_width = getattr(estimator, "n_features_in_", expected_width)
    if actual_width != expected_width:
        raise ValueError(
            f"{estimator_file} takes {actual_width} features but its record "
            f"declares {expected_width}"
        )
    return DenseMeshScreenPredictor(
        estimator=estimator,
        record=record,
        model_id=f"{RUNTIME}@{schema}",
        threshold=float(derived["threshold"]),
    )


register_predictor(RUNTIME, load)
