"""Serve a CSLR quantile forest that predicts zero-based k-index."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.inference import ModelPrediction, contract_for
from goldilocks_ml.models.k_points.k_distance.qrf.trainer import (
    CALIBRATION_METHOD,
    ENDPOINT_ADJUSTMENT,
    KINDEX_RUNTIME,
    KINDEX_RUNTIME_VERSION,
    calibrate_interval,
    prediction_matrix,
    publish,
)
from goldilocks_ml.registry import register_predictor

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

WIDE_INTERVAL_FACTOR = 2.0


@dataclass(frozen=True, slots=True)
class KIndexQRFPredictor:
    """A verified quantile forest plus its CSLR and calibration contract."""

    estimator: Any
    record: Mapping[str, Any]
    model_id: str
    levels: tuple[float, ...]
    published: int

    def predict(self, structure: Structure) -> ModelPrediction:
        return self.predict_batch([structure])[0]

    def predict_batch(self, structures: Sequence[Structure]) -> list[ModelPrediction]:
        from goldilocks_ml.models.k_points.k_index.qrf import features

        if not structures:
            return []
        raw = prediction_matrix(
            self.estimator, features.feature_rows(structures), len(self.levels)
        )
        low, mid, high = (
            self.levels.index(level) for level in self.record["quantiles"]
        )
        decision = self.record.get("decision")
        calibration = self.record.get("calibration")
        correction = float(calibration["correction"]) if calibration else 0.0
        coverage = float(calibration["coverage"]) if calibration else None
        target_contract = self.record["target"]["contract"]
        contract = contract_for(target_contract)

        predictions: list[ModelPrediction] = []
        for index in range(len(structures)):
            lower, _, upper = calibrate_interval(
                float(raw[low, index]),
                float(raw[mid, index]),
                float(raw[high, index]),
                correction,
            )
            # The decision rule rounds to a whole rung and lifts the bands
            # where the model is weakest. Applying it here, through the same
            # function the trainer scored, is what makes the served number the
            # number the run bundle measured.
            value = publish(float(raw[self.published, index]), decision)
            prediction = ModelPrediction(
                parameter=contract.parameter,
                quantity=contract.quantity,
                value=value,
                target_contract=target_contract,
                model_id=self.model_id,
                confidence=coverage,
                details={
                    "interval": [lower, upper],
                    "coverage": coverage,
                    "calibrated": calibration is not None,
                    "units": None,
                    "index_base": 0,
                    "max_kpoints_per_axis": 50,
                    "decision": dict(decision) if decision else None,
                },
                warnings=self._warnings(upper - lower),
            )
            contract.check_value(float(prediction.value))
            predictions.append(prediction)
        return predictions

    def _warnings(self, width: float) -> tuple[str, ...]:
        expected = (self.record.get("calibration") or {}).get("mean_interval_width")
        if expected is None or float(expected) <= 0:
            return ()
        if width <= WIDE_INTERVAL_FACTOR * float(expected):
            return ()
        return (
            f"The {self.model_id} prediction interval spans {width:.2f} rungs, "
            f"more than {WIDE_INTERVAL_FACTOR:g} times the {float(expected):.2f} "
            "seen during calibration. Verify k-point convergence directly.",
        )


def load(
    record: Mapping[str, Any], directory: Path, artifacts: Mapping[str, Path]
) -> KIndexQRFPredictor:
    """Load an integrity-checked k-index forest without trusting its pickle."""
    from goldilocks_ml.models.k_points.k_index.qrf import features

    if artifacts:
        raise ValueError("the CSLR k-index model has no artifact dependencies")
    version = record.get("runtime", {}).get("version")
    if version != KINDEX_RUNTIME_VERSION:
        raise ValueError(
            f"this artifact declares {KINDEX_RUNTIME} runtime version {version!r}; "
            f"this build implements version {KINDEX_RUNTIME_VERSION}"
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

    # A k-index model must say which quantile it publishes. Serving the median
    # by default is what this runtime exists to stop: on this ladder it is the
    # choice that under-converges roughly a quarter of the time.
    levels = tuple(float(level) for level in record.get("levels", record["quantiles"]))
    decision = record.get("decision")
    if not decision:
        raise ValueError(
            "this artifact declares no decision rule; a k-index model must "
            "record which quantile it publishes"
        )
    if decision.get("rule") != "quantile":
        raise ValueError(
            f"this build serves the 'quantile' decision rule; the artifact "
            f"records {decision.get('rule')!r}"
        )
    if float(decision["level"]) not in levels:
        raise ValueError(
            f"the decision level {decision['level']} is not among the fitted "
            f"levels {list(levels)}"
        )

    calibration = record.get("calibration")
    if calibration is not None:
        if calibration.get("method") != CALIBRATION_METHOD:
            raise ValueError(
                f"this build applies {CALIBRATION_METHOD!r} calibration; the "
                f"artifact records {calibration.get('method')!r}"
            )
        if calibration.get("endpoint_adjustment") != ENDPOINT_ADJUSTMENT:
            raise ValueError(
                f"this build applies endpoint rule {ENDPOINT_ADJUSTMENT!r}; the "
                f"artifact records {calibration.get('endpoint_adjustment')!r}"
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
    return KIndexQRFPredictor(
        estimator=estimator,
        record=record,
        model_id=f"{KINDEX_RUNTIME}@{schema}",
        levels=levels,
        published=levels.index(float(decision["level"])),
    )


register_predictor(KINDEX_RUNTIME, load)
