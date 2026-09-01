"""Serve the QRF95 k-distance model for single structures.

The serving counterpart of this package's :mod:`trainer`. It
reads back what that trainer wrote and applies the same calibration, so the two
cannot disagree about what a stored correction means.
"""

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
    RUNTIME,
    RUNTIME_VERSION,
)
from goldilocks_ml.registry import register_predictor

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

# A prediction interval this many times the mean width measured during
# calibration is flagged. This is a heuristic for structures unlike the
# training set, not a statistical claim about the individual prediction.
WIDE_INTERVAL_FACTOR = 2.0


@dataclass(frozen=True, slots=True)
class QRF95Predictor:
    """The published QRF95 model, loaded for prediction from structures."""

    estimator: Any
    record: Mapping[str, Any]
    artifacts: Mapping[str, Path]
    model_id: str

    def predict(self, structure: Structure) -> ModelPrediction:
        """Return the calibrated median k-distance for one structure."""
        return self.predict_batch([structure])[0]

    def predict_batch(self, structures: Sequence[Structure]) -> list[ModelPrediction]:
        """Return one prediction per structure, featurising them together."""
        from goldilocks_ml.models.k_points.k_distance.qrf import (
            features as qrf_features,
        )
        from goldilocks_ml.models.k_points.k_distance.qrf.trainer import (
            calibrate_interval,
            prediction_matrix,
        )

        if not structures:
            return []

        rows = qrf_features.feature_rows(
            list(structures),
            soap=qrf_features.resolve_soap(self.record.get("feature_parameters", {})),
            metallicity_checkpoint=self.artifacts["metallicity_checkpoint"],
            metallicity_atom_init=self.artifacts["metallicity_atom_init"],
        )
        # prediction_matrix normalises the estimator's output shape, which
        # differs for a single row, and rejects unordered or non-finite values.
        raw = prediction_matrix(self.estimator, rows)

        calibration = self.record["calibration"]
        correction = float(calibration["correction"])
        coverage = float(calibration["coverage"])
        target_contract = self.record["target"]["contract"]
        contract = contract_for(target_contract)

        predictions = []
        for index in range(len(structures)):
            lower, median, upper = calibrate_interval(
                float(raw[0, index]),
                float(raw[1, index]),
                float(raw[2, index]),
                correction,
            )
            predictions.append(
                ModelPrediction(
                    parameter=contract.parameter,
                    quantity=contract.quantity,
                    value=median,
                    target_contract=target_contract,
                    model_id=self.model_id,
                    confidence=coverage,
                    details={
                        "interval": [lower, upper],
                        "coverage": coverage,
                        "units": self.record["target"]["units"],
                    },
                    warnings=self._warnings(upper - lower),
                )
            )
        for prediction in predictions:
            contract.check_value(float(prediction.value))
        return predictions

    def _warnings(self, width: float) -> tuple[str, ...]:
        """Flag an interval far wider than calibration led us to expect."""
        expected = self.record["calibration"].get("mean_interval_width")
        if expected is None or float(expected) <= 0:
            return ()
        if width <= WIDE_INTERVAL_FACTOR * float(expected):
            return ()
        return (
            f"The {self.model_id} prediction interval is {width:.4f}, more than "
            f"{WIDE_INTERVAL_FACTOR:g} times the {float(expected):.4f} seen "
            "during calibration. This structure may be unlike the training set; "
            "verify k-point convergence directly.",
        )


def load(
    record: Mapping[str, Any], directory: Path, artifacts: Mapping[str, Path]
) -> QRF95Predictor:
    """Build a predictor from a stored record, checking what it declares."""
    from goldilocks_ml.models.k_points.k_distance.qrf import features as qrf_features

    version = record.get("runtime", {}).get("version")
    if version != RUNTIME_VERSION:
        raise ValueError(
            f"this artifact declares {RUNTIME} runtime version {version!r}; "
            f"this build implements version {RUNTIME_VERSION}"
        )

    schema = record["feature_schema"]
    if schema != qrf_features.SCHEMA:
        raise ValueError(
            f"this artifact was built against feature contract {schema!r}, but "
            f"the installed goldilocks-ml provides {qrf_features.SCHEMA!r}; "
            "upgrade goldilocks-ml to load it"
        )

    # A matching width over reordered or renamed columns would predict from a
    # scrambled vector, so compare the contract itself, not its size.
    recorded = tuple(record["feature_columns"])
    if recorded != qrf_features.column_names():
        raise ValueError(
            f"the {schema!r} columns this build produces differ from the ones "
            "the artifact was fitted on; upgrade goldilocks-ml to load it"
        )

    calibration = record["calibration"]
    if calibration.get("method") != CALIBRATION_METHOD:
        raise ValueError(
            f"this build applies {CALIBRATION_METHOD!r} calibration; the "
            f"artifact records {calibration.get('method')!r}"
        )
    if calibration.get("endpoint_adjustment") != ENDPOINT_ADJUSTMENT:
        raise ValueError(
            f"this build applies the {ENDPOINT_ADJUSTMENT!r} endpoint rule; "
            f"the artifact records {calibration.get('endpoint_adjustment')!r}"
        )

    missing = [
        name
        for name in ("metallicity_checkpoint", "metallicity_atom_init")
        if name not in artifacts
    ]
    if missing:
        raise ValueError(
            "the QRF95 feature contract needs artifact(s): " + ", ".join(missing)
        )

    estimator_file = record["artifacts"]["estimator"]
    estimator_path = Path(directory) / estimator_file
    # Unpickling executes code. The record pins what may be unpickled, so a
    # substituted file is refused before it is opened as anything but bytes.
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
            f"declares {expected_width}; the artifact and its record disagree"
        )

    return QRF95Predictor(
        estimator=estimator,
        record=record,
        artifacts=artifacts,
        model_id=f"{RUNTIME}@{schema}",
    )


register_predictor(RUNTIME, load)
