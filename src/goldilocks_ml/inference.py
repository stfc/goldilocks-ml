"""Predict from one structure, for consumers that only want the answer.

This is the seam between this package and Goldilocks Core. Core hands over a
pymatgen ``Structure`` and receives a single value together with the physical
quantity it is expressed in; featurisation, artifact loading, and calibration
all stay on this side of it. Core never sees a feature vector, and adding a
model that predicts a different quantity does not change Core.

A prediction carries one value, not an interval. Core can only emit one mesh,
and the original study reported its regression metrics against the median, so
choosing which point to publish is a modelling decision and belongs here. What
uncertainty a model does have travels in ``details`` and ``warnings``, which
a consumer records but never branches on.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # a consumer can read these types without the qrf95 stack
    from pymatgen.core.structure import Structure

MODEL_RECORD_FILE = "model.json"

# The quantity each published target contract is expressed in. A consumer needs
# this to convert a prediction into a mesh, and the contract -- not the bare
# quantity -- is what pins the convention: two k-distances can differ by 2pi.
QUANTITY_BY_TARGET_CONTRACT: Mapping[str, str] = {
    "goldilocks.k_distance.mesh_lower_bound.2pi.v1": "k_distance",
}

# A prediction interval this many times the mean width seen during calibration
# is flagged. This is a heuristic for structures unlike the training set, not a
# statistical statement about the individual prediction.
WIDE_INTERVAL_FACTOR = 2.0


@dataclass(frozen=True, slots=True)
class KMeshPrediction:
    """One model's recommendation for a k-point mesh.

    ``value`` is expressed in ``quantity``; a consumer converts it. ``details``
    and ``warnings`` are opaque provenance to be recorded verbatim.
    """

    quantity: str
    value: float
    target_contract: str
    model_id: str
    confidence: float | None = None
    details: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = field(default=())


@runtime_checkable
class KMeshModel(Protocol):
    """Anything that turns a structure into a k-mesh recommendation."""

    def predict(self, structure: Structure) -> KMeshPrediction:
        """Return the recommendation for one structure."""
        ...


@dataclass(frozen=True, slots=True)
class QRF95Inference:
    """The published QRF95 k-distance model, loaded for single structures."""

    estimator: Any
    record: Mapping[str, Any]
    metallicity_checkpoint: Path
    metallicity_atom_init: Path
    model_id: str

    def predict(self, structure: Structure) -> KMeshPrediction:
        """Return the calibrated median k-distance for one structure."""
        return self.predict_batch([structure])[0]

    def predict_batch(self, structures: Sequence[Structure]) -> list[KMeshPrediction]:
        """Return one prediction per structure, featurising them together."""
        from goldilocks_ml.models.kmesh.qrf95 import features as qrf_features
        from goldilocks_ml.models.kmesh.qrf95.trainer import (
            calibrate_interval,
            prediction_matrix,
        )

        if not structures:
            return []

        soap = qrf_features.resolve_soap(self.record.get("feature_parameters", {}))
        rows = qrf_features.feature_rows(
            list(structures),
            soap=soap,
            metallicity_checkpoint=self.metallicity_checkpoint,
            metallicity_atom_init=self.metallicity_atom_init,
        )
        # prediction_matrix normalises the estimator's shape, which differs
        # for a single row, and rejects unordered or non-finite output.
        raw = prediction_matrix(self.estimator, rows)
        calibration = self.record["calibration"]
        correction = float(calibration["correction"])
        target_contract = self.record["target"]["contract"]
        quantity = QUANTITY_BY_TARGET_CONTRACT[target_contract]

        predictions = []
        for index in range(len(structures)):
            lower, median, upper = calibrate_interval(
                float(raw[0, index]),
                float(raw[1, index]),
                float(raw[2, index]),
                correction,
            )
            predictions.append(
                KMeshPrediction(
                    quantity=quantity,
                    value=median,
                    target_contract=target_contract,
                    model_id=self.model_id,
                    confidence=float(calibration["coverage"]),
                    details={
                        "interval": [lower, upper],
                        "coverage": float(calibration["coverage"]),
                        "units": self.record["target"]["units"],
                    },
                    warnings=self._warnings(upper - lower),
                )
            )
        return predictions

    def _warnings(self, width: float) -> tuple[str, ...]:
        """Flag an interval far wider than calibration led us to expect."""
        expected = self.record["calibration"].get("mean_interval_width")
        if expected is None or expected <= 0:
            return ()
        if width <= WIDE_INTERVAL_FACTOR * float(expected):
            return ()
        return (
            f"The {self.model_id} prediction interval is {width:.4f}, more than "
            f"{WIDE_INTERVAL_FACTOR:g} times the {float(expected):.4f} seen "
            "during calibration. This structure may be unlike the training set; "
            "verify k-point convergence directly.",
        )


def load_kmesh_model(
    directory: Path,
    *,
    metallicity_checkpoint: Path,
    metallicity_atom_init: Path,
    model_id: str | None = None,
) -> KMeshModel:
    """Load a k-mesh model from a directory a training run or deposit produced.

    The directory holds the estimator named in the model record alongside
    ``model.json``. Every contract the record declares is checked here, so a
    consumer that is too old for an artifact is told which contract it is
    missing rather than silently predicting from the wrong feature vector.
    """
    record = json.loads((directory / MODEL_RECORD_FILE).read_text(encoding="utf-8"))

    target_contract = record["target"]["contract"]
    if target_contract not in QUANTITY_BY_TARGET_CONTRACT:
        known = ", ".join(sorted(QUANTITY_BY_TARGET_CONTRACT)) or "none"
        raise ValueError(
            f"no k-mesh quantity is defined for target contract "
            f"{target_contract!r}; this build understands: {known}"
        )

    from goldilocks_ml.models.kmesh.qrf95 import features as qrf_features

    schema = record["feature_schema"]
    if schema != qrf_features.SCHEMA:
        raise ValueError(
            f"this artifact was built against feature contract {schema!r}, but "
            f"the installed goldilocks-ml provides {qrf_features.SCHEMA!r}; "
            "upgrade goldilocks-ml to load it"
        )

    estimator_file = record["artifacts"]["estimator"]
    with (directory / estimator_file).open("rb") as handle:
        estimator = pickle.load(handle)

    expected_width = len(record["feature_columns"])
    actual_width = getattr(estimator, "n_features_in_", expected_width)
    if actual_width != expected_width:
        raise ValueError(
            f"{estimator_file} takes {actual_width} features but its record "
            f"declares {expected_width}; the artifact and its record disagree"
        )

    return QRF95Inference(
        estimator=estimator,
        record=record,
        metallicity_checkpoint=Path(metallicity_checkpoint),
        metallicity_atom_init=Path(metallicity_atom_init),
        model_id=model_id or f"{record['trainer']}@{schema}",
    )
