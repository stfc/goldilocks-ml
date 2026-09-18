"""Serve the ``is_magnetic`` classifier for single structures.

The serving counterpart of this package's :mod:`trainer`. A classifier
returns a score; the label it becomes depends on a threshold chosen on the
validation split, so the record carries that threshold and this module
applies it -- a consumer receives the decision, not the number behind it.
Mirrors :mod:`goldilocks_ml.models.metallicity.is_metal.cgcnn.predictor`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.inference import ModelPrediction, contract_for
from goldilocks_ml.models.magnetism._mace_backbone import embed_structures
from goldilocks_ml.models.magnetism.is_magnetic.mace_mlp.features import (
    MACE_BACKBONE,
    SCHEMA,
    column_names,
)
from goldilocks_ml.models.magnetism.is_magnetic.mace_mlp.trainer import (
    RUNTIME,
    RUNTIME_VERSION,
    MagneticMLP,
    class_scores,
)
from goldilocks_ml.registry import register_predictor

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@dataclass(frozen=True, slots=True)
class MagneticPredictor:
    """A fitted ``is_magnetic`` classifier, loaded for prediction from structures."""

    model: MagneticMLP
    scaler_mean: torch.Tensor
    scaler_scale: torch.Tensor
    record: Mapping[str, Any]
    backbone: Path
    model_id: str

    def predict(self, structure: Structure) -> ModelPrediction:
        """Return whether one structure is magnetic."""
        return self.predict_batch([structure])[0]

    def predict_batch(self, structures: Sequence[Structure]) -> list[ModelPrediction]:
        """Return one decision per structure, embedding them together."""
        if not structures:
            return []

        rows = torch.tensor(
            embed_structures(list(structures), checkpoint=self.backbone),
            dtype=torch.float32,
        )
        scores = class_scores(self.model, rows, self.scaler_mean, self.scaler_scale)

        decision = self.record["decision"]
        threshold = float(decision["threshold"])
        target_contract = self.record["target"]["contract"]
        contract = contract_for(target_contract)
        classes = self.record["classes"]
        validation_status = self.record.get("validation_status") or {}

        warnings: tuple[str, ...] = ()
        if validation_status.get("external_validation") == "pending":
            note = validation_status.get(
                "external_validation_note", "see the model card"
            )
            warnings = (
                f"{self.model_id} has not been externally validated ({note})",
            )

        predictions = []
        for score in scores:
            is_positive = score >= threshold
            predictions.append(
                ModelPrediction(
                    parameter=contract.parameter,
                    quantity=contract.quantity,
                    value=is_positive,
                    target_contract=target_contract,
                    model_id=self.model_id,
                    # See cgcnn/predictor.py: a held-out score, not a coverage
                    # guarantee, so it travels as provenance, not a claim.
                    confidence=None,
                    details={
                        "score": score,
                        "threshold": threshold,
                        "label": classes["positive"]
                        if is_positive
                        else classes["negative"],
                        "score_is": "uncalibrated positive-class sigmoid",
                        "threshold_selected_on": decision.get("selected_on"),
                    },
                    warnings=warnings,
                )
            )
        for prediction in predictions:
            contract.check_value(prediction.value)
        return predictions


def load(
    record: Mapping[str, Any], directory: Path, artifacts: Mapping[str, Path]
) -> MagneticPredictor:
    """Build a predictor from a stored record, checking what it declares."""
    version = record.get("runtime", {}).get("version")
    if version != RUNTIME_VERSION:
        raise ValueError(
            f"this artifact declares {RUNTIME} runtime version {version!r}; "
            f"this build implements version {RUNTIME_VERSION}"
        )

    schema = record["feature_schema"]
    if schema != SCHEMA:
        raise ValueError(
            f"this artifact was built against feature contract {schema!r}, but "
            f"the installed goldilocks-ml provides {SCHEMA!r}; upgrade "
            "goldilocks-ml to load it"
        )

    # A matching width over reordered or renamed columns would predict from a
    # scrambled vector, so compare the contract itself, not its size.
    recorded = tuple(record["feature_columns"])
    if recorded != column_names():
        raise ValueError(
            f"the {schema!r} columns this build produces differ from the ones "
            "the artifact was fitted on; upgrade goldilocks-ml to load it"
        )

    decision = record.get("decision") or {}
    if "threshold" not in decision:
        raise ValueError(
            "this artifact records no decision threshold, so its score cannot "
            "be turned into a label; it predates the threshold being written "
            "into the record and must be retrained or repaired"
        )

    if MACE_BACKBONE not in artifacts:
        raise ValueError(
            f"the {schema} feature contract needs artifact: {MACE_BACKBONE}"
        )
    backbone = Path(artifacts[MACE_BACKBONE])

    weights_file = record["artifacts"]["estimator"]
    weights_path = Path(directory) / weights_file
    pinned = record["artifacts"].get("estimator_sha256")
    if not pinned:
        raise ValueError(
            f"the record does not pin a SHA-256 for {weights_file}; refusing to "
            "load unverified weights"
        )
    digest = sha256_file(weights_path)
    if digest != pinned:
        raise ValueError(
            f"{weights_file} has SHA-256 {digest}; its record pins {pinned}"
        )

    stored = torch.load(weights_path, map_location="cpu", weights_only=True)
    architecture = dict(record["architecture"])
    if dict(stored["architecture"]) != architecture:
        raise ValueError(
            f"{weights_file} was built with a different architecture than its "
            "record declares; the artifact and its record disagree"
        )
    model = MagneticMLP(**architecture)
    model.load_state_dict(stored["state_dict"])
    model.eval()

    return MagneticPredictor(
        model=model,
        scaler_mean=stored["scaler_mean"],
        scaler_scale=stored["scaler_scale"],
        record=record,
        backbone=backbone,
        model_id=f"{RUNTIME}@{schema}",
    )


register_predictor(RUNTIME, load)
