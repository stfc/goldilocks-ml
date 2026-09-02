"""Serve the metallicity classifier for single structures.

The serving counterpart of this package's :mod:`trainer`. A classifier returns
a score; the label it becomes depends on a threshold chosen on the validation
split, so the record carries that threshold and this module applies it. A
consumer receives the decision, not the number behind it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.inference import ModelPrediction, contract_for
from goldilocks_ml.models.metallicity.is_metal.cgcnn.trainer import (
    RUNTIME,
    RUNTIME_VERSION,
)
from goldilocks_ml.registry import register_predictor

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

ATOM_INIT = "atom_init"


@dataclass(frozen=True, slots=True)
class MetallicityPredictor:
    """A fitted metallicity classifier, loaded for prediction from structures."""

    model: Any
    record: Mapping[str, Any]
    atom_init: Path
    model_id: str

    def predict(self, structure: Structure) -> ModelPrediction:
        """Return whether one structure is metallic."""
        return self.predict_batch([structure])[0]

    def predict_batch(self, structures: Sequence[Structure]) -> list[ModelPrediction]:
        """Return one decision per structure, graphing them together."""
        from goldilocks_ml.models.k_points.k_distance.qrf.embedding import build_graph
        from goldilocks_ml.models.metallicity.is_metal.cgcnn.trainer import class_scores

        if not structures:
            return []

        graphs = [build_graph(structure, self.atom_init) for structure in structures]
        scores = class_scores(self.model, graphs)

        decision = self.record["decision"]
        threshold = float(decision["threshold"])
        target_contract = self.record["target"]["contract"]
        contract = contract_for(target_contract)
        classes = self.record["classes"]

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
                    # `confidence` carries a guarantee, and this score is not
                    # one. It measures well on held-out data but nothing proves
                    # it, unlike a conformal coverage level, so it travels as
                    # provenance instead of as a claim.
                    confidence=None,
                    details={
                        "score": score,
                        "threshold": threshold,
                        "label": classes["positive"]
                        if is_positive
                        else classes["negative"],
                        "score_is": "uncalibrated positive-class softmax",
                        "threshold_selected_on": decision.get("selected_on"),
                        "threshold_metric": decision.get("metric"),
                        "min_recall": decision.get("min_recall"),
                    },
                )
            )
        for prediction in predictions:
            contract.check_value(prediction.value)
        return predictions


def load(
    record: Mapping[str, Any], directory: Path, artifacts: Mapping[str, Path]
) -> MetallicityPredictor:
    """Build a predictor from a stored record, checking what it declares."""
    import torch

    from goldilocks_ml.models.k_points.k_distance.qrf.embedding import CGCNN
    from goldilocks_ml.models.metallicity.is_metal.cgcnn import graphs as crystal_graphs

    version = record.get("runtime", {}).get("version")
    if version != RUNTIME_VERSION:
        raise ValueError(
            f"this artifact declares {RUNTIME} runtime version {version!r}; "
            f"this build implements version {RUNTIME_VERSION}"
        )

    schema = record["feature_schema"]
    if schema != crystal_graphs.SCHEMA:
        raise ValueError(
            f"this artifact was built against feature contract {schema!r}, but "
            f"the installed goldilocks-ml provides {crystal_graphs.SCHEMA!r}; "
            "upgrade goldilocks-ml to load it"
        )

    decision = record.get("decision") or {}
    if "threshold" not in decision:
        raise ValueError(
            "this artifact records no decision threshold, so its score cannot "
            "be turned into a label; it predates the threshold being written "
            "into the record and must be retrained or repaired"
        )

    if ATOM_INIT not in artifacts:
        raise ValueError(f"the {schema} feature contract needs artifact: {ATOM_INIT}")
    atom_init = Path(artifacts[ATOM_INIT])
    # The embedding table is part of the feature definition: a different table
    # produces different graphs and therefore different answers, silently.
    pinned_table = record.get("atom_init_sha256")
    if pinned_table:
        digest = sha256_file(atom_init)
        if digest != pinned_table:
            raise ValueError(
                f"{atom_init.name} has SHA-256 {digest}; its record pins {pinned_table}"
            )

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
    model = CGCNN(**architecture)
    model.load_state_dict(stored["state_dict"])
    model.eval()

    return MetallicityPredictor(
        model=model,
        record=record,
        atom_init=atom_init,
        model_id=f"{RUNTIME}@{schema}",
    )


register_predictor(RUNTIME, load)
