"""Predict a DFT input parameter from one structure.

This is the seam between this package and Goldilocks Core. Core hands over a
pymatgen ``Structure`` and receives one value together with the parameter it
advises and the quantity it is expressed in. Featurisation, artifact
resolution, and calibration all stay on this side of it.

There is one prediction type, not one per parameter. Core already names the
parameters it advises -- k-points, smearing, magnetism, spin-orbit,
pseudopotentials, convergence, exchange-correlation -- so a model says which
one it speaks to and what its number means, and Core routes it. Adding a model
for a parameter no model covered before adds a row to a table here and a
resolver on Core's side; it does not change either side's plumbing.

A prediction carries one value, never an interval. Uncertainty a model does
have travels in ``details`` and ``warnings``, which a consumer records but
never branches on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from goldilocks_ml.registry import get_predictor

if TYPE_CHECKING:  # a consumer can read these types without the qrf95 stack
    from pymatgen.core.structure import Structure

MODEL_RECORD_FILE = "model.json"


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """What a published target contract means to a consumer.

    ``parameter`` names the advice a prediction speaks to, matching the field
    Core carries it in. ``quantity`` says what the number is, which is what
    decides how Core converts it.
    """

    parameter: str
    quantity: str


# The target contracts this build understands. The contract string, not the
# bare quantity, is the key: two models can both predict a "k-distance" and
# differ by a factor of 2 pi, and only the contract distinguishes them.
CONTRACTS: Mapping[str, ContractSpec] = {
    "goldilocks.k_distance.mesh_lower_bound.2pi.v1": ContractSpec(
        parameter="k_points", quantity="k_distance"
    ),
}


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """One model's recommendation for one DFT input parameter.

    ``value`` is expressed in ``quantity``; a consumer converts it. ``details``
    and ``warnings`` are opaque provenance, to be recorded verbatim.
    """

    parameter: str
    quantity: str
    value: float | int | bool | str | tuple[Any, ...]
    target_contract: str
    model_id: str
    confidence: float | None = None
    details: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = field(default=())


@runtime_checkable
class StructureModel(Protocol):
    """Anything that turns a structure into advice about one DFT parameter."""

    def predict(self, structure: Structure) -> ModelPrediction:
        """Return the recommendation for one structure."""
        ...

    def predict_batch(self, structures: Sequence[Structure]) -> list[ModelPrediction]:
        """Return one recommendation per structure."""
        ...


def contract_for(target_contract: str) -> ContractSpec:
    """Return what a target contract means, or say which ones are known."""
    try:
        return CONTRACTS[target_contract]
    except KeyError:
        known = ", ".join(sorted(CONTRACTS)) or "none"
        raise ValueError(
            f"no DFT parameter is defined for target contract "
            f"{target_contract!r}; this build understands: {known}"
        ) from None


def read_record(directory: Path) -> dict[str, Any]:
    """Return the self-describing record a training run or deposit ships."""
    return json.loads((Path(directory) / MODEL_RECORD_FILE).read_text(encoding="utf-8"))


def required_artifacts(
    record: Mapping[str, Any], artifact_directory: Path | None = None
) -> dict[str, Path]:
    """Resolve and verify the supporting artifacts a record declares.

    A model's featurisation may depend on other released artifacts -- QRF95
    embeds a metallicity checkpoint. The record pins each by record id and
    digest, so a consumer never needs to know that any of them exist.
    """
    declared = record.get("requires_artifacts", ())
    if not declared:
        return {}

    from goldilocks_ml.artifacts import artifact_directory as default_directory
    from goldilocks_ml.artifacts import resolve
    from goldilocks_ml.protocol import ArtifactDependency

    dependencies = [
        ArtifactDependency(
            name=item["name"],
            record_id=item["record_id"],
            file=item["file"],
            sha256=item["sha256"],
        )
        for item in declared
    ]
    return resolve(dependencies, default_directory(artifact_directory))


def load_model(
    directory: Path,
    *,
    artifacts: Mapping[str, Path] | None = None,
    artifact_directory: Path | None = None,
    model_id: str | None = None,
) -> StructureModel:
    """Load a model from a directory a training run or a deposit produced.

    The directory holds the estimator named in the record alongside
    ``model.json``. Everything the record declares is checked here, so a
    consumer too old for an artifact is told what it is missing rather than
    predicting from the wrong feature vector.
    """
    directory = Path(directory)
    record = read_record(directory)
    contract_for(record["target"]["contract"])
    resolved = (
        dict(artifacts)
        if artifacts is not None
        else required_artifacts(record, artifact_directory)
    )
    predictor = get_predictor(record["trainer"])
    model = predictor(record, directory, resolved)
    if model_id is not None:
        return replace_model_id(model, model_id)
    return model


def replace_model_id(model: StructureModel, model_id: str) -> StructureModel:
    """Return the model relabelled with the identity a registry gave it."""
    from dataclasses import replace

    return replace(model, model_id=model_id)  # type: ignore[type-var]
