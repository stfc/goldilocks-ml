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

if TYPE_CHECKING:  # a consumer can read these types without the model stack
    from pymatgen.core.structure import Structure

MODEL_RECORD_FILE = "model.json"

# What a published artifact is for. Most are models a consumer can call. Some
# are not: the released metallicity checkpoint is deposited because the
# k-distance feature contract embeds its pooled representation, and it carries
# no threshold, so it cannot answer a question on its own.
SERVABLE_MODEL = "model"
FEATURE_EXTRACTOR = "feature_extractor"
ROLES = frozenset({SERVABLE_MODEL, FEATURE_EXTRACTOR})

# The record shapes this build can read. A record from the future is refused
# rather than read with fields interpreted by an older meaning.
SUPPORTED_RECORD_SCHEMA_VERSIONS = frozenset({1})


# What a prediction speaks to. A DFT parameter is written into an input file.
# A material property is a fact about the structure that informs several inputs
# -- metallicity changes both mesh density and the smearing choice -- so it is
# predicted once and consumed in more than one place.
# A dataset judgement is neither. It does not advise a calculation: it says
# whether a structure is worth measuring, and its consumer is the campaign that
# builds the next dataset rather than an input file. Kept distinct so that a
# consumer looking for advice about a calculation cannot pick it up by mistake.
DFT_PARAMETER = "dft_parameter"
MATERIAL_PROPERTY = "material_property"
DATASET_SELECTION = "dataset_selection"
KINDS = frozenset({DFT_PARAMETER, MATERIAL_PROPERTY, DATASET_SELECTION})


@dataclass(frozen=True, slots=True)
class ConvergenceCriterion:
    """What "converged" means for a contract whose value comes off a ladder.

    ``metric`` names which physical quantity was tracked, never its size:
    ``threshold`` is the width of the trailing oscillation the sweep required
    before it accepted a rung, not a bound on the quantity itself. Two
    datasets can rank the same integer with a looser or tighter threshold on
    the same metric, or on a different metric entirely, and those are three
    different questions even though the answer is the same-shaped integer.
    """

    metric: str
    threshold: float
    unit: str


@dataclass(frozen=True, slots=True)
class IndexConvention:
    """What position in a per-entity prediction corresponds to.

    ``basis`` says what one entry means: ``"site"`` -- one value per site, in
    the structure's own input order, before any relabeling a consumer performs
    (Core may split "Fe" into "Fe1"/"Fe2" for an antiferromagnetic cell; that
    expansion is Core's problem, not this contract's); ``"species"`` -- one
    value per distinct chemical symbol, keyed by symbol as pymatgen spells it.

    ``order`` is only meaningful for ``basis == "site"`` and names the
    ordering rule. Only ``"input_order"`` is defined: the order Core's own
    Structure object already has when handed to ``predict()``. A model that
    reorders sites internally must undo that before returning, not document it
    here.
    """

    basis: str
    order: str | None = None

    def check_value(self, value: Any, *, n_entities: int | None = None) -> None:
        """Reject a per-entity prediction with the wrong shape."""
        if self.basis == "site":
            if not isinstance(value, tuple):
                raise ValueError(
                    "a per-site prediction must be a tuple, one entry per "
                    f"site in input order; got {type(value).__name__}"
                )
            if n_entities is not None and len(value) != n_entities:
                raise ValueError(
                    f"a per-site prediction must have {n_entities} entries "
                    f"(one per site); got {len(value)}"
                )
        elif self.basis == "species":
            if not isinstance(value, Mapping):
                raise ValueError(
                    "a per-species prediction must be a mapping keyed by "
                    f"chemical symbol; got {type(value).__name__}"
                )
        else:
            raise ValueError(f"unknown index_convention.basis {self.basis!r}")


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """What a published target contract means to a consumer.

    ``parameter`` names the advice a prediction speaks to, matching the field
    Core carries it in. ``quantity`` says what the number is, which is what
    decides how Core converts it. ``units`` and the domain are checked here so
    that a mismatch fails rather than producing a plausible, wrong setting.

    ``min_k_distance`` and ``convergence`` are structured facts about a k-mesh
    ladder rather than semantics encoded into the contract name: a rung means
    nothing without the resolution floor it was enumerated to, and nothing
    about what "converged" meant when the label was assigned.

    ``labels`` and ``index_convention`` are the same idea applied to a
    multi-class or per-entity quantity: the legal label set, or the position
    convention, is part of what the contract means and belongs here rather
    than in a name a consumer has to parse.
    """

    parameter: str
    quantity: str
    kind: str = "dft_parameter"
    units: str | None = None
    positive: bool = False
    non_negative: bool = False
    boolean: bool = False
    min_k_distance: float | None = None
    convergence: ConvergenceCriterion | None = None
    labels: tuple[str, ...] | None = None
    index_convention: IndexConvention | None = None

    def check_units(self, units: str | None) -> None:
        """Reject a record whose units are not the ones this contract means."""
        if self.units is not None and units != self.units:
            raise ValueError(
                f"target contract expects units {self.units!r}, but the record "
                f"declares {units!r}"
            )

    def check_value(self, value: Any, *, n_entities: int | None = None) -> None:
        """Reject a prediction outside the domain the quantity admits.

        ``n_entities`` is only meaningful for a per-site ``index_convention``
        and is optional: a caller that does not know the site count yet still
        gets the type check.
        """
        if self.boolean:
            if not isinstance(value, bool):
                raise ValueError(
                    f"{self.quantity} must be a boolean; the model predicted {value!r}"
                )
            return
        if self.labels is not None:
            if not isinstance(value, str) or value not in self.labels:
                allowed = ", ".join(self.labels)
                raise ValueError(
                    f"{self.quantity} must be one of {{{allowed}}}; the model "
                    f"predicted {value!r}"
                )
            return
        if self.index_convention is not None:
            self.index_convention.check_value(value, n_entities=n_entities)
            return
        if self.positive and not float(value) > 0:
            raise ValueError(
                f"{self.quantity} must be positive; the model predicted {value}"
            )
        if self.non_negative and float(value) < 0:
            raise ValueError(
                f"{self.quantity} must be non-negative; the model predicted {value}"
            )


# The target contracts this build understands. The contract string, not the
# bare quantity, is the key: two models can both predict a "k-distance" and
# differ by a factor of 2 pi, and only the contract distinguishes them.
CONTRACTS: Mapping[str, ContractSpec] = {
    "goldilocks.k_distance.mesh_lower_bound.2pi.v1": ContractSpec(
        parameter="k_points",
        quantity="k_distance",
        kind=DFT_PARAMETER,
        units="1/angstrom",
        positive=True,
    ),
    "goldilocks.k_index.ladder_0based.max50.v1": ContractSpec(
        parameter="k_points",
        quantity="k_index",
        kind=DFT_PARAMETER,
        units=None,
        non_negative=True,
    ),
    # The same ladder, counted from one, so that rung 1 is the Gamma-only mesh
    # rather than rung 0. A consumer must not treat the two as interchangeable:
    # the integers differ by one over most of the ladder, and by more wherever
    # the 1-based enumeration also dropped a repeated mesh.
    #
    # v2, not a v1 with the name corrected: the superseded v1 name declared
    # "max50", an enumeration cap that never existed for this ladder -- it was
    # enumerated to a resolution floor instead. A contract that lies about its
    # own bound is worse than one that is merely incomplete, so this is a
    # replacement, not a patch.
    "goldilocks.k_index.ladder_1based.v2": ContractSpec(
        parameter="k_points",
        quantity="k_index",
        kind=DFT_PARAMETER,
        units=None,
        positive=True,
        min_k_distance=0.03,
        convergence=ConvergenceCriterion(
            metric="total_energy", threshold=1.0, unit="meV/atom"
        ),
    ),
    "goldilocks.is_metal.dft_band_gap_zero.v1": ContractSpec(
        parameter="metallicity",
        quantity="is_metal",
        kind=MATERIAL_PROPERTY,
        boolean=True,
    ),
    # Whether a structure needs a mesh at rung 11 or above on the same ladder
    # the k_index contract counts. The rung is part of the contract string
    # because it is part of what the answer means: a model screening at a
    # different rung answers a different question and must not be swapped in.
    "goldilocks.k_index_dense.ladder_0based.ge11.v1": ContractSpec(
        parameter="dataset_candidate",
        quantity="needs_dense_mesh",
        kind=DATASET_SELECTION,
        boolean=True,
    ),
    # The same question asked on the 1-based ladder, where the rung that was
    # 11 is now 12.
    "goldilocks.k_index_dense.ladder_1based.ge12.v1": ContractSpec(
        parameter="dataset_candidate",
        quantity="needs_dense_mesh",
        kind=DATASET_SELECTION,
        boolean=True,
    ),
    # Whether DFT gives this structure a spin-polarised ground state. The name
    # states the label rule, not a decision threshold (that lives in the
    # record's own `decision` block, the same split `is_metal` uses): a
    # structure whose maximum on-site moment lands strictly between 0.05 and
    # 0.5 Bohr magnetons was excluded from the training and test data this
    # contract's models are fitted and scored against, so a prediction near
    # that boundary is extrapolation, not interpolation. That caveat belongs
    # in a model card, not in this name -- see the "max50" lesson in the
    # k_index history above for why a name should not try to carry it.
    "goldilocks.is_magnetic.dft_max_site_magmom_0p5ub.v1": ContractSpec(
        parameter="magnetism",
        quantity="is_magnetic",
        kind=MATERIAL_PROPERTY,
        boolean=True,
    ),
    # `ordering` (NM/FM/AFM/FiM, a `labels`-shaped quantity) and
    # `magnetic_moments` (one signed z-component per site, an
    # `index_convention`-shaped quantity) are not added here: no model backs
    # either yet, and a contract with no loadable release would be a promise
    # this build cannot keep. `IndexConvention` and `ContractSpec.labels`
    # above exist so that when one is trained, declaring it is a dict entry,
    # not new mechanism.
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
    record: Mapping[str, Any],
    artifact_directory: Path | None = None,
    overrides: Mapping[str, Path] | None = None,
    record_directory: Path | None = None,
) -> dict[str, Path]:
    """Resolve and verify the supporting artifacts a record declares.

    A model's featurisation may depend on other released artifacts -- QRF95
    embeds a metallicity checkpoint. The record pins each by record id and
    digest, so a consumer never needs to know that any of them exist. An
    override supplies a path but does not excuse it from verification.

    A deposit may also ship a copy of a small dependency beside its own
    ``model.json``, which makes the record self-contained: one download runs
    the model. A file found there is preferred over the pinned record's copy
    and is verified against the same digest, so the two cannot drift.
    """
    declared = record.get("requires_artifacts", ())
    if not declared:
        return dict(overrides or {})

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
    supplied = dict(overrides or {})
    if record_directory is not None:
        for dependency in dependencies:
            beside = Path(record_directory) / dependency.file
            if dependency.name not in supplied and beside.is_file():
                supplied[dependency.name] = beside
    return resolve(dependencies, default_directory(artifact_directory), supplied)


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

    version = record.get("record_schema_version")
    if version not in SUPPORTED_RECORD_SCHEMA_VERSIONS:
        supported = ", ".join(
            str(item) for item in sorted(SUPPORTED_RECORD_SCHEMA_VERSIONS)
        )
        raise ValueError(
            f"model record schema version {version!r} is not supported; this "
            f"build reads: {supported}"
        )

    role = record.get("role", SERVABLE_MODEL)
    if role not in ROLES:
        known = ", ".join(sorted(ROLES))
        raise ValueError(f"unknown record role {role!r}; this build knows: {known}")
    if role == FEATURE_EXTRACTOR:
        supplies = record.get("supplies", {})
        consumer = supplies.get("consumed_by")
        raise ValueError(
            "this artifact is published as a feature extractor, not as a model "
            "that answers a question"
            + (
                f"; it supplies input to the {consumer!r} feature contract"
                if consumer
                else ""
            )
        )

    contract = contract_for(record["target"]["contract"])
    contract.check_units(record["target"].get("units"))

    resolved = required_artifacts(
        record, artifact_directory, artifacts, record_directory=directory
    )
    runtime = record.get("runtime", {})
    predictor = get_predictor(runtime.get("id"))
    model = predictor(record, directory, resolved)
    if model_id is not None:
        return replace_model_id(model, model_id)
    return model


def replace_model_id(model: StructureModel, model_id: str) -> StructureModel:
    """Return the model relabelled with the identity a registry gave it."""
    from dataclasses import replace

    return replace(model, model_id=model_id)  # type: ignore[type-var]
