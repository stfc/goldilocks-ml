"""Tests for the single-structure inference seam."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_ml import inference
from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.inference import (
    ModelPrediction,
    StructureModel,
    contract_for,
    load_model,
)
from goldilocks_ml.models.k_points.k_distance.qrf import features as qrf_features
from goldilocks_ml.models.k_points.k_distance.qrf.trainer import (
    CALIBRATION_METHOD,
    ENDPOINT_ADJUSTMENT,
    RUNTIME,
    RUNTIME_VERSION,
)

K_DISTANCE_CONTRACT = "goldilocks.k_distance.mesh_lower_bound.2pi.v1"


class StubEstimator:
    """A quantile estimator with a fixed answer and a declared input width."""

    def __init__(self, triple: tuple[float, float, float], width: int = 483) -> None:
        self.triple = triple
        self.n_features_in_ = width

    def predict(self, rows: np.ndarray) -> np.ndarray:
        count = len(rows)
        return np.array([[value] * count for value in self.triple], dtype=float)


def structure() -> Structure:
    return Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])


def write_model(
    directory: Path,
    *,
    triple: tuple[float, float, float] = (0.18, 0.22, 0.30),
    correction: float = 0.01,
    mean_width: float = 0.14,
    feature_schema: str = qrf_features.SCHEMA,
    target_contract: str = K_DISTANCE_CONTRACT,
    units: str | None = "1/angstrom",
    runtime: str = RUNTIME,
    runtime_version: int = RUNTIME_VERSION,
    record_schema_version: int = 1,
    method: str = CALIBRATION_METHOD,
    endpoint_adjustment: str = ENDPOINT_ADJUSTMENT,
    width: int | None = None,
    columns: tuple[str, ...] | None = None,
    pin_estimator: bool = True,
    requires_artifacts: list[dict[str, str]] | None = None,
) -> Path:
    """Write the estimator and record a training run would leave behind."""
    directory.mkdir(parents=True, exist_ok=True)
    names = qrf_features.column_names() if columns is None else columns
    estimator_path = directory / "QRF95.pkl"
    with estimator_path.open("wb") as handle:
        pickle.dump(
            StubEstimator(triple, len(names) if width is None else width), handle
        )
    artifacts: dict[str, Any] = {"estimator": "QRF95.pkl"}
    if pin_estimator:
        artifacts["estimator_sha256"] = sha256_file(estimator_path)
    record: dict[str, Any] = {
        "record_schema_version": record_schema_version,
        "runtime": {"id": runtime, "version": runtime_version},
        "trainer": "quantile_random_forest",
        "target": {
            "name": "k_distance",
            "contract": target_contract,
            "units": units,
        },
        "feature_schema": feature_schema,
        "feature_columns": list(names),
        "feature_parameters": {},
        "requires_artifacts": requires_artifacts or [],
        "calibration": {
            "method": method,
            "endpoint_adjustment": endpoint_adjustment,
            "coverage": 0.9,
            "correction": correction,
            "mean_interval_width": mean_width,
        },
        "artifacts": artifacts,
    }
    (directory / "model.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


def load(directory: Path, **kwargs: Any) -> StructureModel:
    kwargs.setdefault(
        "artifacts",
        {
            "metallicity_checkpoint": directory / "is_metal.ckpt",
            "metallicity_atom_init": directory / "atom_init.json",
        },
    )
    return load_model(directory, **kwargs)


@pytest.fixture
def stub_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the 483-column pipeline; this suite tests the seam, not features."""
    monkeypatch.setattr(
        qrf_features,
        "feature_rows",
        lambda structures, **kwargs: np.zeros((len(structures), 483)),
    )


def test_a_prediction_names_the_parameter_it_advises(
    tmp_path: Path, stub_features: None
) -> None:
    """Core routes on the parameter, so a model must say which one it speaks to."""
    model = load(
        write_model(tmp_path / "model"), model_id="k_points.k_distance.qrf@test"
    )

    prediction = model.predict(structure())

    assert isinstance(prediction, ModelPrediction)
    assert prediction.parameter == "k_points"
    assert prediction.quantity == "k_distance"
    assert prediction.value == pytest.approx(0.22)
    assert prediction.target_contract == K_DISTANCE_CONTRACT
    assert prediction.model_id == "k_points.k_distance.qrf@test"
    assert prediction.confidence == pytest.approx(0.9)


def test_the_interval_travels_in_details_not_in_the_value(
    tmp_path: Path, stub_features: None
) -> None:
    """Core acts on one number; the interval is recorded, never branched on."""
    model = load(write_model(tmp_path / "model"))

    details = model.predict(structure()).details

    assert details is not None
    assert details["interval"] == pytest.approx([0.17, 0.31])
    assert details["coverage"] == pytest.approx(0.9)
    assert details["units"] == "1/angstrom"


def test_a_typical_interval_raises_no_warning(
    tmp_path: Path, stub_features: None
) -> None:
    model = load(write_model(tmp_path / "model", mean_width=0.14))

    assert model.predict(structure()).warnings == ()


def test_an_unusually_wide_interval_warns_the_consumer(
    tmp_path: Path, stub_features: None
) -> None:
    """A structure unlike the training set is flagged here, not in Core."""
    model = load(write_model(tmp_path / "model", mean_width=0.01))

    warnings = model.predict(structure()).warnings

    assert len(warnings) == 1
    assert "verify k-point convergence" in warnings[0]


def test_a_single_structure_and_a_batch_agree(
    tmp_path: Path, stub_features: None
) -> None:
    """The estimator flattens a one-row prediction; the seam must not care."""
    model = load(write_model(tmp_path / "model"))

    one = model.predict(structure())
    batch = model.predict_batch([structure(), structure()])

    assert [prediction.value for prediction in batch] == pytest.approx(
        [one.value, one.value]
    )


def test_an_empty_batch_predicts_nothing(tmp_path: Path, stub_features: None) -> None:
    model = load(write_model(tmp_path / "model"))

    assert model.predict_batch([]) == []


def test_declared_artifacts_are_resolved_and_verified(
    tmp_path: Path, stub_features: None
) -> None:
    """A consumer never learns that QRF95 depends on a metallicity checkpoint."""
    store = tmp_path / "artifacts" / "ptc95-vbq12"
    store.mkdir(parents=True)
    declared = []
    for name, filename in (
        ("metallicity_checkpoint", "is_metal.ckpt"),
        ("metallicity_atom_init", "atom_init.json"),
    ):
        path = store / filename
        path.write_text(f"contents of {filename}", encoding="utf-8")
        declared.append(
            {
                "name": name,
                "record_id": "ptc95-vbq12",
                "file": filename,
                "sha256": sha256_file(path),
            }
        )

    model = load_model(
        write_model(tmp_path / "model", requires_artifacts=declared),
        artifact_directory=tmp_path / "artifacts",
    )

    assert model.predict(structure()).value == pytest.approx(0.22)


def test_a_tampered_artifact_is_refused(tmp_path: Path) -> None:
    """The record pins a digest, so a swapped checkpoint cannot predict."""
    store = tmp_path / "artifacts" / "ptc95-vbq12"
    store.mkdir(parents=True)
    (store / "is_metal.ckpt").write_text("something else", encoding="utf-8")
    declared = [
        {
            "name": "metallicity_checkpoint",
            "record_id": "ptc95-vbq12",
            "file": "is_metal.ckpt",
            "sha256": "0" * 64,
        }
    ]

    with pytest.raises(ValueError, match="SHA-256"):
        load_model(
            write_model(tmp_path / "model", requires_artifacts=declared),
            artifact_directory=tmp_path / "artifacts",
        )


def test_a_missing_supporting_artifact_is_refused(
    tmp_path: Path, stub_features: None
) -> None:
    with pytest.raises(ValueError, match="metallicity_checkpoint"):
        load_model(write_model(tmp_path / "model"), artifacts={})


def test_an_unknown_feature_contract_names_what_to_upgrade(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "model", feature_schema="soap_only.v9")

    with pytest.raises(ValueError, match="upgrade goldilocks-ml"):
        load(directory)


def test_an_unknown_target_contract_is_refused(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "model", target_contract="goldilocks.kppra.v1")

    with pytest.raises(ValueError, match="no DFT parameter is defined"):
        load(directory)


def test_a_runtime_with_no_predictor_is_refused(tmp_path: Path) -> None:
    """Serving dispatches on the runtime, not the fitting algorithm."""
    directory = write_model(tmp_path / "model", runtime="smearing.qrf")

    with pytest.raises(ValueError, match="no predictor implements runtime"):
        load(directory)


def test_a_newer_runtime_version_is_refused(tmp_path: Path) -> None:
    """A record served by different semantics must not load under this one."""
    directory = write_model(tmp_path / "model", runtime_version=RUNTIME_VERSION + 1)

    with pytest.raises(ValueError, match="runtime version"):
        load(directory)


def test_a_newer_record_schema_is_refused(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "model", record_schema_version=99)

    with pytest.raises(ValueError, match="record schema version"):
        load(directory)


def test_units_that_contradict_the_contract_are_refused(tmp_path: Path) -> None:
    """A 2 pi convention error would otherwise give a plausible, wrong mesh."""
    directory = write_model(tmp_path / "model", units="angstrom")

    with pytest.raises(ValueError, match="expects units"):
        load(directory)


def test_reordered_feature_columns_are_refused(tmp_path: Path) -> None:
    """A matching width over a scrambled contract must not predict."""
    scrambled = tuple(reversed(qrf_features.column_names()))
    directory = write_model(tmp_path / "model", columns=scrambled)

    with pytest.raises(ValueError, match="columns this build produces differ"):
        load(directory)


def test_an_unknown_calibration_method_is_refused(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "model", method="jackknife_plus")

    with pytest.raises(ValueError, match="calibration"):
        load(directory)


def test_an_unknown_endpoint_rule_is_refused(tmp_path: Path) -> None:
    """A record asking to sort would be silently clamped instead."""
    directory = write_model(tmp_path / "model", endpoint_adjustment="sorted")

    with pytest.raises(ValueError, match="endpoint rule"):
        load(directory)


def test_an_unpinned_estimator_is_not_unpickled(tmp_path: Path) -> None:
    """Unpickling executes code, so the record must say what is allowed."""
    directory = write_model(tmp_path / "model", pin_estimator=False)

    with pytest.raises(ValueError, match="refusing"):
        load(directory)


def test_a_substituted_estimator_is_refused(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "model")
    with (directory / "QRF95.pkl").open("wb") as handle:
        pickle.dump(StubEstimator((0.1, 0.2, 0.3)), handle)

    with pytest.raises(ValueError, match="its record pins"):
        load(directory)


def test_an_estimator_that_disagrees_with_its_record_is_refused(
    tmp_path: Path,
) -> None:
    """A 483-column record over a 500-column estimator must not predict."""
    directory = write_model(tmp_path / "model", width=500)

    with pytest.raises(ValueError, match="artifact and its record disagree"):
        load(directory)


def test_every_contract_names_a_parameter_and_a_quantity() -> None:
    """Core dispatches on these; a blank here is a runtime failure there."""
    for name in inference.CONTRACTS:
        contract = contract_for(name)
        assert contract.parameter and contract.quantity


def test_contracts_say_what_kind_of_advice_they_carry() -> None:
    from goldilocks_ml.inference import (
        DFT_PARAMETER,
        KINDS,
        MATERIAL_PROPERTY,
        contract_for,
    )

    mesh = contract_for("goldilocks.k_distance.mesh_lower_bound.2pi.v1")
    metal = contract_for("goldilocks.is_metal.dft_band_gap_zero.v1")

    # A k-distance is written into an input file; metallicity is a fact about
    # the material that several inputs depend on.
    assert mesh.kind == DFT_PARAMETER
    assert metal.kind == MATERIAL_PROPERTY
    assert {mesh.kind, metal.kind} <= KINDS
    assert metal.parameter == "metallicity"
    assert metal.quantity == "is_metal"


def test_a_boolean_quantity_refuses_a_number() -> None:
    from goldilocks_ml.inference import contract_for

    contract = contract_for("goldilocks.is_metal.dft_band_gap_zero.v1")

    contract.check_value(True)
    contract.check_value(False)
    with pytest.raises(ValueError, match="must be a boolean"):
        contract.check_value(0.87)
