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
from goldilocks_ml.inference import (
    KMeshModel,
    KMeshPrediction,
    QRF95Inference,
    load_kmesh_model,
)
from goldilocks_ml.models.kmesh.qrf95 import features as qrf_features

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
    width: int = 483,
    columns: int = 483,
) -> Path:
    """Write the estimator and record a training run would leave behind."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "QRF95.pkl").open("wb") as handle:
        pickle.dump(StubEstimator(triple, width), handle)
    record: dict[str, Any] = {
        "trainer": "quantile_random_forest",
        "target": {
            "name": "k_distance",
            "contract": target_contract,
            "units": "1/angstrom",
        },
        "feature_schema": feature_schema,
        "feature_columns": [f"f{index}" for index in range(columns)],
        "feature_parameters": {},
        "calibration": {
            "coverage": 0.9,
            "correction": correction,
            "mean_interval_width": mean_width,
        },
        "artifacts": {"estimator": "QRF95.pkl"},
    }
    (directory / "model.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


def load(directory: Path, **kwargs: Any) -> KMeshModel:
    return load_kmesh_model(
        directory,
        metallicity_checkpoint=directory / "is_metal.ckpt",
        metallicity_atom_init=directory / "atom_init.json",
        **kwargs,
    )


@pytest.fixture
def stub_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the 483-column pipeline; this suite tests the seam, not features."""
    monkeypatch.setattr(
        qrf_features,
        "feature_rows",
        lambda structures, **kwargs: np.zeros((len(structures), 483)),
    )


def test_prediction_reports_the_calibrated_median_and_its_quantity(
    tmp_path: Path, stub_features: None
) -> None:
    model = load(write_model(tmp_path / "model"), model_id="kmesh/qrf95@test")

    prediction = model.predict(structure())

    assert isinstance(prediction, KMeshPrediction)
    assert prediction.quantity == "k_distance"
    assert prediction.value == pytest.approx(0.22)
    assert prediction.target_contract == K_DISTANCE_CONTRACT
    assert prediction.model_id == "kmesh/qrf95@test"
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
    assert isinstance(model, QRF95Inference)

    one = model.predict(structure())
    batch = model.predict_batch([structure(), structure()])

    assert [prediction.value for prediction in batch] == pytest.approx(
        [one.value, one.value]
    )


def test_an_empty_batch_predicts_nothing(tmp_path: Path, stub_features: None) -> None:
    model = load(write_model(tmp_path / "model"))
    assert isinstance(model, QRF95Inference)

    assert model.predict_batch([]) == []


def test_an_unknown_feature_contract_names_what_to_upgrade(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "model", feature_schema="soap_only.v9")

    with pytest.raises(ValueError, match="upgrade goldilocks-ml"):
        load(directory)


def test_an_unknown_target_contract_is_refused(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "model", target_contract="goldilocks.kppra.v1")

    with pytest.raises(ValueError, match="no k-mesh quantity is defined"):
        load(directory)


def test_an_estimator_that_disagrees_with_its_record_is_refused(
    tmp_path: Path,
) -> None:
    """A 483-column record over a 500-column estimator must not predict."""
    directory = write_model(tmp_path / "model", width=500, columns=483)

    with pytest.raises(ValueError, match="artifact and its record disagree"):
        load(directory)


def test_every_published_quantity_is_a_string(tmp_path: Path) -> None:
    """Core dispatches on these; a typo here is a runtime failure there."""
    assert all(
        isinstance(quantity, str) and quantity
        for quantity in inference.QUANTITY_BY_TARGET_CONTRACT.values()
    )
