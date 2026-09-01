"""Tests for QRF95 features, training, calibration, and run output."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from conftest import build_snapshot, regression_document, write_protocol
from pymatgen.core import Lattice, Structure

from goldilocks_ml.cli import execute
from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.inference import SUPPORTED_RECORD_SCHEMA_VERSIONS
from goldilocks_ml.models.k_points.k_distance.qrf import features as qrf_features
from goldilocks_ml.models.k_points.k_distance.qrf.features import (
    SOAP_DEFAULTS,
    TOTAL_WIDTH,
    composition_block,
    lattice_block,
    soap_block,
    structure_block,
)
from goldilocks_ml.models.k_points.k_distance.qrf.trainer import (
    CALIBRATION_METHOD,
    ENDPOINT_ADJUSTMENT,
    RUNTIME,
    RUNTIME_VERSION,
    calibrate_interval,
    conformal_correction,
)
from goldilocks_ml.protocol import load_protocol
from goldilocks_ml.registry import get_trainer
from goldilocks_ml.snapshot import Sample, Snapshot, load_snapshot

QRF_PROTOCOL = (
    Path(__file__).parents[1]
    / "protocols/k_points/k_distance/qrf/goldilocks_kdist_ultra.v1.toml"
)


def _qrf_document(**parameters: object) -> dict[str, object]:
    document = regression_document(
        id="k_points.k_distance.qrf.synthetic.v1",
        trainer="quantile_random_forest",
        split={
            "train": 0.5,
            "validation": 0.1,
            "calibration": 0.2,
            "test": 0.2,
        },
    )
    document["model"] = {
        "seed": 17,
        "parameters": {
            "n_estimators": 12,
            "quantiles": [0.05, 0.5, 0.95],
            **parameters,
        },
    }
    return document


def test_feature_blocks_have_the_published_widths() -> None:
    structure = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25] * 3])

    assert composition_block([structure]).shape == (1, 146)
    assert structure_block([structure]).shape == (1, 6)
    assert soap_block([structure], SOAP_DEFAULTS).shape == (1, 252)
    assert lattice_block([structure]).shape == (1, 15)
    assert 146 + 6 + 252 + 15 + 64 == TOTAL_WIDTH


def test_qrf_protocol_binds_the_scientific_contract() -> None:
    protocol = load_protocol(QRF_PROTOCOL)

    assert protocol.trainer == "quantile_random_forest"
    assert protocol.dataset.target == "k_distance"
    assert (
        protocol.dataset.target_contract
        == "goldilocks.k_distance.mesh_lower_bound.2pi.v1"
    )
    assert protocol.features.schema == "comp_struct_soap_lattice_metal.v1"


def test_feature_contract_concatenates_blocks_in_published_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure_path = tmp_path / "sample.cif"
    Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]]).to(filename=structure_path)
    sample = Sample("sample", 0.2, "Si", structure_path)
    snapshot = Snapshot(
        directory=tmp_path,
        record_id="fixture",
        snapshot_version="v1",
        manifest_sha256="a" * 64,
        target_name="k_distance",
        target_contract="goldilocks.k_distance.mesh_lower_bound.2pi.v1",
        target_definition="Fixture target.",
        target_units="1/angstrom",
        capabilities=frozenset({"structures", "groups"}),
        features_file=None,
        samples=(sample,),
    )
    monkeypatch.setattr(
        qrf_features, "composition_block", lambda structures: np.full((1, 146), 1.0)
    )
    monkeypatch.setattr(
        qrf_features, "structure_block", lambda structures: np.full((1, 6), 2.0)
    )
    monkeypatch.setattr(
        qrf_features,
        "soap_block",
        lambda structures, parameters: np.full((1, 252), 3.0),
    )
    monkeypatch.setattr(
        qrf_features, "lattice_block", lambda structures: np.full((1, 15), 4.0)
    )
    monkeypatch.setattr(
        qrf_features,
        "metallicity_block",
        lambda structures, checkpoint, atom_init: np.full((1, 64), 5.0),
    )

    matrix = qrf_features.build(
        load_protocol(QRF_PROTOCOL),
        snapshot,
        {
            "metallicity_checkpoint": tmp_path / "checkpoint",
            "metallicity_atom_init": tmp_path / "atom_init.json",
        },
    )

    row = matrix.rows["sample"]
    assert row[:146] == (1.0,) * 146
    assert row[146:152] == (2.0,) * 6
    assert row[152:404] == (3.0,) * 252
    assert row[404:419] == (4.0,) * 15
    assert row[419:] == (5.0,) * 64


def test_conformal_correction_uses_calibration_count_not_test_count() -> None:
    correction = conformal_correction([0.0, 10.0], [1.0, 9.0], [2.0, 9.5], coverage=0.5)

    assert correction == pytest.approx(1.0)


def test_positive_correction_widens_without_clamping() -> None:
    lower, median, upper = calibrate_interval(0.20, 0.25, 0.30, 0.05)

    assert (lower, median, upper) == pytest.approx((0.15, 0.25, 0.35))


def test_negative_correction_clamps_the_endpoint_that_passed_the_median() -> None:
    """Narrowing past the median moves only the offending endpoint."""
    lower, median, upper = calibrate_interval(0.20, 0.21, 0.30, -0.03)

    # Raw calibration gives (0.23, 0.27), which no longer contains the median.
    assert (lower, median, upper) == pytest.approx((0.21, 0.21, 0.27))
    assert lower <= median <= upper


def test_inverted_interval_collapses_onto_the_median_rather_than_sorting() -> None:
    """Clamping is not a sort: the endpoints do not swap."""
    lower, median, upper = calibrate_interval(0.20, 0.25, 0.22, -0.03)

    # Raw calibration gives (0.23, 0.19), which is inverted. Sorting the triple
    # would yield (0.19, 0.23, 0.25); clamping collapses both ends to the median.
    assert (lower, median, upper) == pytest.approx((0.23, 0.25, 0.25))
    assert lower <= median <= upper


def test_qrf_run_records_intervals_and_core_artifacts(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    protocol = load_protocol(write_protocol(tmp_path / "qrf.toml", _qrf_document()))
    snapshot = load_snapshot(snapshot_dir, protocol)

    result = execute(
        protocol,
        snapshot,
        tmp_path / "run",
        artifact_dir=tmp_path / "artifacts",
        splits_source=None,
        overwrite=False,
    )

    model_dir = result["directory"] / "model"
    assert (model_dir / "QRF95.pkl").is_file()
    assert (model_dir / "calibration.json").is_file()
    assert (model_dir / "model.json").is_file()
    with (model_dir / "QRF95.pkl").open("rb") as handle:
        estimator = pickle.load(handle)
    assert estimator.q == [0.05, 0.5, 0.95]
    assert estimator.random_state == 17
    test_metrics = result["metrics"]["splits"]["model"]["test"]
    assert 0.0 <= test_metrics["interval_coverage"] <= 1.0
    assert test_metrics["mean_interval_width"] >= 0.0

    with (result["directory"] / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    model_rows = [row for row in rows if row["source"] == "model"]
    assert all(row["lower"] and row["upper"] for row in model_rows)
    assert all(
        float(row["lower"]) <= float(row["prediction"]) <= float(row["upper"])
        for row in model_rows
    )


def test_the_written_record_carries_what_loading_requires(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    """The trainer writes the record; the loader reads it. They must agree."""
    build_snapshot(snapshot_dir)
    protocol = load_protocol(write_protocol(tmp_path / "qrf.toml", _qrf_document()))
    snapshot = load_snapshot(snapshot_dir, protocol)

    result = execute(
        protocol,
        snapshot,
        tmp_path / "run",
        artifact_dir=tmp_path / "artifacts",
        splits_source=None,
        overwrite=False,
    )

    model_dir = result["directory"] / "model"
    record = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))

    assert record["record_schema_version"] in SUPPORTED_RECORD_SCHEMA_VERSIONS
    assert record["runtime"] == {"id": RUNTIME, "version": RUNTIME_VERSION}
    assert record["calibration"]["method"] == CALIBRATION_METHOD
    assert record["calibration"]["endpoint_adjustment"] == ENDPOINT_ADJUSTMENT
    assert "mean_interval_width" in record["calibration"]
    assert record["target"]["contract"] == protocol.dataset.target_contract
    assert record["feature_parameters"] == protocol.features.parameters
    assert record["artifacts"]["estimator_sha256"] == sha256_file(
        model_dir / record["artifacts"]["estimator"]
    )


def test_qrf_training_is_repeatable_for_one_seed(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir)
    protocol = load_protocol(write_protocol(tmp_path / "qrf.toml", _qrf_document()))
    snapshot = load_snapshot(snapshot_dir, protocol)

    first = execute(
        protocol,
        snapshot,
        tmp_path / "first",
        artifact_dir=tmp_path / "artifacts",
        splits_source=None,
        overwrite=False,
    )
    second = execute(
        protocol,
        snapshot,
        tmp_path / "second",
        artifact_dir=tmp_path / "artifacts",
        splits_source=first["directory"] / "splits.csv",
        overwrite=False,
    )

    assert (first["directory"] / "predictions.csv").read_bytes() == (
        second["directory"] / "predictions.csv"
    ).read_bytes()
    assert json.loads((first["directory"] / "model" / "model.json").read_text()) == (
        json.loads((second["directory"] / "model" / "model.json").read_text())
    )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"n_estimators": 0}, "positive integer"),
        ({"quantiles": [0.1, 0.5]}, "three numbers"),
        ({"quantiles": [0.05, 0.4, 0.95]}, "0.5 median"),
        ({"n_jobs": 0}, "non-zero integer"),
        ({"max_depth": 4}, "unknown quantile forest parameter"),
    ],
)
def test_qrf_rejects_invalid_parameters(
    tmp_path: Path,
    snapshot_dir: Path,
    parameters: dict[str, object],
    message: str,
) -> None:
    build_snapshot(snapshot_dir)
    protocol = load_protocol(
        write_protocol(tmp_path / "qrf.toml", _qrf_document(**parameters))
    )
    snapshot = load_snapshot(snapshot_dir, protocol)
    from goldilocks_ml.cli import build_features
    from goldilocks_ml.registry import TrainingContext, TrainingPartition

    features, _ = build_features(protocol, snapshot, tmp_path / "artifacts")
    train = snapshot.samples[:16]
    calibration = snapshot.samples[16:]
    context = TrainingContext(
        train=TrainingPartition(train, features.subset(train)),
        validation=None,
        calibration=TrainingPartition(calibration, features.subset(calibration)),
        artifacts={},
        output_dir=tmp_path / "model",
    )

    with pytest.raises(ValueError, match=message):
        get_trainer(protocol.trainer)(protocol, context)
