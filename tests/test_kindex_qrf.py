"""Tests for the CSLR k-index feature and quantile-forest contracts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from conftest import build_snapshot, regression_document, write_protocol
from pymatgen.core import Lattice, Structure

from goldilocks_ml.cli import execute
from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.inference import load_model
from goldilocks_ml.models.k_points.k_distance.qrf.trainer import (
    CALIBRATION_METHOD,
    ENDPOINT_ADJUSTMENT,
    KINDEX_MODEL_FILE,
    KINDEX_RUNTIME,
    KINDEX_RUNTIME_VERSION,
)
from goldilocks_ml.models.k_points.k_index.qrf import features
from goldilocks_ml.protocol import load_protocol
from goldilocks_ml.snapshot import load_snapshot

KINDEX_PROTOCOL = (
    Path(__file__).parents[1] / "protocols/k_points/k_index/qrf/d5ds2_64f16.v1.toml"
)


class ConstantQuantileEstimator:
    """Small pickleable estimator used to exercise the serving seam."""

    n_features_in_ = features.TOTAL_WIDTH

    def predict(self, rows: np.ndarray) -> np.ndarray:
        # Four fitted levels: the q05/q50/q95 interval plus the q90 this model
        # publishes, which is deliberately not the median.
        return np.tile(np.asarray([[1.0], [2.0], [2.6], [4.0]]), (1, len(rows)))


def silicon() -> Structure:
    return Structure(
        Lattice.cubic(3.5),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )


def test_cslr_blocks_match_the_core_contract() -> None:
    structure = silicon()

    assert features.composition_block([structure]).shape == (1, 146)
    assert features.structure_block([structure]).shape == (1, 7)
    assert features.lattice_block([structure]).shape == (1, 7)
    assert features.reciprocal_block([structure]).shape == (1, 14)
    assert features.feature_rows([structure]).shape == (1, features.TOTAL_WIDTH)
    assert len(features.column_names()) == 174


def test_kindex_protocol_pins_the_published_scientific_contract() -> None:
    protocol = load_protocol(KINDEX_PROTOCOL)

    assert protocol.id == "k_points.k_index.qrf.d5ds2_64f16.v1"
    assert protocol.dataset.target == "k_index"
    assert (
        protocol.dataset.target_contract == "goldilocks.k_index.ladder_0based.max50.v1"
    )
    assert protocol.dataset.pinned is not None
    assert protocol.dataset.pinned.record_id == "d5ds2-64f16"
    assert protocol.features.schema == features.SCHEMA


def test_direct_and_reciprocal_blocks_have_semantic_values() -> None:
    structure = silicon()
    direct = features.lattice_block([structure])[0]
    reciprocal = features.reciprocal_block([structure])[0]
    reciprocal_length = 2 * np.pi / 3.5

    assert direct == pytest.approx([3.5, 3.5, 3.5, 90, 90, 90, 3.5**3])
    assert reciprocal[:7] == pytest.approx(
        [
            reciprocal_length,
            reciprocal_length,
            reciprocal_length,
            reciprocal_length**3,
            90,
            90,
            90,
        ]
    )
    assert reciprocal[10:] == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_cslr_output_is_finite_and_deterministic() -> None:
    first = features.feature_rows([silicon()])
    second = features.feature_rows([silicon()])

    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_missing_atomic_radius_uses_documented_structure_fallback() -> None:
    structure = Structure(Lattice.cubic(4.0), ["He"], [[0, 0, 0]])

    with pytest.warns(UserWarning, match="using zeros"):
        block = features.structure_block([structure])

    assert np.array_equal(block, np.zeros((1, features.STRUCTURE_WIDTH)))


@pytest.mark.parametrize("value", [0, -2, True, 1.5, "32"])
def test_cslr_rejects_invalid_batch_sizes(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        features._batch_size({"batch_size": value})


def test_quantile_forest_writes_a_distinct_kindex_runtime(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    digest = build_snapshot(snapshot_dir)
    document = regression_document(
        id="k_points.k_index.qrf.synthetic.v1",
        trainer="quantile_random_forest",
        split={
            "method": "group",
            "train": 0.5,
            "validation": 0.1,
            "calibration": 0.2,
            "test": 0.2,
            "seed": 17,
        },
        dataset={
            "record_id": "synthetic",
            "snapshot_version": "v1",
            "manifest_sha256": digest,
        },
    )
    document["model"] = {
        "seed": 17,
        "parameters": {
            "n_estimators": 8,
            "quantiles": [0.05, 0.5, 0.95],
            "n_jobs": 1,
        },
    }
    protocol = load_protocol(write_protocol(tmp_path / "kindex.toml", document))
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
    assert record["runtime"] == {
        "id": KINDEX_RUNTIME,
        "version": KINDEX_RUNTIME_VERSION,
    }
    assert record["artifacts"]["estimator"] == KINDEX_MODEL_FILE
    assert (model_dir / KINDEX_MODEL_FILE).is_file()


def test_kindex_runtime_loads_verified_estimator_and_serves_cslr(
    tmp_path: Path,
) -> None:
    estimator_path = tmp_path / KINDEX_MODEL_FILE
    with estimator_path.open("wb") as handle:
        pickle.dump(ConstantQuantileEstimator(), handle)
    record = {
        "record_schema_version": 1,
        "runtime": {"id": KINDEX_RUNTIME, "version": KINDEX_RUNTIME_VERSION},
        "trainer": "quantile_random_forest",
        "target": {
            "name": "k_index",
            "contract": "goldilocks.k_index.ladder_0based.max50.v1",
            "units": None,
        },
        "feature_schema": features.SCHEMA,
        "feature_columns": list(features.column_names()),
        "feature_parameters": {"batch_size": 128},
        "requires_artifacts": [],
        "quantiles": [0.05, 0.5, 0.95],
        "levels": [0.05, 0.5, 0.9, 0.95],
        "decision": {
            "rule": "quantile",
            "level": 0.9,
            "metric": "mean_excess",
            "max_underprediction": 0.06,
            "selected_on": "validation",
            "rounding": "half_up",
            "bands": [{"upper": 6, "offset": 0}, {"upper": None, "offset": 2}],
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "coverage": 0.9,
            "correction": 0.5,
            "mean_interval_width": 4.0,
            "endpoint_adjustment": ENDPOINT_ADJUSTMENT,
        },
        "artifacts": {
            "estimator": KINDEX_MODEL_FILE,
            "estimator_sha256": sha256_file(estimator_path),
        },
    }
    (tmp_path / "model.json").write_text(json.dumps(record), encoding="utf-8")

    prediction = load_model(tmp_path).predict(silicon())

    assert prediction.parameter == "k_points"
    assert prediction.quantity == "k_index"
    # The published value is the q90 quantile, not the q50 the interval is
    # centred on, and it comes back as a whole rung: 2.6 rounds up to 3, which
    # falls in the first band and is lifted by nothing.
    assert prediction.value == 3
    assert prediction.confidence == pytest.approx(0.9)
    assert prediction.details == {
        "interval": [0.5, 4.5],
        "coverage": 0.9,
        "calibrated": True,
        "units": None,
        "index_base": 0,
        "max_kpoints_per_axis": 50,
        "decision": {
            "rule": "quantile",
            "level": 0.9,
            "metric": "mean_excess",
            "max_underprediction": 0.06,
            "selected_on": "validation",
            "rounding": "half_up",
            "bands": [{"upper": 6, "offset": 0}, {"upper": None, "offset": 2}],
        },
    }


def test_a_k_index_artifact_without_a_decision_rule_is_refused(tmp_path: Path) -> None:
    """Publishing the median is a choice, and this runtime will not make it."""
    estimator_path = tmp_path / KINDEX_MODEL_FILE
    with estimator_path.open("wb") as handle:
        pickle.dump(ConstantQuantileEstimator(), handle)
    record = {
        "record_schema_version": 1,
        "runtime": {"id": KINDEX_RUNTIME, "version": KINDEX_RUNTIME_VERSION},
        "trainer": "quantile_random_forest",
        "target": {
            "name": "k_index",
            "contract": "goldilocks.k_index.ladder_0based.max50.v1",
            "units": None,
        },
        "feature_schema": features.SCHEMA,
        "feature_columns": list(features.column_names()),
        "feature_parameters": {"batch_size": 128},
        "requires_artifacts": [],
        "quantiles": [0.05, 0.5, 0.95],
        "artifacts": {
            "estimator": KINDEX_MODEL_FILE,
            "estimator_sha256": sha256_file(estimator_path),
        },
    }
    (tmp_path / "model.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="declares no decision rule"):
        load_model(tmp_path)
