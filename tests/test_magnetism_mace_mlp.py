"""Tests for the ``is_magnetic`` classifier and its embedding feature contract.

Deliberately mace-free except for one skip-guarded end-to-end test: every
other test here monkeypatches the embedding step, mirroring
``tests/test_inference.py``'s ``stub_features`` fixture for QRF95. The
``mace``/``e3nn``/``sphericart`` stack is a separate, heavier extra
(``magnetism``) than the rest of ``models``, and this suite must pass without
it installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import write_protocol
from pymatgen.core import Lattice, Structure

from goldilocks_ml.models.magnetism._mace_backbone import EMBEDDING_WIDTH
from goldilocks_ml.models.magnetism.is_magnetic.mace_mlp import (
    features as mace_features,
)
from goldilocks_ml.models.magnetism.is_magnetic.mace_mlp import (
    predictor as mace_predictor,
)
from goldilocks_ml.models.magnetism.is_magnetic.mace_mlp.trainer import (
    RUNTIME,
    RUNTIME_VERSION,
    TRAINER,
    MagneticMLP,
    MagneticMLPClassifier,
)
from goldilocks_ml.protocol import load_protocol
from goldilocks_ml.registry import (
    TrainingContext,
    feature_contract_names,
    get_trainer,
    trainer_names,
)

IS_MAGNETIC_CONTRACT = "goldilocks.is_magnetic.dft_max_site_magmom_0p5ub.v1"
TINY_ARCHITECTURE = {"d_in": EMBEDDING_WIDTH, "hidden": [4], "dropout": 0.0}


def a_structure() -> Structure:
    return Structure(Lattice.cubic(4.0), ["Fe"], [[0.0, 0.0, 0.0]])


def document(**parameters: Any) -> dict[str, Any]:
    """A minimal classification protocol document for this trainer.

    Not read from a shipped protocol file: this model has no released
    protocol yet (there is no MatPES snapshot in goldilocks-data's format to
    pin one against), so the document is built by hand, the same way
    ``tests/conftest.py``'s own synthetic protocols are.
    """
    return {
        "schema_version": 1,
        "id": "magnetism.is_magnetic.mace_mlp.test_dataset.v1",
        "task": "classification",
        "trainer": TRAINER,
        "dataset": {
            "target": "is_magnetic",
            "target_contract": IS_MAGNETIC_CONTRACT,
            "requires": ["structures"],
        },
        "split": {
            "method": "random",
            "train": 0.5,
            "validation": 0.2,
            "calibration": 0.1,
            "test": 0.2,
            "seed": 7,
        },
        "features": {"schema": mace_features.SCHEMA, "parameters": {}},
        "model": {"seed": 7, "parameters": parameters},
        "evaluation": {
            "primary_metric": "roc_auc",
            "metrics": ["accuracy", "roc_auc"],
            "baseline": "train_majority",
            "positive_label": "magnetic",
        },
    }


def unpinned(tmp_path: Path, **parameters: Any) -> Any:
    return load_protocol(
        write_protocol(tmp_path / "protocol.toml", document(**parameters))
    )


def empty_context() -> TrainingContext:
    return TrainingContext(
        train=None,  # type: ignore[arg-type]
        validation=None,
        calibration=None,
        artifacts={},
        output_dir=Path(),
    )


def test_the_trainer_and_contract_are_registered() -> None:
    assert TRAINER in trainer_names()
    assert mace_features.SCHEMA in feature_contract_names()
    assert get_trainer(TRAINER) is not None


def test_the_runtime_is_distinct_from_the_metallicity_one() -> None:
    assert RUNTIME == "magnetism.is_magnetic.mace_mlp"
    assert RUNTIME != "metallicity.is_metal.cgcnn"
    assert RUNTIME_VERSION == 1


def test_column_names_are_a_stable_384_wide_contract() -> None:
    names = mace_features.column_names()

    assert len(names) == EMBEDDING_WIDTH
    assert len(set(names)) == EMBEDDING_WIDTH
    assert all(name.startswith("mace_probe_") for name in names)


def test_the_feature_contract_needs_the_backbone_artifact(tmp_path: Path) -> None:
    from goldilocks_ml.snapshot import Snapshot

    empty_snapshot = Snapshot(
        directory=tmp_path,
        record_id="test",
        snapshot_version="v1",
        manifest_sha256="0" * 64,
        target_name="is_magnetic",
        target_contract=IS_MAGNETIC_CONTRACT,
        target_definition="test",
        target_units=None,
        capabilities=frozenset({"structures"}),
        features_file=None,
        samples=(),
    )

    with pytest.raises(ValueError, match="mace_backbone"):
        mace_features.build(unpinned(tmp_path), empty_snapshot, {})


def test_an_unknown_parameter_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, unknown_option=1)

    with pytest.raises(ValueError, match=f"unknown {TRAINER} parameter"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_an_unknown_selection_metric_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, selection_metric="f1")

    with pytest.raises(ValueError, match="selection_metric must be one of"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_a_non_positive_learning_rate_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, learning_rate=0.0)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_a_malformed_hidden_shape_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, hidden=[0, 64])

    with pytest.raises(ValueError, match="hidden must be"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_an_out_of_range_dropout_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, dropout=1.0)

    with pytest.raises(ValueError, match="dropout must lie in"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_training_without_a_validation_split_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path)

    with pytest.raises(ValueError, match="non-empty validation split"):
        get_trainer(TRAINER)(protocol, empty_context())


def fitted_classifier(**overrides: Any) -> MagneticMLPClassifier:
    """An untrained classifier of the shipped shape, for round-trip tests."""
    torch.manual_seed(0)
    fields: dict[str, Any] = {
        "state_dict": MagneticMLP(**TINY_ARCHITECTURE).state_dict(),
        "architecture": dict(TINY_ARCHITECTURE),
        "scaler_mean": torch.zeros(EMBEDDING_WIDTH),
        "scaler_scale": torch.ones(EMBEDDING_WIDTH),
        "positive_label": "magnetic",
        "negative_label": "non_magnetic",
        "seed": 7,
        "target_name": "is_magnetic",
        "target_contract": IS_MAGNETIC_CONTRACT,
        "feature_columns": mace_features.column_names(),
        "requires_artifacts": (),
        "hyperparameters": {},
        "training": {},
    }
    fields.update(overrides)
    return MagneticMLPClassifier(**fields)


def saved_model(tmp_path: Path, **overrides: Any) -> Path:
    """Write a classifier carrying a decision, and return its directory."""
    model = fitted_classifier(**overrides)
    if "decision" not in overrides:
        model = model.with_decision({"threshold": 0.5, "selected_on": "validation"})
    directory = tmp_path / "model"
    directory.mkdir()
    model.save(directory)
    return directory


@pytest.fixture
def stub_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the mMACE backbone call; this suite tests the seam, not mace."""

    def fake_embed_structures(structures: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros((len(structures), EMBEDDING_WIDTH), dtype=np.float32)

    monkeypatch.setattr(mace_predictor, "embed_structures", fake_embed_structures)


def test_a_served_classifier_decides_rather_than_scoring(
    tmp_path: Path, stub_embedding: None
) -> None:
    from goldilocks_ml.inference import load_model

    directory = saved_model(tmp_path)
    backbone = tmp_path / "mace_matpes_pbe_baseline_run-3.model"

    model = load_model(directory, artifacts={"mace_backbone": backbone})
    prediction = model.predict(a_structure())

    assert isinstance(prediction.value, bool)
    assert prediction.parameter == "magnetism"
    assert prediction.quantity == "is_magnetic"
    assert prediction.target_contract == IS_MAGNETIC_CONTRACT
    assert prediction.confidence is None
    assert 0.0 <= prediction.details["score"] <= 1.0
    assert prediction.details["threshold"] == 0.5
    assert prediction.details["label"] in {"magnetic", "non_magnetic"}
    assert (prediction.details["label"] == "magnetic") is prediction.value


def test_a_batch_decides_once_per_structure(
    tmp_path: Path, stub_embedding: None
) -> None:
    from goldilocks_ml.inference import load_model

    directory = saved_model(tmp_path)
    backbone = tmp_path / "mace_matpes_pbe_baseline_run-3.model"
    model = load_model(directory, artifacts={"mace_backbone": backbone})

    predictions = model.predict_batch([a_structure(), a_structure()])

    assert len(predictions) == 2
    assert predictions[0].details["score"] == predictions[1].details["score"]
    assert model.predict_batch([]) == []


def test_pending_validation_status_is_surfaced_as_a_warning(
    tmp_path: Path, stub_embedding: None
) -> None:
    """A model card caveat must reach the consumer, not stay paper-only."""
    from goldilocks_ml.inference import load_model

    directory = saved_model(tmp_path)
    record = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    record["validation_status"] = {
        "external_validation": "pending",
        "external_validation_note": "MP-ALOE external validation not yet run.",
    }
    (directory / "model.json").write_text(json.dumps(record), encoding="utf-8")
    backbone = tmp_path / "mace_matpes_pbe_baseline_run-3.model"

    model = load_model(directory, artifacts={"mace_backbone": backbone})
    prediction = model.predict(a_structure())

    assert len(prediction.warnings) == 1
    assert "not been externally validated" in prediction.warnings[0]


def test_a_record_without_a_threshold_cannot_be_served(
    tmp_path: Path, stub_embedding: None
) -> None:
    from goldilocks_ml.inference import load_model

    directory = saved_model(tmp_path, decision={})

    with pytest.raises(ValueError, match="records no decision threshold"):
        load_model(directory, artifacts={"mace_backbone": tmp_path / "backbone.model"})


def test_substituted_weights_are_refused(tmp_path: Path, stub_embedding: None) -> None:
    from goldilocks_ml.inference import load_model

    directory = saved_model(tmp_path)
    weights = directory / "is_magnetic.pt"
    weights.write_bytes(weights.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="its record pins"):
        load_model(directory, artifacts={"mace_backbone": tmp_path / "backbone.model"})


def test_the_backbone_artifact_must_be_supplied(
    tmp_path: Path, stub_embedding: None
) -> None:
    from goldilocks_ml.inference import load_model

    directory = saved_model(tmp_path)

    with pytest.raises(ValueError, match="needs artifact"):
        load_model(directory, artifacts={})


def test_reordered_feature_columns_are_refused(
    tmp_path: Path, stub_embedding: None
) -> None:
    """A matching width over a scrambled contract must not predict."""
    from goldilocks_ml.inference import load_model

    scrambled = tuple(reversed(mace_features.column_names()))
    directory = saved_model(tmp_path, feature_columns=scrambled)

    with pytest.raises(ValueError, match="columns this build produces differ"):
        load_model(directory, artifacts={"mace_backbone": tmp_path / "backbone.model"})


def test_the_real_backbone_end_to_end(tmp_path: Path) -> None:
    """A coarse sanity check against the real checkpoint, when it is present.

    Never runs in CI: the ~80 MB backbone is undistributed pending its
    licence (see deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md),
    and the `magnetism` extra (mace/e3nn/sphericart) is not installed there.
    """
    checkpoint = Path(
        "local_data/artifacts/UNPUBLISHED-PENDING-LICENCE/"
        "mace_matpes_pbe_baseline_run-3.model"
    )
    if not checkpoint.is_file():
        pytest.skip("the mMACE backbone checkpoint is not present locally")
    try:
        import mace  # noqa: F401
    except ImportError:
        pytest.skip("the magnetism extra is not installed")

    from goldilocks_ml.models.magnetism._mace_backbone import embed_structures

    rows = embed_structures([a_structure()], checkpoint=checkpoint)

    assert rows.shape == (1, EMBEDDING_WIDTH)
    assert np.isfinite(rows).all()
