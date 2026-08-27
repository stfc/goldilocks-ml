"""Tests for the registry, the feature contract, and the baseline trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import (
    build_snapshot,
    classification_document,
    regression_document,
    write_protocol,
)

from goldilocks_ml.baselines import Standardizer
from goldilocks_ml.cli import build_features
from goldilocks_ml.protocol import load_protocol
from goldilocks_ml.registry import (
    FeatureMatrix,
    FittedModel,
    TrainingContext,
    TrainingPartition,
    feature_contract_names,
    get_feature_contract,
    get_trainer,
    register_feature_contract,
    register_trainer,
    trainer_names,
)
from goldilocks_ml.snapshot import Sample, load_snapshot


def _setup(
    tmp_path: Path,
    snapshot_dir: Path,
    *,
    classification: bool = False,
    structures: bool = False,
    **overrides: Any,
):
    build_snapshot(
        snapshot_dir,
        target="label" if classification else "value",
        structures=structures,
    )
    document = (
        classification_document(**overrides)
        if classification
        else regression_document(**overrides)
    )
    protocol = load_protocol(write_protocol(tmp_path / "protocol.toml", document))
    snapshot = load_snapshot(snapshot_dir, protocol)
    features, artifacts = build_features(protocol, snapshot, tmp_path / "artifacts")
    context = TrainingContext(
        train=TrainingPartition(samples=snapshot.samples, features=features),
        validation=None,
        calibration=None,
        artifacts=artifacts,
        output_dir=tmp_path / "model",
    )
    return protocol, snapshot, context


def test_the_shipped_trainers_and_contracts_are_registered() -> None:
    assert "linear_regression" in trainer_names()
    assert "logistic_regression" in trainer_names()
    assert "tabular" in feature_contract_names()


@pytest.mark.parametrize(
    ("lookup", "name", "message"),
    [
        (get_trainer, "qrf", "unknown trainer 'qrf'; registered:"),
        (get_feature_contract, "soap", "unknown feature contract 'soap'; registered:"),
    ],
)
def test_unknown_names_list_what_is_registered(lookup, name, message) -> None:
    with pytest.raises(ValueError, match=message):
        lookup(name)


@pytest.mark.parametrize(
    ("register", "name"),
    [
        (register_trainer, "linear_regression"),
        (register_feature_contract, "tabular"),
    ],
)
def test_registration_rejects_a_duplicate_name(register, name) -> None:
    with pytest.raises(ValueError, match="already registered"):
        register(name, lambda *args: None)


def test_feature_matrix_reports_a_skipped_sample(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    _, snapshot, _ = _setup(tmp_path, snapshot_dir)
    partial = FeatureMatrix(columns=("x1",), rows={"syn-000": (1.0,)})

    with pytest.raises(ValueError, match="no row for 23 sample"):
        partial.validate(snapshot)


def test_feature_matrix_reports_an_inconsistent_width(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    _, snapshot, _ = _setup(tmp_path, snapshot_dir)
    ragged = FeatureMatrix(
        columns=("x1", "x2"),
        rows={sample_id: (1.0,) for sample_id in snapshot.sample_ids},
    )

    with pytest.raises(ValueError, match="has 1 features; expected 2"):
        ragged.validate(snapshot)


def test_the_tabular_contract_selects_requested_columns(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    _, _, context = _setup(
        tmp_path, snapshot_dir, features={"parameters": {"columns": ["x3", "x1"]}}
    )

    assert context.train.features.columns == ("x3", "x1")
    assert context.train.features.rows["syn-000"] == (0.0, -2.0)


def test_the_tabular_contract_rejects_a_missing_column(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    with pytest.raises(ValueError, match="missing column\\(s\\): soap_0"):
        _setup(tmp_path, snapshot_dir, features={"parameters": {"columns": ["soap_0"]}})


def test_the_tabular_contract_rejects_unknown_parameters(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    with pytest.raises(ValueError, match="unknown tabular feature parameter"):
        _setup(tmp_path, snapshot_dir, features={"parameters": {"radius": 10.0}})


def test_the_tabular_contract_needs_a_features_file(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    build_snapshot(snapshot_dir, features=False)
    protocol = load_protocol(
        write_protocol(
            tmp_path / "protocol.toml", regression_document(dataset={"requires": []})
        )
    )
    snapshot = load_snapshot(snapshot_dir, protocol)

    with pytest.raises(ValueError, match="manifest declares none"):
        build_features(protocol, snapshot, tmp_path / "artifacts")


def test_standardizer_uses_only_the_rows_it_is_given() -> None:
    standardizer = Standardizer.fit([(1.0,), (3.0,)])

    assert standardizer.means == (2.0,)
    assert standardizer.scales == (1.0,)
    assert standardizer.apply((5.0,)) == [3.0]


def test_standardizer_keeps_constant_features_finite() -> None:
    standardizer = Standardizer.fit([(2.0,), (2.0,)])

    assert standardizer.scales == (1.0,)
    assert standardizer.apply((2.0,)) == [0.0]


def test_standardizer_rejects_an_empty_split() -> None:
    with pytest.raises(ValueError, match="empty split"):
        Standardizer.fit([])


def test_linear_regression_recovers_a_known_linear_target(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot, context = _setup(tmp_path, snapshot_dir)

    model = get_trainer(protocol.trainer)(protocol, context)

    predicted = model.predict(snapshot.samples, context.train.features)
    truth = [float(sample.target) for sample in snapshot.samples]
    assert predicted == pytest.approx(truth, abs=1e-6)


def test_linear_regression_saves_a_readable_model(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot, context = _setup(tmp_path, snapshot_dir)
    model = get_trainer(protocol.trainer)(protocol, context)
    directory = tmp_path / "model"
    directory.mkdir()

    model.save(directory)

    assert isinstance(model, FittedModel)
    description = model.describe()
    assert description["trainer"] == "linear_regression"
    assert description["deterministic"] is True
    assert description["feature_columns"] == ["x1", "x2", "x3"]
    assert (directory / "model.json").is_file()


def test_linear_regression_rejects_a_negative_penalty(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot, context = _setup(
        tmp_path, snapshot_dir, model={"parameters": {"l2": -1.0}}
    )

    with pytest.raises(ValueError, match="l2 must not be negative"):
        get_trainer(protocol.trainer)(protocol, context)


def test_logistic_regression_separates_a_separable_target(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot, context = _setup(
        tmp_path,
        snapshot_dir,
        classification=True,
        model={"parameters": {"iterations": 800}},
    )

    model = get_trainer(protocol.trainer)(protocol, context)

    scores = model.predict(snapshot.samples, context.train.features)
    for sample, score in zip(snapshot.samples, scores, strict=True):
        assert (score > 0.5) is (str(sample.target) == "metal")


def test_logistic_regression_is_deterministic(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot, context = _setup(tmp_path, snapshot_dir, classification=True)
    trainer = get_trainer(protocol.trainer)

    first_model = trainer(protocol, context)
    second_model = trainer(protocol, context)
    first = first_model.predict(snapshot.samples, context.train.features)
    second = second_model.predict(snapshot.samples, context.train.features)

    assert first == second


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"l2": -1.0}, "l2 must not be negative"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"iterations": 0}, "iterations must be positive"),
    ],
)
def test_logistic_regression_rejects_invalid_hyperparameters(
    tmp_path: Path, snapshot_dir: Path, parameters: dict[str, Any], message: str
) -> None:
    protocol, snapshot, context = _setup(
        tmp_path, snapshot_dir, classification=True, model={"parameters": parameters}
    )

    with pytest.raises(ValueError, match=message):
        get_trainer(protocol.trainer)(protocol, context)


def test_a_trainer_only_sees_the_samples_it_is_handed(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot, context = _setup(tmp_path, snapshot_dir)
    subset = snapshot.samples[:6]

    restricted = TrainingContext(
        train=TrainingPartition(
            samples=subset, features=context.train.features.subset(subset)
        ),
        validation=None,
        calibration=None,
        artifacts=context.artifacts,
        output_dir=context.output_dir,
    )
    model = get_trainer(protocol.trainer)(protocol, restricted)

    fitted_on = Standardizer.fit(restricted.train.features.matrix(subset))
    assert model.describe()["standardizer"]["means"] == list(fitted_on.means)


def test_training_context_has_no_test_partition(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    _, _, context = _setup(tmp_path, snapshot_dir)

    assert not hasattr(context, "test")
    assert not hasattr(context, "snapshot")


def test_a_sample_outside_the_feature_matrix_is_reported() -> None:
    matrix = FeatureMatrix(columns=("x",), rows={"a": (1.0,)})

    with pytest.raises(ValueError, match="no row for b"):
        matrix.matrix([Sample("b", 1.0, None, None)])
