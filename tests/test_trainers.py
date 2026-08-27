"""Tests for the trainer interface and the CPU-only fixture trainers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    build_snapshot,
    classification_document,
    regression_document,
    write_protocol,
)

from goldilocks_ml.datasets import Sample, load_snapshot
from goldilocks_ml.protocol import FeatureSpec, load_protocol
from goldilocks_ml.trainers import (
    FittedModel,
    Standardizer,
    get_trainer,
    register_trainer,
    trainer_names,
)


def _sample(sample_id: str, target: float | str, features: tuple[float, ...]) -> Sample:
    return Sample(sample_id=sample_id, target=target, group=None, features=features)


def _setup(
    tmp_path: Path,
    snapshot_dir: Path,
    *,
    classification: bool = False,
    **overrides: Any,
):
    digest = build_snapshot(snapshot_dir)
    document = (
        classification_document(digest, **overrides)
        if classification
        else regression_document(digest, **overrides)
    )
    protocol = load_protocol(write_protocol(tmp_path / "protocol.toml", document))
    return protocol, load_snapshot(snapshot_dir, protocol)


def test_registry_exposes_the_built_in_trainers() -> None:
    assert "linear_regression" in trainer_names()
    assert "logistic_regression" in trainer_names()


def test_get_trainer_reports_the_registered_names() -> None:
    with pytest.raises(ValueError, match="unknown trainer 'qrf'; registered:"):
        get_trainer("qrf")


def test_register_trainer_rejects_a_duplicate_name() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_trainer("linear_regression", lambda protocol, samples: None)


def test_standardizer_uses_only_the_samples_it_is_given() -> None:
    train = [_sample("a", 0.0, (1.0,)), _sample("b", 0.0, (3.0,))]

    standardizer = Standardizer.fit(train)

    assert standardizer.means == (2.0,)
    assert standardizer.scales == (1.0,)
    assert standardizer.apply(_sample("c", 0.0, (5.0,))) == [3.0]


def test_standardizer_keeps_constant_features_finite() -> None:
    standardizer = Standardizer.fit(
        [_sample("a", 0.0, (2.0,)), _sample("b", 0.0, (2.0,))]
    )

    assert standardizer.scales == (1.0,)
    assert standardizer.apply(_sample("c", 0.0, (2.0,))) == [0.0]


def test_standardizer_rejects_an_empty_split() -> None:
    with pytest.raises(ValueError, match="empty split"):
        Standardizer.fit([])


def test_linear_regression_recovers_a_known_linear_target(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)

    model = get_trainer(protocol.trainer)(protocol, snapshot.samples)

    predicted = model.predict(snapshot.samples)
    truth = [float(sample.target) for sample in snapshot.samples]
    assert predicted == pytest.approx(truth, abs=1e-6)


def test_linear_regression_saves_a_readable_model(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    model = get_trainer(protocol.trainer)(protocol, snapshot.samples)
    directory = tmp_path / "model"
    directory.mkdir()

    model.save(directory)

    assert isinstance(model, FittedModel)
    description = model.describe()
    assert description["trainer"] == "linear_regression"
    assert description["deterministic"] is True
    assert description["feature_columns"] == ["x1", "x2", "x3"]
    assert (directory / "model.json").is_file()


def test_linear_regression_needs_feature_columns(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    stripped = replace(protocol, features=FeatureSpec(schema="empty", columns=()))

    with pytest.raises(ValueError, match="needs features.columns"):
        get_trainer("linear_regression")(stripped, snapshot.samples)


def test_linear_regression_rejects_a_negative_penalty(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(
        tmp_path, snapshot_dir, model={"parameters": {"l2": -1.0}}
    )

    with pytest.raises(ValueError, match="l2 must not be negative"):
        get_trainer(protocol.trainer)(protocol, snapshot.samples)


def test_logistic_regression_separates_a_separable_target(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(
        tmp_path,
        snapshot_dir,
        classification=True,
        model={"parameters": {"iterations": 800}},
    )

    model = get_trainer(protocol.trainer)(protocol, snapshot.samples)

    scores = model.predict(snapshot.samples)
    for sample, score in zip(snapshot.samples, scores, strict=True):
        if str(sample.target) == "metal":
            assert score > 0.5
        else:
            assert score < 0.5


def test_logistic_regression_is_deterministic(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir, classification=True)
    trainer = get_trainer(protocol.trainer)

    first = trainer(protocol, snapshot.samples).predict(snapshot.samples)
    second = trainer(protocol, snapshot.samples).predict(snapshot.samples)

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
    protocol, snapshot = _setup(
        tmp_path, snapshot_dir, classification=True, model={"parameters": parameters}
    )

    with pytest.raises(ValueError, match=message):
        get_trainer(protocol.trainer)(protocol, snapshot.samples)


def test_logistic_regression_rejects_an_absent_positive_label(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(
        tmp_path,
        snapshot_dir,
        classification=True,
        evaluation={"positive_label": "superconductor"},
    )

    with pytest.raises(ValueError, match="absent from the train split"):
        get_trainer(protocol.trainer)(protocol, snapshot.samples)
