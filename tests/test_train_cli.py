"""End-to-end tests for goldilocks-train validate and run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    FIXTURE_ROOT,
    PROTOCOL_ROOT,
    build_snapshot,
    classification_document,
    regression_document,
    write_protocol,
)

from goldilocks_ml import trainers
from goldilocks_ml.runs import MANIFEST_NAME
from goldilocks_ml.train_cli import cli

BUNDLE_FILES = {
    "dataset.json",
    "environment.json",
    MANIFEST_NAME,
    "metrics.json",
    "model/model.json",
    "predictions.csv",
    "protocol.toml",
    "run.json",
    "splits.csv",
}


def _setup(
    tmp_path: Path,
    snapshot_dir: Path,
    *,
    classification: bool = False,
    **overrides: Any,
) -> Path:
    digest = build_snapshot(snapshot_dir)
    document = (
        classification_document(digest, **overrides)
        if classification
        else regression_document(digest, **overrides)
    )
    return write_protocol(tmp_path / "protocol.toml", document)


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["goldilocks-train", *argv])
    cli()


def _bundle_files(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_validate_reports_the_split_without_training(
    tmp_path: Path,
    snapshot_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)

    _run(monkeypatch, "validate", str(protocol), "--dataset", str(snapshot_dir))

    output = capsys.readouterr().out
    assert "Valid protocol synthetic-regression-v1" in output
    assert "24 samples" in output
    assert "train=12" in output
    assert not list(tmp_path.glob("**/metrics.json"))


def test_validate_fails_before_training_on_a_dataset_mismatch(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_snapshot(snapshot_dir)
    protocol = write_protocol(tmp_path / "protocol.toml", regression_document("f" * 64))

    with pytest.raises(ValueError, match="protocol pins"):
        _run(monkeypatch, "validate", str(protocol), "--dataset", str(snapshot_dir))


def test_run_writes_the_documented_bundle(
    tmp_path: Path,
    snapshot_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    output = tmp_path / "run-1"

    _run(
        monkeypatch,
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(output),
    )

    assert _bundle_files(output) == BUNDLE_FILES
    assert "Wrote run bundle" in capsys.readouterr().out

    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["task"] == "regression"
    assert metrics["primary_metric"] == "mae"
    assert set(metrics["splits"]) == {"baseline", "model"}
    assert set(metrics["splits"]["model"]) == {
        "train",
        "validation",
        "calibration",
        "test",
    }
    # The fixture target is exactly linear, so the model must beat the baseline.
    assert (
        metrics["splits"]["model"]["test"]["mae"]
        < metrics["splits"]["baseline"]["test"]["mae"]
    )

    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["protocol_id"] == "synthetic-regression-v1"
    assert run["splits_reused"] is False

    dataset = json.loads((output / "dataset.json").read_text())
    assert dataset["record_id"] == "synthetic"
    assert dataset["sample_count"] == 24


def test_run_records_every_prediction_once_per_source(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    output = tmp_path / "run-1"

    _run(
        monkeypatch,
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(output),
    )

    with (output / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48
    assert {row["source"] for row in rows} == {"baseline", "model"}
    assert len({row["sample_id"] for row in rows}) == 24


def test_run_is_reproducible(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    first = tmp_path / "run-1"
    second = tmp_path / "run-2"

    for output in (first, second):
        _run(
            monkeypatch,
            "run",
            str(protocol),
            "--dataset",
            str(snapshot_dir),
            "--output",
            str(output),
        )

    first_manifest = json.loads((first / MANIFEST_NAME).read_text())
    second_manifest = json.loads((second / MANIFEST_NAME).read_text())
    assert (
        first_manifest["deterministic_digest"]
        == second_manifest["deterministic_digest"]
    )
    assert (first / "splits.csv").read_bytes() == (second / "splits.csv").read_bytes()
    assert (first / "metrics.json").read_text() == (second / "metrics.json").read_text()


def test_run_refuses_to_overwrite_without_the_flag(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    output = tmp_path / "run-1"
    arguments = (
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(output),
    )

    _run(monkeypatch, *arguments)

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        _run(monkeypatch, *arguments)

    _run(monkeypatch, *arguments, "--overwrite")
    assert _bundle_files(output) == BUNDLE_FILES


def test_run_reuses_an_existing_split_manifest(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    first = tmp_path / "run-1"
    _run(
        monkeypatch,
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(first),
    )

    second = tmp_path / "run-2"
    _run(
        monkeypatch,
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(second),
        "--splits",
        str(first / "splits.csv"),
    )

    assert (second / "splits.csv").read_bytes() == (first / "splits.csv").read_bytes()
    assert json.loads((second / "run.json").read_text())["splits_reused"] is True


def test_run_rejects_a_split_manifest_from_another_dataset(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    splits = tmp_path / "splits.csv"
    splits.write_text("sample_id,split\nother-1,train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 24 sample"):
        _run(
            monkeypatch,
            "run",
            str(protocol),
            "--dataset",
            str(snapshot_dir),
            "--output",
            str(tmp_path / "run-1"),
            "--splits",
            str(splits),
        )


def test_preprocessing_is_fitted_on_the_train_split_only(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    output = tmp_path / "run-1"
    seen: list[tuple[str, ...]] = []
    original = trainers.Standardizer.fit

    def spy(samples):
        seen.append(tuple(sample.sample_id for sample in samples))
        return original(samples)

    monkeypatch.setattr(trainers.Standardizer, "fit", staticmethod(spy))
    _run(
        monkeypatch,
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(output),
    )

    with (output / "splits.csv").open(encoding="utf-8", newline="") as handle:
        train = {
            row["sample_id"]
            for row in csv.DictReader(handle)
            if row["split"] == "train"
        }
    assert seen, "the trainer never fitted preprocessing"
    for fitted_on in seen:
        assert set(fitted_on) == train


def test_classification_run_selects_its_threshold_on_validation_only(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(
        tmp_path,
        snapshot_dir,
        classification=True,
        evaluation={
            "metrics": ["accuracy", "mcc", "roc_auc"],
            "threshold_metric": "mcc",
        },
    )
    output = tmp_path / "run-1"
    import goldilocks_ml.train_cli as train_cli

    seen: list[tuple[str, ...]] = []
    original = train_cli.select_threshold

    def spy(predictions, metric, positive, negative):
        seen.append(tuple(item.split for item in predictions))
        return original(predictions, metric, positive, negative)

    monkeypatch.setattr(train_cli, "select_threshold", spy)
    _run(
        monkeypatch,
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(output),
    )

    assert seen and all(set(splits) == {"validation"} for splits in seen)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["decision_threshold"]["selected_on"] == "validation"
    assert metrics["decision_threshold"]["metric"] == "mcc"
    assert metrics["positive_label"] == "metal"
    assert "confusion_matrix" in metrics["splits"]["model"]["test"]


def test_threshold_selection_requires_a_validation_split(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(
        tmp_path,
        snapshot_dir,
        classification=True,
        split={"train": 0.7, "validation": 0.0, "calibration": 0.1, "test": 0.2},
        evaluation={
            "metrics": ["accuracy", "mcc", "roc_auc"],
            "threshold_metric": "mcc",
        },
    )

    with pytest.raises(ValueError, match="requires a non-empty validation split"):
        _run(
            monkeypatch,
            "run",
            str(protocol),
            "--dataset",
            str(snapshot_dir),
            "--output",
            str(tmp_path / "run-1"),
        )


@pytest.mark.parametrize(
    "name", ["tabular_regression.toml", "tabular_classification.toml"]
)
def test_committed_protocols_run_against_the_committed_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    output = tmp_path / "run"

    _run(
        monkeypatch,
        "run",
        str(PROTOCOL_ROOT / name),
        "--dataset",
        str(FIXTURE_ROOT),
        "--output",
        str(output),
    )

    assert _bundle_files(output) == BUNDLE_FILES
    metrics = json.loads((output / "metrics.json").read_text())
    primary = metrics["primary_metric"]
    model = metrics["splits"]["model"]["test"][primary]
    baseline = metrics["splits"]["baseline"]["test"][primary]
    assert model > baseline if primary == "mcc" else model < baseline
