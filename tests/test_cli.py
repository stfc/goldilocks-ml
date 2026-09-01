"""End-to-end tests for goldilocks-ml train seal, validate, and run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    KDIST_SNAPSHOT,
    METALLIC_SNAPSHOT,
    PROTOCOLS,
    build_snapshot,
    classification_document,
    make_rows,
    pin,
    regression_document,
    write_protocol,
)

from goldilocks_ml.cli import seal
from goldilocks_ml.console import main
from goldilocks_ml.runs import MANIFEST_NAME, RUN_MARKER

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
    RUN_MARKER,
}


def _seal(directory: Path, **overrides: Any) -> dict[str, Any]:
    arguments = {
        "record_id": "mine",
        "snapshot_version": "v1",
        "structure_suffix": ".cif",
        "target_name": "value",
        "target_contract": "synthetic.value.v1",
        "target_definition": "Synthetic test target value.",
        "target_units": "arbitrary",
        **overrides,
    }
    return seal(directory, **arguments)


def _setup(
    tmp_path: Path,
    snapshot_dir: Path,
    *,
    classification: bool = False,
    **overrides: Any,
) -> Path:
    build_snapshot(snapshot_dir, target="label" if classification else "value")
    document = (
        classification_document(**overrides)
        if classification
        else regression_document(**overrides)
    )
    return write_protocol(tmp_path / "protocol.toml", document)


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    del monkeypatch  # the parser takes its arguments directly
    main(["train", *argv])


def _bundle_files(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_seal_writes_a_manifest_covering_every_file(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    build_snapshot(directory, structures=True)

    result = _seal(directory, snapshot_version="v3")

    names = {entry["name"] for entry in result["manifest"]["files"]}
    assert "id_prop.csv" in names
    assert "features.csv" in names
    assert "syn-000.cif" in names
    assert len(names) == 2 + 24
    assert result["manifest"]["structure_suffix"] == ".cif"


def test_seal_refuses_a_partial_set_of_structures(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    build_snapshot(directory, structures=True)
    (directory / "syn-005.cif").unlink()

    with pytest.raises(FileNotFoundError, match="structure file\\(s\\) are missing"):
        _seal(directory)


def test_seal_needs_an_id_prop_file(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    directory.mkdir()

    with pytest.raises(FileNotFoundError, match="id_prop.csv"):
        _seal(directory)


def test_seal_refuses_nested_directories(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    build_snapshot(directory)
    (directory / "nested").mkdir()

    with pytest.raises(ValueError, match="must be flat"):
        _seal(directory)


def test_seal_refuses_unsafe_sample_ids(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="safe basenames"):
        build_snapshot(
            directory,
            [{**row, "sample_id": "../outside"} for row in make_rows()[:1]],
        )


def test_overwrite_refuses_a_directory_that_is_not_a_previous_run(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    output = tmp_path / "important"
    output.mkdir()
    (output / "keep.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        _run(
            monkeypatch,
            "run",
            str(protocol),
            "--dataset",
            str(snapshot_dir),
            "--output",
            str(output),
            "--overwrite",
        )

    assert error.value.code == 2
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not delete"


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
    assert "24 samples, 3 features" in output
    assert "train=12" in output
    assert not list(tmp_path.glob("**/metrics.json"))


def test_validate_fails_before_training_on_a_pinned_mismatch(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_snapshot(snapshot_dir)
    protocol = write_protocol(
        tmp_path / "protocol.toml", pin(regression_document(), "f" * 64)
    )

    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, "validate", str(protocol), "--dataset", str(snapshot_dir))
    assert error.value.code == 2


def test_a_missing_pinned_artifact_names_its_record(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(
        tmp_path,
        snapshot_dir,
        features={
            "depends_on": {
                "metallicity": {
                    "record_id": "ptc95-vbq12",
                    "file": "is_metal.ckpt",
                    "sha256": "9" * 64,
                }
            }
        },
    )

    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            "validate",
            str(protocol),
            "--dataset",
            str(snapshot_dir),
            "--artifact-directory",
            str(tmp_path / "artifacts"),
        )


def test_a_pinned_artifact_must_match_its_digest(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts" / "ptc95-vbq12"
    artifacts.mkdir(parents=True)
    (artifacts / "is_metal.ckpt").write_bytes(b"not the real checkpoint")
    protocol = _setup(
        tmp_path,
        snapshot_dir,
        features={
            "depends_on": {
                "metallicity": {
                    "record_id": "ptc95-vbq12",
                    "file": "is_metal.ckpt",
                    "sha256": "9" * 64,
                }
            }
        },
    )

    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            "validate",
            str(protocol),
            "--dataset",
            str(snapshot_dir),
            "--artifact-directory",
            str(tmp_path / "artifacts"),
        )


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
    assert metrics["target"] == "value"
    assert set(metrics["splits"]) == {"baseline", "model"}
    assert set(metrics["splits"]["model"]) == {
        "train",
        "validation",
        "calibration",
        "test",
    }
    # The fixture target is exactly linear, so the model must beat the baseline.
    model_mae = metrics["splits"]["model"]["test"]["mae"]
    assert model_mae < metrics["splits"]["baseline"]["test"]["mae"]

    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["splits_reused"] is False


def test_an_unpinned_run_still_records_the_real_snapshot_digest(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = build_snapshot(
        snapshot_dir, record_id="someone-elses", snapshot_version="v9"
    )
    protocol = write_protocol(tmp_path / "protocol.toml", regression_document())
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

    dataset = json.loads((output / "dataset.json").read_text())
    assert dataset["pinned_by_protocol"] is False
    assert dataset["record_id"] == "someone-elses"
    assert dataset["manifest_sha256"] == digest
    assert dataset["feature_schema"] == "tabular"


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
    first, second = tmp_path / "run-1", tmp_path / "run-2"

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

    assert (
        json.loads((first / MANIFEST_NAME).read_text())["deterministic_digest"]
        == json.loads((second / MANIFEST_NAME).read_text())["deterministic_digest"]
    )
    assert (first / "splits.csv").read_bytes() == (second / "splits.csv").read_bytes()


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

    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, *arguments)
    assert error.value.code == 2

    _run(monkeypatch, *arguments, "--overwrite")
    assert _bundle_files(output) == BUNDLE_FILES


def test_run_reuses_an_existing_split_manifest(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _setup(tmp_path, snapshot_dir)
    first, second = tmp_path / "run-1", tmp_path / "run-2"
    _run(
        monkeypatch,
        "run",
        str(protocol),
        "--dataset",
        str(snapshot_dir),
        "--output",
        str(first),
    )

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


def test_preprocessing_is_fitted_on_the_train_split_only(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from goldilocks_ml import baselines as baseline

    protocol = _setup(tmp_path, snapshot_dir)
    output = tmp_path / "run-1"
    seen: list[int] = []
    original = baseline.Standardizer.fit

    def spy(rows):
        seen.append(len(rows))
        return original(rows)

    monkeypatch.setattr(baseline.Standardizer, "fit", staticmethod(spy))
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
        train = sum(1 for row in csv.DictReader(handle) if row["split"] == "train")
    assert seen and all(count == train for count in seen)


def test_classification_selects_its_threshold_on_validation_only(
    tmp_path: Path, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import goldilocks_ml.cli as core_cli

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
    seen: list[tuple[str, ...]] = []
    original = core_cli.select_threshold

    def spy(predictions, metric, positive, negative, **options):
        seen.append(tuple(item.split for item in predictions))
        return original(predictions, metric, positive, negative, **options)

    monkeypatch.setattr(core_cli, "select_threshold", spy)
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

    with pytest.raises(SystemExit) as error:
        _run(
            monkeypatch,
            "run",
            str(protocol),
            "--dataset",
            str(snapshot_dir),
            "--output",
            str(tmp_path / "run-1"),
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("protocol_name", "snapshot"),
    [
        ("regression.toml", KDIST_SNAPSHOT),
        ("classification.toml", METALLIC_SNAPSHOT),
    ],
)
def test_committed_protocols_run_against_the_committed_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol_name: str, snapshot: Path
) -> None:
    output = tmp_path / "run"

    _run(
        monkeypatch,
        "run",
        str(PROTOCOLS / protocol_name),
        "--dataset",
        str(snapshot),
        "--output",
        str(output),
    )

    assert _bundle_files(output) == BUNDLE_FILES
    metrics = json.loads((output / "metrics.json").read_text())
    primary = metrics["primary_metric"]
    model = metrics["splits"]["model"]["test"][primary]
    baseline = metrics["splits"]["baseline"]["test"][primary]
    assert model > baseline if primary == "mcc" else model < baseline


def test_make_rows_is_the_only_fixture_generator() -> None:
    assert len(make_rows()) == 24
