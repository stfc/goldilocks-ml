"""Tests for run bundle serialisation and checksums."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_snapshot, regression_document, write_protocol

from goldilocks_ml.evaluation import Prediction
from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.protocol import load_protocol
from goldilocks_ml.runs import (
    NON_DETERMINISTIC_FILES,
    RUN_MARKER,
    dumps_toml,
    environment_record,
    prepare_directory,
    resolved_document,
    write_manifest,
    write_predictions,
)


def _protocol(tmp_path: Path, snapshot_dir: Path, **overrides):
    build_snapshot(snapshot_dir)
    return load_protocol(
        write_protocol(tmp_path / "protocol.toml", regression_document(**overrides))
    )


def test_dumps_toml_emits_scalars_arrays_and_nested_tables() -> None:
    text = dumps_toml(
        {
            "id": "x",
            "count": 2,
            "ratio": 0.25,
            "flag": True,
            "table": {"names": ["a", "b"], "nested": {"depth": 3}},
        }
    )

    assert text.splitlines()[:4] == [
        'id = "x"',
        "count = 2",
        "ratio = 0.25",
        "flag = true",
    ]
    assert "[table]" in text
    assert 'names = ["a", "b"]' in text
    assert "[table.nested]" in text


def test_dumps_toml_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError, match="cannot serialise"):
        dumps_toml({"when": object()})


def test_resolved_document_round_trips_through_the_loader(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol = _protocol(
        tmp_path,
        snapshot_dir,
        split={"method": "group"},
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
    path = tmp_path / "resolved.toml"

    path.write_text(dumps_toml(resolved_document(protocol)), encoding="utf-8")
    reloaded = load_protocol(path)

    assert resolved_document(reloaded) == resolved_document(protocol)


def test_resolved_document_makes_defaults_explicit(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol = _protocol(tmp_path, snapshot_dir)

    document = resolved_document(protocol)

    assert document["split"]["stratify"] is False
    assert document["dataset"]["requires"] == ["features"]
    assert "record_id" not in document["dataset"]
    assert "depends_on" not in document["features"]
    assert "threshold_metric" not in document["evaluation"]


def test_environment_record_names_the_interpreter_and_lock() -> None:
    record = environment_record()

    assert record["python_version"].startswith("3.")
    assert record["packages"]["goldilocks-ml"]
    assert record["uv_lock_sha256"] is None or len(record["uv_lock_sha256"]) == 64


def test_prepare_directory_refuses_to_clobber_a_previous_run(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        prepare_directory(directory, overwrite=False)


def test_prepare_directory_replaces_a_previous_run_when_asked(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    (directory / "model").mkdir(parents=True)
    (directory / "model" / "stale.json").write_text("{}", encoding="utf-8")
    (directory / RUN_MARKER).write_text("goldilocks-ml\n", encoding="utf-8")

    prepared = prepare_directory(directory, overwrite=True)

    assert (prepared / "model").is_dir()
    assert not (prepared / "model" / "stale.json").exists()
    assert (prepared / RUN_MARKER).is_file()


def test_write_predictions_sorts_by_source_then_sample(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"

    write_predictions(
        path,
        {
            "model": [
                Prediction("s002", 2.0, 2.1, None, "test"),
                Prediction("s001", 1.0, 1.1, None, "train"),
            ],
            "baseline": [Prediction("s001", 1.0, 0.0, 0.25, "train")],
        },
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sample_id,split,source,truth,prediction,score,lower,upper"
    assert lines[1] == "s001,train,baseline,1.0,0.0,0.25,,"
    assert lines[2] == "s001,train,model,1.0,1.1,,,"
    assert lines[3] == "s002,test,model,2.0,2.1,,,"


def test_write_manifest_covers_every_file_but_itself(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model" / "model.json").write_text("{}", encoding="utf-8")

    manifest = write_manifest(tmp_path)

    names = [entry["name"] for entry in manifest["files"]]
    assert names == ["metrics.json", "model/model.json"]
    assert manifest["files"][0]["sha256"] == sha256_file(tmp_path / "metrics.json")
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_deterministic_digest_ignores_provenance_only_files(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text('{"mae": 1}', encoding="utf-8")
    (tmp_path / "run.json").write_text('{"started_at": "a"}', encoding="utf-8")
    first = write_manifest(tmp_path)

    (tmp_path / "run.json").write_text('{"started_at": "b"}', encoding="utf-8")
    second = write_manifest(tmp_path)

    assert first["deterministic_digest"] == second["deterministic_digest"]
    assert "run.json" in NON_DETERMINISTIC_FILES


def test_deterministic_digest_changes_with_scientific_content(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text('{"mae": 1}', encoding="utf-8")
    first = write_manifest(tmp_path)

    (tmp_path / "metrics.json").write_text('{"mae": 2}', encoding="utf-8")
    second = write_manifest(tmp_path)

    assert first["deterministic_digest"] != second["deterministic_digest"]
