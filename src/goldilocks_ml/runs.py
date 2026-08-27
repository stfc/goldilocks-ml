"""Write the self-describing run bundle that a colleague can audit or publish."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from goldilocks_ml.evaluation import Prediction
from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.protocol import TrainingProtocol

MANIFEST_NAME = "manifest.json"
PREDICTIONS_HEADER = ("sample_id", "split", "source", "truth", "prediction", "score")

# Provenance that legitimately differs between two runs of the same protocol.
NON_DETERMINISTIC_FILES = frozenset({"run.json", "environment.json", MANIFEST_NAME})


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"cannot serialise {type(value).__name__} to TOML")


def _value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    return _scalar(value)


def dumps_toml(document: dict[str, Any]) -> str:
    """Serialise the restricted protocol document shape back to TOML."""
    lines: list[str] = []
    for key, value in document.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_value(value)}")
    for key, value in document.items():
        if not isinstance(value, dict):
            continue
        lines.extend(("", f"[{key}]"))
        for inner_key, inner_value in value.items():
            if not isinstance(inner_value, dict):
                lines.append(f"{inner_key} = {_value(inner_value)}")
        for inner_key, inner_value in value.items():
            if isinstance(inner_value, dict):
                lines.extend(("", f"[{key}.{inner_key}]"))
                for leaf_key, leaf_value in inner_value.items():
                    lines.append(f"{leaf_key} = {_value(leaf_value)}")
    return "\n".join(lines) + "\n"


def resolved_document(protocol: TrainingProtocol) -> dict[str, Any]:
    """Return the protocol with every default made explicit."""
    evaluation: dict[str, Any] = {
        "primary_metric": protocol.evaluation.primary_metric,
        "metrics": list(protocol.evaluation.metrics),
        "baseline": protocol.evaluation.baseline,
    }
    if protocol.evaluation.threshold_metric is not None:
        evaluation["threshold_metric"] = protocol.evaluation.threshold_metric
    if protocol.evaluation.positive_label is not None:
        evaluation["positive_label"] = protocol.evaluation.positive_label
    split: dict[str, Any] = {
        "method": protocol.split.method,
        "train": protocol.split.train,
        "validation": protocol.split.validation,
        "calibration": protocol.split.calibration,
        "test": protocol.split.test,
        "seed": protocol.split.seed,
        "stratify": protocol.split.stratify,
    }
    if protocol.split.group_column is not None:
        split["group_column"] = protocol.split.group_column
    return {
        "schema_version": protocol.schema_version,
        "id": protocol.id,
        "task": protocol.task,
        "trainer": protocol.trainer,
        "dataset": {
            "record_id": protocol.dataset.record_id,
            "snapshot_version": protocol.dataset.snapshot_version,
            "manifest_sha256": protocol.dataset.manifest_sha256,
            "sample_id": protocol.dataset.sample_id,
            "target": protocol.dataset.target,
        },
        "split": split,
        "features": {
            "schema": protocol.features.schema,
            "columns": list(protocol.features.columns),
        },
        "model": {"seed": protocol.model.seed, "parameters": protocol.model.parameters},
        "evaluation": evaluation,
    }


def _git_commit(start: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def environment_record() -> dict[str, Any]:
    """Return the interpreter, package, and hardware facts behind a run."""
    packages: dict[str, str] = {}
    for name in ("goldilocks-ml", "data-collections-api"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
    }


def write_predictions(path: Path, predictions: dict[str, list[Prediction]]) -> None:
    """Write every baseline and model prediction, sorted for stable output."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTIONS_HEADER)
        for source in sorted(predictions):
            for item in sorted(predictions[source], key=lambda row: row.sample_id):
                writer.writerow(
                    [
                        item.sample_id,
                        item.split,
                        source,
                        item.truth,
                        item.prediction,
                        "" if item.score is None else repr(item.score),
                    ]
                )


def write_manifest(directory: Path) -> dict[str, Any]:
    """Record the size and digest of every bundle file, plus a stable identity."""
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        name = path.relative_to(directory).as_posix()
        entries.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for entry in entries:
        if entry["name"] in NON_DETERMINISTIC_FILES:
            continue
        digest.update(f"{entry['name']}:{entry['sha256']}\n".encode())
    manifest = {
        "schema_version": 1,
        "files": entries,
        "deterministic_digest": digest.hexdigest(),
        "non_deterministic_files": sorted(NON_DETERMINISTIC_FILES - {MANIFEST_NAME}),
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def write_json(path: Path, document: Any) -> None:
    """Write one bundle document with stable key order."""
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_record(
    run_id: str,
    protocol: TrainingProtocol,
    *,
    status: str,
    started_at: datetime,
    splits_reused: bool,
) -> dict[str, Any]:
    """Return the run's own provenance document."""
    return {
        "run_id": run_id,
        "protocol_id": protocol.id,
        "protocol_source": protocol.source.name,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(protocol.source.parent),
        "splits_reused": splits_reused,
    }


def prepare_directory(directory: Path, *, overwrite: bool) -> Path:
    """Create an empty bundle directory, refusing to clobber a previous run."""
    directory = directory.resolve()
    if directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"{directory} already exists; pass --overwrite to replace it"
            )
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        for path in sorted(directory.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model").mkdir()
    return directory


def bundle_files(directory: Path) -> Sequence[str]:
    """Return every file in a bundle, relative and sorted."""
    return tuple(
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    )
