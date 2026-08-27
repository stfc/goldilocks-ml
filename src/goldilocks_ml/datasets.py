"""Verify and load immutable dataset snapshots produced by goldilocks-data."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldilocks_ml.hashing import is_sha256, sha256_file
from goldilocks_ml.protocol import TrainingProtocol

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class Sample:
    """One verified row of a dataset snapshot."""

    sample_id: str
    target: float | str
    group: str | None
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Snapshot:
    """An immutable dataset snapshot verified against a protocol."""

    directory: Path
    record_id: str
    snapshot_version: str
    manifest_sha256: str
    data_file: str
    feature_columns: tuple[str, ...]
    samples: tuple[Sample, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return every sample identifier in snapshot order."""
        return tuple(sample.sample_id for sample in self.samples)

    def by_id(self) -> dict[str, Sample]:
        """Return samples keyed by their stable identifier."""
        return {sample.sample_id: sample for sample in self.samples}

    def identity(self) -> dict[str, Any]:
        """Return the provenance record written into a run bundle."""
        return {
            "record_id": self.record_id,
            "snapshot_version": self.snapshot_version,
            "manifest_sha256": self.manifest_sha256,
            "data_file": self.data_file,
            "sample_count": len(self.samples),
            "feature_columns": list(self.feature_columns),
        }


def _load_manifest(directory: Path) -> tuple[dict[str, Any], str]:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    digest = sha256_file(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return manifest, digest


def _verify_files(directory: Path, manifest: dict[str, Any]) -> None:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("snapshot manifest files must be a non-empty list")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each snapshot manifest file must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("snapshot file names must be non-empty basenames")
        if name in seen:
            raise ValueError(f"duplicate snapshot file entry: {name}")
        seen.add(name)
        size_bytes = entry.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
            raise ValueError(f"invalid size for snapshot file {name}")
        sha256 = entry.get("sha256")
        if not is_sha256(sha256):
            raise ValueError(f"invalid SHA-256 for snapshot file {name}")
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != size_bytes:
            raise ValueError(f"{name} has {actual_size} bytes; expected {size_bytes}")
        actual_digest = sha256_file(path)
        if actual_digest != sha256:
            raise ValueError(f"{name} SHA-256 is {actual_digest}; expected {sha256}")


def _read_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        columns = tuple(reader.fieldnames)
        rows = [dict(row) for row in reader]
    if len(columns) != len(set(columns)):
        raise ValueError(f"{path} has duplicate column names")
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return columns, rows


def _number(value: str | None, column: str, sample_id: str) -> float:
    if value is None or not value.strip():
        raise ValueError(f"{sample_id} has an empty {column}")
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"{sample_id} has a non-numeric {column}: {value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"{sample_id} has a non-finite {column}: {value!r}")
    return number


def load_snapshot(directory: Path, protocol: TrainingProtocol) -> Snapshot:
    """Verify a snapshot's identity, integrity, and columns against a protocol."""
    directory = directory.resolve()
    manifest, manifest_sha256 = _load_manifest(directory)

    if manifest.get("schema_version") != 1:
        raise ValueError("snapshot manifest schema_version must be 1")
    if manifest_sha256 != protocol.dataset.manifest_sha256:
        raise ValueError(
            f"snapshot manifest SHA-256 is {manifest_sha256}; "
            f"protocol pins {protocol.dataset.manifest_sha256}"
        )
    record_id = manifest.get("record_id")
    if record_id != protocol.dataset.record_id:
        raise ValueError(
            f"snapshot record_id is {record_id!r}; "
            f"protocol requires {protocol.dataset.record_id!r}"
        )
    snapshot_version = manifest.get("snapshot_version")
    if snapshot_version != protocol.dataset.snapshot_version:
        raise ValueError(
            f"snapshot version is {snapshot_version!r}; "
            f"protocol requires {protocol.dataset.snapshot_version!r}"
        )
    data_file = manifest.get("data_file")
    if not isinstance(data_file, str) or not data_file:
        raise ValueError("snapshot manifest data_file must be a non-empty string")

    _verify_files(directory, manifest)

    columns, rows = _read_rows(directory / data_file)
    missing = [column for column in protocol.required_columns if column not in columns]
    if missing:
        raise ValueError(f"snapshot is missing column(s): {', '.join(missing)}")

    sample_id_column = protocol.dataset.sample_id
    target_column = protocol.dataset.target
    group_column = protocol.split.group_column
    feature_columns = protocol.features.columns

    samples: list[Sample] = []
    seen_ids: set[str] = set()
    for row in rows:
        sample_id = (row.get(sample_id_column) or "").strip()
        if not sample_id:
            raise ValueError(f"snapshot has an empty {sample_id_column}")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)

        target: float | str
        if protocol.task == "regression":
            target = _number(row.get(target_column), target_column, sample_id)
        else:
            label = (row.get(target_column) or "").strip()
            if not label:
                raise ValueError(f"{sample_id} has an empty {target_column}")
            target = label

        group: str | None = None
        if group_column is not None:
            group = (row.get(group_column) or "").strip()
            if not group:
                raise ValueError(f"{sample_id} has an empty {group_column}")

        features = tuple(
            _number(row.get(column), column, sample_id) for column in feature_columns
        )
        samples.append(
            Sample(sample_id=sample_id, target=target, group=group, features=features)
        )

    if protocol.task == "classification":
        labels = {str(sample.target) for sample in samples}
        if len(labels) < 2:
            raise ValueError("classification snapshots need at least two classes")

    return Snapshot(
        directory=directory,
        record_id=protocol.dataset.record_id,
        snapshot_version=protocol.dataset.snapshot_version,
        manifest_sha256=manifest_sha256,
        data_file=data_file,
        feature_columns=feature_columns,
        samples=tuple(samples),
    )
