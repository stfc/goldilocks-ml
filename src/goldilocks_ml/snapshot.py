"""The immutable dataset snapshot contract.

A snapshot is the CGCNN-style layout this project already uses, plus a manifest
that makes it verifiable:

    snapshot/
    ├── manifest.json          # identity and a SHA-256 for every file
    ├── id_prop.csv            # sample_id, target[, group]  -- no header row
    ├── <sample_id>.cif        # optional, one per sample
    └── features.csv           # optional, precomputed model inputs

Users convert their own data into this layout. Nothing here converts for them.
"""

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
ID_PROP_NAME = "id_prop.csv"
FEATURES_NAME = "features.csv"
CAPABILITIES = frozenset({"structures", "features", "groups"})


@dataclass(frozen=True, slots=True)
class Sample:
    """One verified sample: a stable id, a target, and where its files live."""

    sample_id: str
    target: float | str
    group: str | None
    structure_path: Path | None


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A dataset snapshot verified against a protocol."""

    directory: Path
    record_id: str
    snapshot_version: str
    manifest_sha256: str
    capabilities: frozenset[str]
    features_file: str | None
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
            "sample_count": len(self.samples),
            "capabilities": sorted(self.capabilities),
        }


def _load_manifest(directory: Path) -> tuple[dict[str, Any], str]:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run 'goldilocks-train seal' on the snapshot"
        )
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


def _number(value: str, label: str, sample_id: str) -> float:
    if not value.strip():
        raise ValueError(f"{sample_id} has an empty {label}")
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{sample_id} has a non-numeric {label}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{sample_id} has a non-finite {label}: {value!r}")
    return number


def _reject_row_position_ids(sample_ids: list[str]) -> None:
    """Refuse ids that are indistinguishable from dataframe row positions.

    A split derived from row positions changes whenever the rows are reordered,
    deduplicated, or filtered, which makes the run irreproducible. Historical
    preprocessing wrote the dataframe index here; a real identifier is required
    instead.
    """
    if not all(value.lstrip("-").isdigit() for value in sample_ids):
        return
    numbers = sorted(int(value) for value in sample_ids)
    count = len(numbers)
    if numbers == list(range(count)) or numbers == list(range(1, count + 1)):
        raise ValueError(
            "sample ids are consecutive integers, which is indistinguishable from "
            "dataframe row positions; use a stable identifier such as the source "
            "database id so the split survives reordering and deduplication"
        )


def _read_id_prop(path: Path, task: str) -> list[tuple[str, float | str, str | None]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")

    parsed: list[tuple[str, float | str, str | None]] = []
    seen: set[str] = set()
    for line, row in enumerate(rows, start=1):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) not in (2, 3):
            raise ValueError(
                f"{path}:{line} has {len(row)} fields; expected sample_id,target"
                " with an optional group"
            )
        sample_id = row[0].strip()
        if not sample_id:
            raise ValueError(f"{path}:{line} has an empty sample id")
        if sample_id in seen:
            raise ValueError(f"{path}:{line} repeats sample id {sample_id}")
        seen.add(sample_id)

        target: float | str
        if task == "regression":
            target = _number(row[1], "target", sample_id)
        else:
            target = row[1].strip()
            if not target:
                raise ValueError(f"{sample_id} has an empty target")

        group = row[2].strip() if len(row) == 3 else None
        if len(row) == 3 and not group:
            raise ValueError(f"{sample_id} has an empty group")
        parsed.append((sample_id, target, group))

    if not parsed:
        raise ValueError(f"{path} has no data rows")
    _reject_row_position_ids([sample_id for sample_id, _, _ in parsed])
    return parsed


def read_sample_ids(path: Path) -> list[str]:
    """Return the sample ids in an id_prop file without interpreting targets."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    ids = [row[0].strip() for row in rows if row and row[0].strip()]
    if not ids:
        raise ValueError(f"{path} has no data rows")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} repeats a sample id")
    _reject_row_position_ids(ids)
    return ids


def load_snapshot(directory: Path, protocol: TrainingProtocol) -> Snapshot:
    """Verify a snapshot's identity, integrity, and contents against a protocol."""
    directory = directory.resolve()
    manifest, manifest_sha256 = _load_manifest(directory)

    if manifest.get("schema_version") != 1:
        raise ValueError("snapshot manifest schema_version must be 1")
    record_id = manifest.get("record_id")
    snapshot_version = manifest.get("snapshot_version")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("snapshot manifest record_id must be a non-empty string")
    if not isinstance(snapshot_version, str) or not snapshot_version:
        raise ValueError(
            "snapshot manifest snapshot_version must be a non-empty string"
        )

    pinned = protocol.dataset.pinned
    if pinned is not None:
        if manifest_sha256 != pinned.manifest_sha256:
            raise ValueError(
                f"snapshot manifest SHA-256 is {manifest_sha256}; "
                f"protocol pins {pinned.manifest_sha256}"
            )
        if record_id != pinned.record_id:
            raise ValueError(
                f"snapshot record_id is {record_id!r}; "
                f"protocol requires {pinned.record_id!r}"
            )
        if snapshot_version != pinned.snapshot_version:
            raise ValueError(
                f"snapshot version is {snapshot_version!r}; "
                f"protocol requires {pinned.snapshot_version!r}"
            )

    _verify_files(directory, manifest)

    features_file = manifest.get("features_file")
    if features_file is not None and not isinstance(features_file, str):
        raise ValueError("snapshot manifest features_file must be a string")
    structure_suffix = manifest.get("structure_suffix")
    if structure_suffix is not None and not isinstance(structure_suffix, str):
        raise ValueError("snapshot manifest structure_suffix must be a string")

    rows = _read_id_prop(directory / ID_PROP_NAME, protocol.task)
    samples: list[Sample] = []
    for sample_id, target, group in rows:
        structure_path = None
        if structure_suffix is not None:
            structure_path = directory / f"{sample_id}{structure_suffix}"
            if not structure_path.is_file():
                raise FileNotFoundError(structure_path)
        samples.append(
            Sample(
                sample_id=sample_id,
                target=target,
                group=group,
                structure_path=structure_path,
            )
        )

    capabilities = set()
    if structure_suffix is not None:
        capabilities.add("structures")
    if features_file is not None:
        capabilities.add("features")
    if all(sample.group is not None for sample in samples):
        capabilities.add("groups")

    missing = sorted(set(protocol.dataset.requires) - capabilities)
    if missing:
        raise ValueError(
            f"the protocol needs snapshot capabilities this snapshot lacks: "
            f"{', '.join(missing)}"
        )
    if protocol.split.method == "group" and "groups" not in capabilities:
        raise ValueError(
            "group splitting needs a third column in id_prop.csv naming each "
            "sample's group"
        )

    if protocol.task == "classification":
        labels = {str(sample.target) for sample in samples}
        if len(labels) < 2:
            raise ValueError("classification snapshots need at least two classes")
        positive = protocol.evaluation.positive_label
        if positive is not None and positive not in labels:
            raise ValueError(
                f"evaluation.positive_label {positive!r} is absent from the "
                f"snapshot; its classes are {', '.join(sorted(labels))}"
            )

    return Snapshot(
        directory=directory,
        record_id=record_id,
        snapshot_version=snapshot_version,
        manifest_sha256=manifest_sha256,
        capabilities=frozenset(capabilities),
        features_file=features_file,
        samples=tuple(samples),
    )
