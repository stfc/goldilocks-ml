"""Deterministic, leakage-checked split manifests keyed by stable sample IDs."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from goldilocks_ml.core.protocol import SPLIT_NAMES, TrainingProtocol
from goldilocks_ml.core.snapshot import Sample, Snapshot

SPLITS_HEADER = ("sample_id", "split")


def _units(snapshot: Snapshot, protocol: TrainingProtocol) -> dict[str, list[Sample]]:
    """Group samples into the indivisible units a split may allocate."""
    units: dict[str, list[Sample]] = {}
    for sample in snapshot.samples:
        if protocol.split.method == "group":
            if sample.group is None:
                raise ValueError(f"{sample.sample_id} has no group")
            key = sample.group
        else:
            key = sample.sample_id
        units.setdefault(key, []).append(sample)
    return units


def _stratum(samples: list[Sample], stratify: bool) -> str:
    """Return the stratum label for one unit, or a single shared stratum."""
    if not stratify:
        return ""
    counts: dict[str, int] = {}
    for sample in samples:
        label = str(sample.target)
        counts[label] = counts.get(label, 0) + 1
    return min(counts, key=lambda label: (-counts[label], label))


def _allocate(
    keys: list[str],
    sizes: dict[str, int],
    ratios: tuple[tuple[str, float], ...],
) -> dict[str, str]:
    """Assign shuffled units to splits by largest remaining sample deficit."""
    total = sum(sizes[key] for key in keys)
    targets = {name: ratio * total for name, ratio in ratios if ratio > 0}
    assigned = {name: 0 for name in targets}
    placement: dict[str, str] = {}
    order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    for key in keys:
        name = min(
            targets,
            key=lambda candidate: (
                assigned[candidate] - targets[candidate],
                order[candidate],
            ),
        )
        placement[key] = name
        assigned[name] += sizes[key]
    return placement


def assign_splits(snapshot: Snapshot, protocol: TrainingProtocol) -> dict[str, str]:
    """Derive a deterministic sample-to-split assignment from stable IDs."""
    units = _units(snapshot, protocol)
    sizes = {key: len(samples) for key, samples in units.items()}
    strata: dict[str, list[str]] = {}
    for key, samples in units.items():
        strata.setdefault(_stratum(samples, protocol.split.stratify), []).append(key)

    assignment: dict[str, str] = {}
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        random.Random(f"{protocol.split.seed}:{stratum}").shuffle(keys)
        for key, name in _allocate(keys, sizes, protocol.split.ratios).items():
            for sample in units[key]:
                assignment[sample.sample_id] = name

    check_assignment(assignment, snapshot, protocol)
    return assignment


def check_assignment(
    assignment: dict[str, str], snapshot: Snapshot, protocol: TrainingProtocol
) -> None:
    """Reject incomplete coverage, unknown splits, and group leakage."""
    valid = {name for name, ratio in protocol.split.ratios if ratio > 0}
    unknown = sorted(set(assignment.values()) - valid)
    if unknown:
        raise ValueError(
            f"split manifest uses unrequested split(s): {', '.join(unknown)}"
        )

    expected = set(snapshot.sample_ids)
    actual = set(assignment)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            f"split manifest is missing {len(missing)} sample(s), "
            f"starting with {missing[0]}"
        )
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(
            f"split manifest has {len(extra)} unknown sample(s), "
            f"starting with {extra[0]}"
        )

    empty = sorted(valid - set(assignment.values()))
    if empty:
        raise ValueError(f"split(s) received no samples: {', '.join(empty)}")

    if protocol.split.method != "group":
        return
    groups: dict[str, set[str]] = {}
    for sample in snapshot.samples:
        if sample.group is None:
            raise ValueError(f"{sample.sample_id} has no group")
        groups.setdefault(sample.group, set()).add(assignment[sample.sample_id])
    leaked = sorted(group for group, names in groups.items() if len(names) > 1)
    if leaked:
        raise ValueError(f"group(s) span more than one split: {', '.join(leaked[:5])}")


def write_splits(path: Path, assignment: dict[str, str]) -> None:
    """Write a split manifest sorted by sample id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SPLITS_HEADER)
        for sample_id in sorted(assignment):
            writer.writerow([sample_id, assignment[sample_id]])


def read_splits(
    path: Path, snapshot: Snapshot, protocol: TrainingProtocol
) -> dict[str, str]:
    """Reload and revalidate an existing split manifest."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"{path} is empty") from error
        if header != SPLITS_HEADER:
            raise ValueError(f"{path} header must be {','.join(SPLITS_HEADER)}")
        assignment: dict[str, str] = {}
        for line, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"{path}:{line} must have two fields")
            sample_id, name = row[0].strip(), row[1].strip()
            if sample_id in assignment:
                raise ValueError(f"{path}:{line} repeats sample {sample_id}")
            assignment[sample_id] = name
    check_assignment(assignment, snapshot, protocol)
    return assignment


def partition(
    assignment: dict[str, str], snapshot: Snapshot
) -> dict[str, tuple[Sample, ...]]:
    """Return samples grouped by split, in snapshot order."""
    parts: dict[str, list[Sample]] = {name: [] for name in SPLIT_NAMES}
    for sample in snapshot.samples:
        parts[assignment[sample.sample_id]].append(sample)
    return {name: tuple(samples) for name, samples in parts.items() if samples}
