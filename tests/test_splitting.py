"""Tests for deterministic, leakage-free split manifests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    build_snapshot,
    classification_document,
    make_rows,
    regression_document,
    write_protocol,
)

from goldilocks_ml.protocol import load_protocol
from goldilocks_ml.snapshot import load_snapshot
from goldilocks_ml.splitting import (
    assign_splits,
    check_assignment,
    partition,
    read_splits,
    write_splits,
)

GROUP_SPLIT = {"method": "group"}


def _setup(
    tmp_path: Path,
    snapshot_dir: Path,
    *,
    rows: list[dict[str, str]] | None = None,
    classification: bool = False,
    **overrides: Any,
):
    build_snapshot(snapshot_dir, rows, target="label" if classification else "value")
    document = (
        classification_document(**overrides)
        if classification
        else regression_document(**overrides)
    )
    protocol = load_protocol(write_protocol(tmp_path / "protocol.toml", document))
    return protocol, load_snapshot(snapshot_dir, protocol)


def test_assign_splits_covers_every_sample_exactly_once(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)

    assignment = assign_splits(snapshot, protocol)

    assert set(assignment) == set(snapshot.sample_ids)
    assert set(assignment.values()) == {"train", "validation", "calibration", "test"}


def test_assign_splits_is_deterministic(tmp_path: Path, snapshot_dir: Path) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)

    assert assign_splits(snapshot, protocol) == assign_splits(snapshot, protocol)


def test_assign_splits_ignores_row_order(
    tmp_path: Path, snapshot_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    ordered = make_rows()
    shuffled = list(ordered)
    random.Random(3).shuffle(shuffled)
    protocol, snapshot = _setup(tmp_path, snapshot_dir, rows=ordered)

    other_dir = tmp_path_factory.mktemp("shuffled") / "snapshot"
    other_protocol, other_snapshot = _setup(
        tmp_path_factory.mktemp("protocol"), other_dir, rows=shuffled
    )

    assert assign_splits(snapshot, protocol) == assign_splits(
        other_snapshot, other_protocol
    )


def test_assign_splits_changes_with_the_seed(
    tmp_path: Path, snapshot_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    other_protocol, other_snapshot = _setup(
        tmp_path_factory.mktemp("other"),
        tmp_path_factory.mktemp("other-snapshot") / "snapshot",
        split={"seed": 99},
    )

    assert assign_splits(snapshot, protocol) != assign_splits(
        other_snapshot, other_protocol
    )


def test_assign_splits_respects_requested_proportions(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)

    assignment = assign_splits(snapshot, protocol)

    counts: dict[str, int] = {}
    for name in assignment.values():
        counts[name] = counts.get(name, 0) + 1
    assert counts == {"train": 12, "validation": 5, "calibration": 2, "test": 5}


def test_group_splitting_keeps_every_group_in_one_split(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir, split=GROUP_SPLIT)

    assignment = assign_splits(snapshot, protocol)

    placements: dict[str, set[str]] = {}
    for sample in snapshot.samples:
        placements.setdefault(sample.group, set()).add(assignment[sample.sample_id])
    assert all(len(names) == 1 for names in placements.values())


def test_stratified_grouping_keeps_both_classes_in_train(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(
        tmp_path,
        snapshot_dir,
        classification=True,
        split={**GROUP_SPLIT, "stratify": True},
    )

    assignment = assign_splits(snapshot, protocol)
    parts = partition(assignment, snapshot)

    labels = {str(sample.target) for sample in parts["train"]}
    assert labels == {"metal", "insulator"}


def test_zero_ratio_splits_receive_no_samples(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(
        tmp_path,
        snapshot_dir,
        split={"train": 0.6, "validation": 0.2, "calibration": 0.0, "test": 0.2},
    )

    assignment = assign_splits(snapshot, protocol)

    assert "calibration" not in set(assignment.values())
    assert "calibration" not in partition(assignment, snapshot)


def test_check_assignment_rejects_missing_samples(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    assignment = assign_splits(snapshot, protocol)
    del assignment["syn-000"]

    with pytest.raises(ValueError, match="missing 1 sample"):
        check_assignment(assignment, snapshot, protocol)


def test_check_assignment_rejects_unknown_samples(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    assignment = assign_splits(snapshot, protocol)
    assignment["syn-999"] = "test"

    with pytest.raises(ValueError, match="1 unknown sample"):
        check_assignment(assignment, snapshot, protocol)


def test_check_assignment_rejects_unrequested_split(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(
        tmp_path,
        snapshot_dir,
        split={"train": 0.6, "validation": 0.2, "calibration": 0.0, "test": 0.2},
    )
    assignment = assign_splits(snapshot, protocol)
    assignment["syn-000"] = "calibration"

    with pytest.raises(ValueError, match="unrequested split\\(s\\): calibration"):
        check_assignment(assignment, snapshot, protocol)


def test_check_assignment_rejects_group_leakage(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir, split=GROUP_SPLIT)
    assignment = assign_splits(snapshot, protocol)
    leaked = snapshot.samples[0]
    assert leaked.group is not None
    sibling = next(
        sample for sample in snapshot.samples[1:] if sample.group == leaked.group
    )
    other = next(
        name for name in assignment.values() if name != assignment[leaked.sample_id]
    )
    assignment[sibling.sample_id] = other

    with pytest.raises(ValueError, match="span more than one split"):
        check_assignment(assignment, snapshot, protocol)


def test_split_manifest_round_trips(tmp_path: Path, snapshot_dir: Path) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    assignment = assign_splits(snapshot, protocol)
    path = tmp_path / "splits.csv"

    write_splits(path, assignment)

    assert read_splits(path, snapshot, protocol) == assignment
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sample_id,split"
    assert lines[1].startswith("syn-000,")


def test_read_splits_rejects_a_manifest_for_other_data(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    path = tmp_path / "splits.csv"
    write_splits(path, {"other-1": "train", "other-2": "test"})

    with pytest.raises(ValueError, match="missing 24 sample"):
        read_splits(path, snapshot, protocol)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "is empty"),
        ("sample\n", "header must be sample_id,split"),
        ("sample_id,split\ns000\n", "must have two fields"),
    ],
)
def test_read_splits_rejects_malformed_files(
    tmp_path: Path, snapshot_dir: Path, content: str, message: str
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    path = tmp_path / "splits.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_splits(path, snapshot, protocol)


def test_partition_returns_samples_in_snapshot_order(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    protocol, snapshot = _setup(tmp_path, snapshot_dir)
    assignment = assign_splits(snapshot, protocol)

    parts = partition(assignment, snapshot)

    assert sum(len(samples) for samples in parts.values()) == len(snapshot.samples)
    for samples in parts.values():
        ids = [sample.sample_id for sample in samples]
        assert ids == sorted(ids)
