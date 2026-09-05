"""Tests for the derived classification target and the dense-mesh screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import classification_document, write_protocol

from goldilocks_ml.inference import contract_for
from goldilocks_ml.models.k_points.k_index.screen.trainer import ranking_quality
from goldilocks_ml.protocol import load_protocol

DERIVE = {
    "kind": "at_or_above",
    "threshold": 11,
    "positive": "dense",
    "negative": "sparse",
    "contract": "goldilocks.k_index_dense.ladder_0based.ge11.v1",
}


def _protocol(tmp_path: Path, **derive: Any) -> Path:
    table = {**DERIVE, **derive}
    return write_protocol(
        tmp_path / "protocol.toml",
        classification_document(dataset={"derive": table}),
    )


def test_a_derived_target_is_loaded_with_its_rung_and_classes(tmp_path: Path) -> None:
    derive = load_protocol(_protocol(tmp_path)).dataset.derive

    assert derive is not None
    assert derive.kind == "at_or_above"
    assert derive.threshold == 11.0
    assert derive.positive == "dense"
    assert derive.contract == "goldilocks.k_index_dense.ladder_0based.ge11.v1"


def test_the_threshold_is_inclusive_at_the_rung_it_names(tmp_path: Path) -> None:
    derive = load_protocol(_protocol(tmp_path)).dataset.derive
    assert derive is not None

    assert derive.label(10.0) == "sparse"
    assert derive.label(11.0) == "dense"
    assert derive.label(41.0) == "dense"


def test_a_protocol_without_derive_keeps_the_recorded_target(tmp_path: Path) -> None:
    protocol = write_protocol(tmp_path / "protocol.toml", classification_document())
    assert load_protocol(protocol).dataset.derive is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"kind": "above"}, "unknown dataset.derive.kind"),
        ({"threshold": "eleven"}, "threshold must be a number"),
        ({"positive": "dense", "negative": "dense"}, "must be different labels"),
    ],
)
def test_a_malformed_derivation_is_rejected(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_protocol(_protocol(tmp_path, **override))


def test_the_screen_target_contract_is_a_dataset_judgement() -> None:
    """It must not be mistaken for advice about how to run a calculation."""
    contract = contract_for("goldilocks.k_index_dense.ladder_0based.ge11.v1")

    assert contract.kind == "dataset_selection"
    assert contract.boolean is True
    contract.check_value(True)
    with pytest.raises(ValueError, match="must be a boolean"):
        contract.check_value(0.7)


def test_ranking_quality_reports_precision_recall_and_enrichment() -> None:
    # Ten samples, two positives, both ranked at the top.
    truth = [True, True] + [False] * 8
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    rows = ranking_quality(truth, scores, (0.2, 1.0))

    top, everything = rows
    assert top["taken"] == 2
    assert top["precision"] == 1.0
    assert top["recall"] == 1.0
    assert top["enrichment"] == pytest.approx(5.0)
    # Taking the whole pool cannot beat the base rate.
    assert everything["precision"] == pytest.approx(0.2)
    assert everything["enrichment"] == pytest.approx(1.0)


def test_ranking_quality_measures_a_fraction_not_a_count() -> None:
    """A budget measured on one pool must transfer to a pool of another size."""
    truth = [True] * 5 + [False] * 95
    scores = [1.0] * 5 + [0.0] * 95

    rows = ranking_quality(truth, scores, (0.05,))

    assert rows[0]["taken"] == 5
    assert rows[0]["fraction"] == 0.05


def test_ranking_quality_rejects_a_fraction_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        ranking_quality([True, False], [1.0, 0.0], (1.5,))


def test_a_numeric_snapshot_is_read_as_the_classes_the_protocol_derives(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    """One snapshot serves both models, so the classes cannot drift."""
    from conftest import build_snapshot, make_rows, pin

    from goldilocks_ml.snapshot import load_snapshot

    rows = make_rows()
    digest = build_snapshot(snapshot_dir, rows, target="value")
    document = classification_document(
        dataset={
            "target": "value",
            "target_contract": "synthetic.value.v1",
            "target_units": "arbitrary",
            "derive": {**DERIVE, "threshold": 3.0},
        },
        evaluation={"positive_label": "dense"},
    )
    protocol = load_protocol(
        write_protocol(tmp_path / "protocol.toml", pin(document, digest))
    )
    snapshot = load_snapshot(snapshot_dir, protocol)

    labels = {sample.sample_id: sample.target for sample in snapshot.samples}
    for row in rows:
        expected = "dense" if row["value"] >= 3.0 else "sparse"
        assert labels[row["sample_id"]] == expected
    # Both classes must actually be present, or the fixture proves nothing.
    assert set(labels.values()) == {"dense", "sparse"}


def _record(**overrides: Any) -> dict[str, Any]:
    """A minimal well-formed screen record, before any field is broken."""
    from goldilocks_ml.models.k_points.k_index.qrf import features

    base: dict[str, Any] = {
        "runtime": {"id": "k_points.k_index.screen", "version": 1},
        "feature_schema": features.SCHEMA,
        "feature_columns": list(features.column_names()),
        "target": {
            "name": "k_index_dense",
            "contract": "goldilocks.k_index_dense.ladder_0based.ge11.v1",
            "units": None,
        },
        "classes": {"positive": "dense", "negative": "sparse"},
        "derived_from": {"target": "k_index", "rule": "at_or_above", "threshold": 11},
        "decision": {
            "rule": "ranking",
            "fractions": [{"fraction": 0.1, "precision": 0.6, "recall": 0.7}],
        },
        "artifacts": {"estimator": "k_index_screen.pkl", "estimator_sha256": "0" * 64},
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            _record(derived_from={"target": "k_index", "rule": "at_or_above"}),
            "does not record the rung it screens at",
        ),
        (
            _record(decision={"rule": "quantile", "fractions": [{}]}),
            "serves the 'ranking' decision rule",
        ),
        (
            _record(decision={"rule": "ranking", "fractions": []}),
            "carries no measured ranking quality",
        ),
        (
            _record(runtime={"id": "k_points.k_index.screen", "version": 99}),
            "runtime version",
        ),
    ],
)
def test_the_runtime_refuses_a_record_that_does_not_say_what_it_screens_for(
    tmp_path: Path, record: dict[str, Any], message: str
) -> None:
    from goldilocks_ml.models.k_points.k_index.screen.predictor import load

    with pytest.raises(ValueError, match=message):
        load(record, tmp_path, {})
