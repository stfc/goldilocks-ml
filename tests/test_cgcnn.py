"""Tests for the trainable metallicity classifier and its graph contract."""

from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import write_protocol
from pymatgen.core import Lattice, Structure

from goldilocks_ml.cli import seal
from goldilocks_ml.models.metallicity.is_metal.cgcnn import graphs as crystal_graphs
from goldilocks_ml.models.metallicity.is_metal.cgcnn.trainer import (
    ARCHITECTURE,
    RUNTIME,
    RUNTIME_VERSION,
    TRAINER,
    resolve_device,
)
from goldilocks_ml.protocol import load_protocol
from goldilocks_ml.registry import (
    TrainingContext,
    feature_contract_names,
    get_trainer,
    trainer_names,
)
from goldilocks_ml.snapshot import Snapshot, load_snapshot

PROTOCOL = (
    Path(__file__).parents[1]
    / "protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v1.toml"
)
ATOM_TABLE = {str(number): [0.1] * 92 for number in (8, 14, 26)}


def build_metallicity_snapshot(directory: Path, *, structures: bool = True) -> None:
    """Write and seal a snapshot of real, parseable crystals."""
    directory.mkdir(parents=True, exist_ok=True)
    lattice = Lattice.cubic(5.43)
    recipes = [
        ("mpm-0001", ["Si", "Si"], "metal"),
        ("mpm-0002", ["Fe", "Fe"], "metal"),
        ("mpm-0003", ["Si", "O"], "insulator"),
        ("mpm-0004", ["Fe", "O"], "insulator"),
    ]
    rows = []
    for index, (identifier, species, label) in enumerate(recipes):
        if structures:
            Structure(lattice, species, [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]).to(
                filename=str(directory / f"{identifier}.cif")
            )
        rows.append([identifier, label, f"group-{index}"])
    with (directory / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    seal(
        directory,
        record_id="test-metallicity",
        snapshot_version="v1",
        structure_suffix=".cif" if structures else None,
        target_name="is_metal",
        target_contract="goldilocks.is_metal.dft_band_gap_zero.v1",
        target_definition="Fixture metallicity labels.",
        target_units=None,
    )


def unpinned(directory: Path, **parameters: Any) -> Any:
    """The shipped protocol with its snapshot pin removed, for fixtures.

    Written beside the snapshot rather than inside it: a sealed snapshot
    refuses to contain a file its manifest does not cover.
    """
    beside = directory.parent / "protocols"
    beside.mkdir(parents=True, exist_ok=True)
    return load_protocol(
        write_protocol(beside / f"{directory.name}.toml", document(**parameters))
    )


def snapshot_at(directory: Path, *, structures: bool = True) -> Snapshot:
    build_metallicity_snapshot(directory, structures=structures)
    return load_snapshot(directory, unpinned(directory))


def atom_table(directory: Path) -> Path:
    path = directory / "atom_init.json"
    path.write_text(json.dumps(ATOM_TABLE), encoding="utf-8")
    return path


def document(**parameters: Any) -> dict[str, Any]:
    """The shipped protocol as a document, unpinned from the real snapshot."""
    body = tomllib.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key in ("record_id", "snapshot_version", "manifest_sha256"):
        body["dataset"].pop(key, None)
    body["model"]["parameters"] = parameters
    return body


def empty_context() -> TrainingContext:
    return TrainingContext(
        train=None,  # type: ignore[arg-type]
        validation=None,
        calibration=None,
        artifacts={},
        output_dir=Path(),
    )


def test_the_trainer_and_contract_are_registered() -> None:
    assert TRAINER in trainer_names()
    assert crystal_graphs.SCHEMA in feature_contract_names()
    assert get_trainer(TRAINER) is not None


def test_the_shipped_protocol_parses() -> None:
    protocol = load_protocol(PROTOCOL)

    assert protocol.trainer == TRAINER
    assert protocol.task == "classification"
    assert protocol.features.schema == crystal_graphs.SCHEMA
    assert [item.name for item in protocol.features.depends_on] == ["atom_init"]
    assert protocol.evaluation.positive_label == "metal"


def test_the_runtime_is_distinct_from_the_published_one() -> None:
    """The published checkpoint and this one must never load as each other."""
    assert RUNTIME == "metallicity.is_metal.cgcnn"
    assert RUNTIME != "metallicity.cgcnn"
    assert RUNTIME_VERSION == 1


def test_the_architecture_matches_the_published_checkpoint() -> None:
    """The two classifiers stay comparable only while the shapes agree."""
    checkpoint = Path("local_data/artifacts/ptc95-vbq12/is_metal.ckpt")
    if not checkpoint.is_file():
        pytest.skip("the published checkpoint is not present locally")

    published = torch.load(checkpoint, map_location="cpu", weights_only=True)

    for key, value in published["hyper_parameters"]["model"].items():
        if key in ARCHITECTURE:
            assert ARCHITECTURE[key] == value, key


def test_the_graph_contract_needs_the_atom_table(tmp_path: Path) -> None:
    snapshot = snapshot_at(tmp_path)

    with pytest.raises(ValueError, match="atom_init"):
        crystal_graphs.build(unpinned(tmp_path), snapshot, {})


def test_the_graph_contract_computes_no_columns(tmp_path: Path) -> None:
    """A graph model reads crystals; the contract only asserts they are there."""
    snapshot = snapshot_at(tmp_path)

    matrix = crystal_graphs.build(
        unpinned(tmp_path), snapshot, {"atom_init": atom_table(tmp_path)}
    )

    assert matrix.columns == ()
    assert set(matrix.rows) == set(snapshot.sample_ids)
    matrix.validate(snapshot)


def test_graphs_carry_the_node_width_the_architecture_expects(tmp_path: Path) -> None:
    snapshot = snapshot_at(tmp_path)

    graphs = crystal_graphs.graphs_for(snapshot.samples, atom_table(tmp_path))

    assert len(graphs) == len(snapshot.samples)
    for graph in graphs:
        assert graph.x.shape[1] == ARCHITECTURE["orig_atom_fea_len"]
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0


def test_an_unknown_parameter_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, lr=0.1)

    with pytest.raises(ValueError, match="unknown cgcnn_classifier parameter"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_an_unknown_selection_metric_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, selection_metric="f1")

    with pytest.raises(ValueError, match="selection_metric must be one of"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_roc_auc_ranks_rather_than_thresholds() -> None:
    """The selection metric does not move when a decision threshold does."""
    from goldilocks_ml.models.metallicity.is_metal.cgcnn.trainer import _roc_auc

    targets = torch.tensor([0, 0, 1, 1])

    assert _roc_auc([0.1, 0.2, 0.3, 0.4], targets) == pytest.approx(1.0)
    # Every score shifted below any plausible threshold; the ranking is the same.
    assert _roc_auc([0.01, 0.02, 0.03, 0.04], targets) == pytest.approx(1.0)
    assert _roc_auc([0.4, 0.3, 0.2, 0.1], targets) == pytest.approx(0.0)


def test_roc_auc_needs_both_classes() -> None:
    from goldilocks_ml.models.metallicity.is_metal.cgcnn.trainer import _roc_auc

    with pytest.raises(ValueError, match="only one class"):
        _roc_auc([0.1, 0.2], torch.tensor([1, 1]))


def test_an_unknown_device_is_refused(tmp_path: Path) -> None:
    """Configuration is checked before data, so this fails without a split."""
    protocol = unpinned(tmp_path, device="tpu")

    with pytest.raises(ValueError, match="device must be auto"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_a_non_positive_learning_rate_is_refused(tmp_path: Path) -> None:
    protocol = unpinned(tmp_path, learning_rate=0.0)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_training_without_a_validation_split_is_refused(tmp_path: Path) -> None:
    """Early stopping decides on validation, so there has to be one."""
    protocol = unpinned(tmp_path)

    with pytest.raises(ValueError, match="non-empty validation split"):
        get_trainer(TRAINER)(protocol, empty_context())


def test_an_explicit_device_is_honoured() -> None:
    """The parameter must decide the device, not merely be accepted."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}


def fitted_classifier(atom_init: Path, **overrides: Any) -> Any:
    """An untrained classifier of the shipped shape, for round-trip tests."""
    from goldilocks_ml.hashing import sha256_file
    from goldilocks_ml.models.k_points.k_distance.qrf.embedding import CGCNN
    from goldilocks_ml.models.metallicity.is_metal.cgcnn.trainer import CGCNNClassifier

    torch.manual_seed(0)
    fields: dict[str, Any] = {
        "state_dict": CGCNN(**ARCHITECTURE).state_dict(),
        "architecture": dict(ARCHITECTURE),
        "atom_init_digest": sha256_file(atom_init),
        "positive_label": "metal",
        "negative_label": "insulator",
        "seed": 42,
        "target_name": "is_metal",
        "target_contract": "goldilocks.is_metal.dft_band_gap_zero.v1",
        "feature_schema": crystal_graphs.SCHEMA,
        "requires_artifacts": (),
        "hyperparameters": {},
        "training": {},
        "atom_init": atom_init,
    }
    fields.update(overrides)
    return CGCNNClassifier(**fields)


def saved_model(tmp_path: Path, **overrides: Any) -> tuple[Path, Path]:
    """Write a classifier carrying a decision, and return its directory."""
    atom_init = atom_table(tmp_path)
    model = fitted_classifier(atom_init, **overrides)
    if "decision" not in overrides:
        model = model.with_decision(
            {
                "threshold": 0.5,
                "metric": "mcc",
                "min_recall": 0.97,
                "selected_on": "validation",
                "positive_label": "metal",
                "negative_label": "insulator",
            }
        )
    directory = tmp_path / "model"
    directory.mkdir()
    model.save(directory)
    return directory, atom_init


def a_structure() -> Structure:
    return Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def test_a_served_classifier_decides_rather_than_scoring(tmp_path: Path) -> None:
    from goldilocks_ml.inference import load_model

    directory, atom_init = saved_model(tmp_path)

    model = load_model(directory, artifacts={"atom_init": atom_init})
    prediction = model.predict(a_structure())

    assert isinstance(prediction.value, bool)
    assert prediction.parameter == "metallicity"
    assert prediction.quantity == "is_metal"
    assert prediction.target_contract == "goldilocks.is_metal.dft_band_gap_zero.v1"
    # The score is provenance, not a claim: `confidence` carries guarantees.
    assert prediction.confidence is None
    assert 0.0 <= prediction.details["score"] <= 1.0
    assert prediction.details["threshold"] == 0.5
    assert prediction.details["label"] in {"metal", "insulator"}
    assert (prediction.details["label"] == "metal") is prediction.value


def test_a_batch_decides_once_per_structure(tmp_path: Path) -> None:
    from goldilocks_ml.inference import load_model

    directory, atom_init = saved_model(tmp_path)
    model = load_model(directory, artifacts={"atom_init": atom_init})

    predictions = model.predict_batch([a_structure(), a_structure()])

    assert len(predictions) == 2
    assert predictions[0].details["score"] == predictions[1].details["score"]
    assert model.predict_batch([]) == []


def test_a_record_without_a_threshold_cannot_be_served(tmp_path: Path) -> None:
    from goldilocks_ml.inference import load_model

    directory, atom_init = saved_model(tmp_path, decision={})

    with pytest.raises(ValueError, match="records no decision threshold"):
        load_model(directory, artifacts={"atom_init": atom_init})


def test_substituted_weights_are_refused(tmp_path: Path) -> None:
    from goldilocks_ml.inference import load_model

    directory, atom_init = saved_model(tmp_path)
    weights = directory / "is_metal.pt"
    weights.write_bytes(weights.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="its record pins"):
        load_model(directory, artifacts={"atom_init": atom_init})


def test_a_substituted_atom_table_is_refused(tmp_path: Path) -> None:
    from goldilocks_ml.inference import load_model

    directory, atom_init = saved_model(tmp_path)
    atom_init.write_text(
        json.dumps({key: [0.2] * 92 for key in ATOM_TABLE}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="its record pins"):
        load_model(directory, artifacts={"atom_init": atom_init})


def test_the_atom_table_must_be_supplied(tmp_path: Path) -> None:
    from goldilocks_ml.inference import load_model

    directory, _ = saved_model(tmp_path)

    with pytest.raises(ValueError, match="needs artifact"):
        load_model(directory, artifacts={})
