"""Train the metallicity classifier this repository owns.

The published checkpoint reports no accuracy, no split, and no dataset beyond a
sentence of prose, so a consumer that serves it cannot say how often it is
right. This trainer produces a classifier whose record states all three,
written by the run that fitted it rather than reconstructed afterwards.

The architecture is the published one, unchanged, so the two are comparable.
What is new is that the fit is reproducible: one seed, one sealed snapshot, one
group-disjoint split, and early stopping decided on validation alone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch_geometric.data import Batch

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.models.k_points.k_distance.qrf.embedding import CGCNN
from goldilocks_ml.models.metallicity.is_metal.cgcnn.graphs import ATOM_INIT, graphs_for
from goldilocks_ml.registry import FeatureMatrix, FittedModel, register_trainer

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from goldilocks_ml.protocol import TrainingProtocol
    from goldilocks_ml.registry import TrainingContext
    from goldilocks_ml.snapshot import Sample

TRAINER = "cgcnn_classifier"
RUNTIME = "metallicity.is_metal.cgcnn"
RUNTIME_VERSION = 1
RECORD_SCHEMA_VERSION = 1
MODEL_FILE = "is_metal.pt"
MODEL_RECORD_FILE = "model.json"

# The published checkpoint's architecture, which this reproduces so that the
# two classifiers can be compared on the same terms.
ARCHITECTURE: dict[str, Any] = {
    "orig_atom_fea_len": 92,
    "edge_feat_dim": 64,
    "h_fea_len": 128,
    "atom_fea_len": 64,
    "n_conv": 3,
    "n_h": 3,
    "num_classes": 2,
    "pooling_type": "mean_pool",
}


@dataclass(frozen=True, slots=True)
class CGCNNClassifier:
    """A fitted metallicity classifier and the record describing its fit."""

    state_dict: dict[str, torch.Tensor]
    architecture: dict[str, Any]
    atom_init_digest: str
    positive_label: str
    negative_label: str
    seed: int
    target_name: str
    target_contract: str
    feature_schema: str
    requires_artifacts: tuple[dict[str, str], ...]
    hyperparameters: dict[str, Any]
    training: dict[str, Any]
    atom_init: Path = field(compare=False, default=Path())

    def _model(self) -> CGCNN:
        model = CGCNN(**self.architecture)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[float]:
        """Return the probability of the positive class, one per sample."""
        del features  # a graph model reads structures, not a feature row
        if not samples:
            return []
        return _scores(self._model(), graphs_for(samples, self.atom_init))

    def describe(self) -> dict[str, Any]:
        """Return the JSON record a predictor reads this model back through."""
        return {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "runtime": {"id": RUNTIME, "version": RUNTIME_VERSION},
            "trainer": TRAINER,
            "task": "classification",
            "seed": self.seed,
            # Seeding fixes the initialisation and the batch order, but the
            # graph convolutions reduce with non-deterministic kernels: two
            # runs of this protocol on the same device agree to about 1e-4 per
            # score and produce different weight bytes.
            "deterministic": False,
            "architecture": dict(self.architecture),
            "classes": {
                "positive": self.positive_label,
                "negative": self.negative_label,
            },
            "target": {
                "name": self.target_name,
                "contract": self.target_contract,
                "units": None,
            },
            "feature_schema": self.feature_schema,
            "feature_columns": [],
            "feature_parameters": {},
            "requires_artifacts": [dict(item) for item in self.requires_artifacts],
            "atom_init_sha256": self.atom_init_digest,
            "hyperparameters": dict(self.hyperparameters),
            "training": dict(self.training),
            "artifacts": {"estimator": MODEL_FILE},
        }

    def save(self, directory: Path) -> None:
        """Write the weights and the record that describes them."""
        torch.save(
            {"architecture": dict(self.architecture), "state_dict": self.state_dict},
            directory / MODEL_FILE,
        )
        record = self.describe()
        record["artifacts"]["estimator_sha256"] = sha256_file(directory / MODEL_FILE)
        (directory / MODEL_RECORD_FILE).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _scores(
    model: CGCNN,
    graphs: Sequence[Data],
    batch_size: int = 256,
    device: torch.device | None = None,
) -> list[float]:
    """Return positive-class probabilities for a list of graphs."""
    target = device or torch.device("cpu")
    model = model.to(target)
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(list(graphs[start : start + batch_size]))
            probabilities = torch.softmax(model(batch.to(target)), dim=1)[:, 1]
            values.extend(float(value) for value in probabilities.cpu())
    return values


def _labels(protocol: TrainingProtocol, samples: Sequence[Sample]) -> tuple[str, str]:
    """Return the positive and negative class names, in that order."""
    present = sorted({str(sample.target) for sample in samples})
    if len(present) != 2:
        raise ValueError(f"{TRAINER} needs exactly two training classes")
    positive = protocol.evaluation.positive_label or present[-1]
    if positive not in present:
        raise ValueError(f"positive label {positive!r} is absent from the train split")
    negative = next(label for label in present if label != positive)
    return positive, negative


def _parameters(protocol: TrainingProtocol) -> dict[str, Any]:
    """Validate and default the trainer's hyperparameters."""
    given = dict(protocol.model.parameters)
    known = {
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 8,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "device": "auto",
        "architecture": {},
    }
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"unknown {TRAINER} parameter(s): {', '.join(unknown)}")
    settings = {**known, **given}
    for name in ("epochs", "batch_size", "patience", "scheduler_patience"):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model.parameters.{name} must be a positive integer")
    for name in ("learning_rate", "weight_decay"):
        value = settings[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"model.parameters.{name} must not be negative")
    if not settings["learning_rate"] > 0:
        raise ValueError("model.parameters.learning_rate must be positive")
    factor = settings["scheduler_factor"]
    if not isinstance(factor, (int, float)) or isinstance(factor, bool):
        raise ValueError("model.parameters.scheduler_factor must be a number")
    if not 0 < factor < 1:
        raise ValueError("model.parameters.scheduler_factor must lie in (0, 1)")
    if settings["device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("model.parameters.device must be auto, cpu, mps, or cuda")
    if not isinstance(settings["architecture"], dict):
        raise ValueError("model.parameters.architecture must be a table")
    unknown_architecture = sorted(set(settings["architecture"]) - set(ARCHITECTURE))
    if unknown_architecture:
        raise ValueError(
            f"unknown architecture key(s): {', '.join(unknown_architecture)}"
        )
    return settings


def _targets(samples: Sequence[Sample], positive: str) -> torch.Tensor:
    return torch.tensor(
        [1 if str(sample.target) == positive else 0 for sample in samples],
        dtype=torch.long,
    )


def _epoch_loss(
    model: CGCNN,
    graphs: Sequence[Data],
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return mean loss over a split without updating anything."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            window = list(graphs[start : start + batch_size])
            batch = Batch.from_data_list(window).to(device)
            loss = criterion(model(batch), targets[start : start + batch_size])
            total += float(loss.detach()) * len(window)
    return total / len(graphs)


def fit(protocol: TrainingProtocol, context: TrainingContext) -> FittedModel:
    """Fit on train, stop on validation, and never look at test."""
    # Configuration is cheap to check and wrong for free, so it is checked
    # before anything reads a split or opens a file.
    if protocol.task != "classification":
        raise ValueError(f"{TRAINER} requires a classification protocol")
    settings = _parameters(protocol)
    architecture = {**ARCHITECTURE, **settings["architecture"]}
    device = resolve_device(str(settings["device"]))

    if context.validation is None or not context.validation.samples:
        raise ValueError(f"{TRAINER} requires a non-empty validation split")
    if ATOM_INIT not in context.artifacts:
        raise ValueError(f"{TRAINER} requires the {ATOM_INIT} artifact")
    atom_init = Path(context.artifacts[ATOM_INIT])
    positive, negative = _labels(protocol, context.train.samples)

    torch.manual_seed(protocol.model.seed)

    train_graphs = graphs_for(context.train.samples, atom_init)
    train_targets = _targets(context.train.samples, positive)
    validation_graphs = graphs_for(context.validation.samples, atom_init)
    validation_targets = _targets(context.validation.samples, positive)

    model = CGCNN(**architecture).to(device)
    train_targets = train_targets.to(device)
    validation_targets = validation_targets.to(device)
    # AdamW rather than Adam: decoupled decay is what the published run
    # configured, and it is the correct pairing for a non-zero weight decay.
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    # The published run configured OneCycle, which needs a step budget fixed in
    # advance and so cannot coexist with stopping when validation stops
    # improving. Halving on a plateau reaches the same place without one.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        factor=float(settings["scheduler_factor"]),
        patience=int(settings["scheduler_patience"]),
    )
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = int(settings["batch_size"])

    order = torch.randperm(len(train_graphs), generator=_generator(protocol.model.seed))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    history: list[dict[str, float]] = []
    since_improvement = 0

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        order = order[torch.randperm(len(order))]
        running = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if len(indices) < 2:
                continue  # BatchNorm needs more than one graph
            batch = Batch.from_data_list([train_graphs[index] for index in indices]).to(
                device
            )
            optimiser.zero_grad()
            loss = criterion(model(batch), train_targets[indices])
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(indices)
        train_loss = running / len(order)
        validation_loss = _epoch_loss(
            model, validation_graphs, validation_targets, criterion, batch_size, device
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimiser.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= int(settings["patience"]):
                break

    if not best_state:
        raise ValueError("training produced no improving epoch")

    return CGCNNClassifier(
        state_dict=best_state,
        architecture=architecture,
        atom_init_digest=sha256_file(atom_init),
        positive_label=positive,
        negative_label=negative,
        seed=protocol.model.seed,
        target_name=protocol.dataset.target,
        target_contract=protocol.dataset.target_contract,
        feature_schema=protocol.features.schema,
        requires_artifacts=tuple(
            {
                "name": dependency.name,
                "record_id": dependency.record_id,
                "file": dependency.file,
                "sha256": dependency.sha256,
            }
            for dependency in protocol.features.depends_on
        ),
        hyperparameters={
            key: settings[key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "patience",
                "scheduler_factor",
                "scheduler_patience",
                "device",
            )
        },
        training={
            "selected_epoch": best_epoch,
            "validation_loss": best_loss,
            "epochs_run": len(history),
            "criterion": "cross_entropy",
            "optimiser": "adamw",
            "scheduler": "reduce_on_plateau",
            "class_weights": False,
            "device": device.type,
            "stopped_early": len(history) < int(settings["epochs"]),
            "history": history,
        },
        atom_init=atom_init,
    )


def resolve_device(requested: str) -> torch.device:
    """Return the device to fit on, preferring an accelerator when asked."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


register_trainer(TRAINER, fit)
