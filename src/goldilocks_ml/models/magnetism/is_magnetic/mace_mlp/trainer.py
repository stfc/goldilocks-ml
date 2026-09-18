"""Train the ``is_magnetic`` classifier: a frozen mMACE embedding + an MLP head.

The architecture is ported from
``2-research/2-mace/1-magnetic-mace/scripts/magnetic_classifier_model.py``
(``MagneticMLP``, ``MagneticClassifier``). What changes on the port is the
same thing the CGCNN port changed: standardisation is fit on the train split
alone and recorded as plain arrays rather than assumed, the decision
threshold is chosen by the run rather than read off a research checkpoint,
and the record states the dataset, the split, and the metrics that produced
it (AGENTS.md's reproducibility rule) instead of a sentence of prose.

One deliberate departure from the ported class: ``MagneticMLP`` no longer
defaults ``hidden``/``dropout`` to a shape no released checkpoint ever used
(the class carried a stale ``(512, 256, 128)`` default; every real checkpoint,
including the research one this ports, is ``(256, 64)``). A caller states the
architecture explicitly -- the trainer's own defaults, in :func:`_parameters`,
are what actually gets recorded.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.models.magnetism.is_magnetic.mace_mlp.features import (
    MACE_BACKBONE,
    SCHEMA,
    column_names,
)
from goldilocks_ml.registry import FeatureMatrix, FittedModel, register_trainer

if TYPE_CHECKING:
    from goldilocks_ml.protocol import TrainingProtocol
    from goldilocks_ml.registry import TrainingContext
    from goldilocks_ml.snapshot import Sample

TRAINER = "magnetic_mlp_classifier"
VALIDATION_LOSS = "validation_loss"
VALIDATION_ROC_AUC = "roc_auc"
SELECTION_METRICS = frozenset({VALIDATION_LOSS, VALIDATION_ROC_AUC})
RUNTIME = "magnetism.is_magnetic.mace_mlp"
RUNTIME_VERSION = 1
RECORD_SCHEMA_VERSION = 1
MODEL_FILE = "is_magnetic.pt"
MODEL_RECORD_FILE = "model.json"


class MagneticMLP(nn.Module):
    """384 -> ... -> 1 raw logit: Linear, BatchNorm1d, GELU, Dropout per layer."""

    def __init__(self, d_in: int, hidden: Sequence[int], dropout: float) -> None:
        super().__init__()
        dimensions = [d_in, *hidden]
        layers: list[nn.Module] = []
        for input_size, output_size in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(input_size, output_size),
                    nn.BatchNorm1d(output_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dimensions[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one raw logit per row; the caller applies sigmoid."""
        return self.net(features).squeeze(-1)


def standardize_fit(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-column mean and scale, fitted on the rows given.

    A caller must pass the train split alone: fitting on validation or test
    would leak their distribution into every prediction made on them.
    """
    mean = rows.mean(dim=0)
    scale = rows.std(dim=0, unbiased=False)
    # A column with zero variance in this split would otherwise divide by
    # zero; leaving it unscaled is equivalent to a scale of one.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return mean, scale


def standardize_apply(
    rows: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Apply a previously fitted standardisation to a batch of rows."""
    return (rows - mean) / scale


def class_scores(
    model: MagneticMLP,
    rows: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> list[float]:
    """Return P(magnetic) for a batch of already-embedded, raw (unscaled) rows."""
    model.eval()
    with torch.no_grad():
        standardized = standardize_apply(rows, mean, scale)
        probabilities = torch.sigmoid(model(standardized))
    return [float(value) for value in probabilities]


@dataclass(frozen=True, slots=True)
class MagneticMLPClassifier:
    """A fitted ``is_magnetic`` classifier and the record describing its fit."""

    state_dict: dict[str, torch.Tensor]
    architecture: dict[str, Any]
    scaler_mean: torch.Tensor
    scaler_scale: torch.Tensor
    positive_label: str
    negative_label: str
    seed: int
    target_name: str
    target_contract: str
    feature_columns: tuple[str, ...]
    requires_artifacts: tuple[dict[str, str], ...]
    hyperparameters: dict[str, Any]
    training: dict[str, Any]
    # Filled in after the fit, when the run has chosen a threshold on the
    # validation split. Empty until then, and an unserved model may stay empty.
    decision: dict[str, Any] = field(default_factory=dict)

    def _model(self) -> MagneticMLP:
        model = MagneticMLP(**self.architecture)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict(
        self, samples: Sequence[Sample], features: FeatureMatrix
    ) -> list[float]:
        """Return the probability of the positive class, one per sample."""
        if not samples:
            return []
        rows = torch.tensor(features.matrix(samples), dtype=torch.float32)
        return class_scores(self._model(), rows, self.scaler_mean, self.scaler_scale)

    def with_decision(self, decision: Any) -> MagneticMLPClassifier:
        """Return a copy carrying the threshold rule the run selected."""
        return replace(self, decision=dict(decision))

    def describe(self) -> dict[str, Any]:
        """Return the JSON record a predictor reads this model back through."""
        return {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "runtime": {"id": RUNTIME, "version": RUNTIME_VERSION},
            "trainer": TRAINER,
            "task": "classification",
            "seed": self.seed,
            "deterministic": False,
            "architecture": dict(self.architecture),
            "classes": {
                "positive": self.positive_label,
                "negative": self.negative_label,
            },
            "decision": dict(self.decision),
            "target": {
                "name": self.target_name,
                "contract": self.target_contract,
                "units": None,
            },
            "feature_schema": SCHEMA,
            "feature_columns": list(self.feature_columns),
            "feature_parameters": {},
            "requires_artifacts": [dict(item) for item in self.requires_artifacts],
            "hyperparameters": dict(self.hyperparameters),
            "training": dict(self.training),
            "artifacts": {"estimator": MODEL_FILE},
        }

    def save(self, directory: Path) -> None:
        """Write the weights, the fitted scaler, and the record describing them."""
        torch.save(
            {
                "architecture": dict(self.architecture),
                "state_dict": self.state_dict,
                "scaler_mean": self.scaler_mean,
                "scaler_scale": self.scaler_scale,
            },
            directory / MODEL_FILE,
        )
        record = self.describe()
        record["artifacts"]["estimator_sha256"] = sha256_file(directory / MODEL_FILE)
        (directory / MODEL_RECORD_FILE).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


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
    """Validate and default the trainer's hyperparameters.

    The defaults reproduce the research checkpoint this ports
    (``notebooks/data/magnetic_clf_full_best.pt``: ``hidden=(256, 64)``,
    ``dropout=0.25``), so a protocol that names nothing still recovers it.
    """
    given = dict(protocol.model.parameters)
    known = {
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 20,
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "selection_metric": VALIDATION_ROC_AUC,
        "hidden": [256, 64],
        "dropout": 0.25,
    }
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"unknown {TRAINER} parameter(s): {', '.join(unknown)}")
    settings = {**known, **given}
    if settings["selection_metric"] not in SELECTION_METRICS:
        allowed = ", ".join(sorted(SELECTION_METRICS))
        raise ValueError(f"model.parameters.selection_metric must be one of: {allowed}")
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
    hidden = settings["hidden"]

    def _bad_layer_width(value: Any) -> bool:
        return not isinstance(value, int) or isinstance(value, bool) or value <= 0

    malformed = not isinstance(hidden, list) or not hidden
    malformed = malformed or any(_bad_layer_width(value) for value in hidden)
    if malformed:
        raise ValueError(
            "model.parameters.hidden must be a non-empty list of positive integers"
        )
    dropout = settings["dropout"]
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0 <= dropout < 1
    ):
        raise ValueError("model.parameters.dropout must lie in [0, 1)")
    return settings


def _targets(samples: Sequence[Sample], positive: str) -> torch.Tensor:
    return torch.tensor(
        [1.0 if str(sample.target) == positive else 0.0 for sample in samples],
        dtype=torch.float32,
    )


def _roc_auc(scores: Sequence[float], targets: torch.Tensor) -> float:
    """Return ROC-AUC of the positive-class scores against binary targets."""
    from sklearn.metrics import roc_auc_score

    truth = targets.detach().cpu().numpy()
    if len(set(truth.tolist())) < 2:
        raise ValueError("validation split carries only one class")
    return float(roc_auc_score(truth, scores))


def fit(protocol: TrainingProtocol, context: TrainingContext) -> FittedModel:
    """Fit on train, stop on validation, and never look at test."""
    if protocol.task != "classification":
        raise ValueError(f"{TRAINER} requires a classification protocol")
    settings = _parameters(protocol)

    if context.validation is None or not context.validation.samples:
        raise ValueError(f"{TRAINER} requires a non-empty validation split")
    if MACE_BACKBONE not in context.artifacts:
        raise ValueError(f"{TRAINER} requires the {MACE_BACKBONE} artifact")
    positive, negative = _labels(protocol, context.train.samples)

    torch.manual_seed(protocol.model.seed)

    train_rows = torch.tensor(
        context.train.features.matrix(context.train.samples), dtype=torch.float32
    )
    train_targets = _targets(context.train.samples, positive)
    validation_rows = torch.tensor(
        context.validation.features.matrix(context.validation.samples),
        dtype=torch.float32,
    )
    validation_targets = _targets(context.validation.samples, positive)

    # Preprocessing is fit on train alone (AGENTS.md): validation and test
    # rows are transformed with it, never used to compute it.
    mean, scale = standardize_fit(train_rows)
    standardized_train = standardize_apply(train_rows, mean, scale)
    standardized_validation = standardize_apply(validation_rows, mean, scale)

    architecture = {
        # A list, not a tuple: this dict is compared against `model.json`'s
        # JSON-decoded copy of itself at load time (predictor.py), and JSON
        # has no tuple -- a tuple here would never compare equal to the list
        # every consumer reads back.
        "d_in": int(train_rows.shape[1]),
        "hidden": list(int(value) for value in settings["hidden"]),
        "dropout": float(settings["dropout"]),
    }
    model = MagneticMLP(**architecture)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        factor=float(settings["scheduler_factor"]),
        patience=int(settings["scheduler_patience"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    batch_size = int(settings["batch_size"])

    order = torch.randperm(
        len(standardized_train), generator=_generator(protocol.model.seed)
    )
    selection = str(settings["selection_metric"])
    best_objective = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    history: list[dict[str, float]] = []
    since_improvement = 0

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        order = order[torch.randperm(len(order))]
        running = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if len(indices) < 2:
                continue  # BatchNorm needs more than one row
            optimiser.zero_grad()
            logits = model(standardized_train[indices])
            loss = criterion(logits, train_targets[indices])
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(indices)
            seen += len(indices)
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            validation_logits = model(standardized_validation)
            validation_loss = float(criterion(validation_logits, validation_targets))
            validation_scores = [
                float(value) for value in torch.sigmoid(validation_logits)
            ]
        validation_roc_auc = _roc_auc(validation_scores, validation_targets)
        objective = (
            validation_loss if selection == VALIDATION_LOSS else -validation_roc_auc
        )
        scheduler.step(objective)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_roc_auc": validation_roc_auc,
                "learning_rate": optimiser.param_groups[0]["lr"],
            }
        )
        if objective < best_objective:
            best_objective = objective
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

    return MagneticMLPClassifier(
        state_dict=best_state,
        architecture=architecture,
        scaler_mean=mean,
        scaler_scale=scale,
        positive_label=positive,
        negative_label=negative,
        seed=protocol.model.seed,
        target_name=protocol.dataset.target,
        target_contract=protocol.dataset.target_contract,
        feature_columns=context.train.features.columns or column_names(),
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
                "selection_metric",
                "hidden",
                "dropout",
            )
        },
        training={
            "selected_epoch": best_epoch,
            "selection_metric": selection,
            "validation_loss": history[best_epoch - 1]["validation_loss"],
            "validation_roc_auc": history[best_epoch - 1]["validation_roc_auc"],
            "epochs_run": len(history),
            "criterion": "bce_with_logits",
            "optimiser": "adamw",
            "scheduler": "reduce_on_plateau",
            "stopped_early": len(history) < int(settings["epochs"]),
            "history": history,
        },
    )


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


register_trainer(TRAINER, fit)
