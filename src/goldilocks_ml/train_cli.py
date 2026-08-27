"""Validate protocols and execute reproducible training runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from goldilocks_ml.datasets import Sample, Snapshot, load_snapshot
from goldilocks_ml.evaluation import (
    Prediction,
    default_positive_label,
    evaluate,
    label_at,
    select_threshold,
    train_majority,
    train_median,
)
from goldilocks_ml.protocol import TrainingProtocol, load_protocol
from goldilocks_ml.runs import (
    dumps_toml,
    environment_record,
    prepare_directory,
    resolved_document,
    run_record,
    write_json,
    write_manifest,
    write_predictions,
)
from goldilocks_ml.splitting import (
    assign_splits,
    partition,
    read_splits,
    write_splits,
)
from goldilocks_ml.trainers import FittedModel, get_trainer

# The test split is scored once, after every choice has already been made.
SELECTION_SPLIT = "validation"


def _regression_predictions(
    parts: dict[str, tuple[Sample, ...]], values: dict[str, list[float]]
) -> list[Prediction]:
    return [
        Prediction(
            sample_id=sample.sample_id,
            truth=float(sample.target),
            prediction=value,
            split=name,
        )
        for name, samples in parts.items()
        for sample, value in zip(samples, values[name], strict=True)
    ]


def _classification_predictions(
    parts: dict[str, tuple[Sample, ...]],
    scores: dict[str, list[float]],
    threshold: float,
    positive: str,
    negative: str,
) -> list[Prediction]:
    return [
        Prediction(
            sample_id=sample.sample_id,
            truth=str(sample.target),
            prediction=label_at(score, threshold, positive, negative),
            score=score,
            split=name,
        )
        for name, samples in parts.items()
        for sample, score in zip(samples, scores[name], strict=True)
    ]


def _by_split(predictions: Sequence[Prediction]) -> dict[str, list[Prediction]]:
    grouped: dict[str, list[Prediction]] = {}
    for item in predictions:
        grouped.setdefault(item.split, []).append(item)
    return grouped


def _score_all(
    protocol: TrainingProtocol,
    predictions: Sequence[Prediction],
    positive: str | None,
) -> dict[str, dict[str, Any]]:
    return {
        name: evaluate(
            protocol.task,
            items,
            protocol.evaluation.metrics,
            positive_label=positive,
        )
        for name, items in sorted(_by_split(predictions).items())
    }


def _train_regression(
    protocol: TrainingProtocol,
    parts: dict[str, tuple[Sample, ...]],
    model: FittedModel,
) -> tuple[dict[str, list[Prediction]], dict[str, Any]]:
    constant = train_median(parts["train"])
    baseline = _regression_predictions(
        parts, {name: [constant] * len(samples) for name, samples in parts.items()}
    )
    fitted = _regression_predictions(
        parts, {name: model.predict(samples) for name, samples in parts.items()}
    )
    return (
        {"baseline": baseline, "model": fitted},
        {"baseline_constant": constant},
    )


def _train_classification(
    protocol: TrainingProtocol,
    parts: dict[str, tuple[Sample, ...]],
    model: FittedModel,
) -> tuple[dict[str, list[Prediction]], dict[str, Any]]:
    labels = sorted({str(sample.target) for sample in parts["train"]})
    positive = protocol.evaluation.positive_label or default_positive_label(labels)
    if positive not in labels:
        raise ValueError(f"positive label {positive!r} is absent from the train split")
    negative = next(label for label in labels if label != positive)

    majority, frequency = train_majority(parts["train"])
    baseline_score = frequency if majority == positive else 1.0 - frequency
    baseline = [
        Prediction(
            sample_id=sample.sample_id,
            truth=str(sample.target),
            prediction=majority,
            score=baseline_score,
            split=name,
        )
        for name, samples in parts.items()
        for sample in samples
    ]

    scores = {name: model.predict(samples) for name, samples in parts.items()}
    threshold = 0.5
    selected_on = None
    metric = protocol.evaluation.threshold_metric
    if metric is not None:
        if SELECTION_SPLIT not in parts:
            raise ValueError(
                "evaluation.threshold_metric requires a non-empty validation split"
            )
        candidates = _classification_predictions(
            {SELECTION_SPLIT: parts[SELECTION_SPLIT]},
            {SELECTION_SPLIT: scores[SELECTION_SPLIT]},
            threshold,
            positive,
            negative,
        )
        threshold = select_threshold(candidates, metric, positive, negative)
        selected_on = SELECTION_SPLIT

    fitted = _classification_predictions(parts, scores, threshold, positive, negative)
    return (
        {"baseline": baseline, "model": fitted},
        {
            "positive_label": positive,
            "negative_label": negative,
            "baseline_label": majority,
            "decision_threshold": {
                "value": threshold,
                "metric": metric,
                "selected_on": selected_on,
            },
        },
    )


def execute(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    output: Path,
    *,
    splits_source: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run one protocol end to end and write its complete bundle."""
    started_at = datetime.now(UTC)
    directory = prepare_directory(output, overwrite=overwrite)
    run_id = directory.name

    if splits_source is None:
        assignment = assign_splits(snapshot, protocol)
    else:
        assignment = read_splits(splits_source, snapshot, protocol)
    write_splits(directory / "splits.csv", assignment)
    parts = partition(assignment, snapshot)

    # Only the training split ever reaches the trainer or its preprocessing.
    model = get_trainer(protocol.trainer)(protocol, parts["train"])
    model.save(directory / "model")

    if protocol.task == "regression":
        predictions, summary = _train_regression(protocol, parts, model)
        positive = None
    else:
        predictions, summary = _train_classification(protocol, parts, model)
        positive = summary["positive_label"]

    metrics = {
        "task": protocol.task,
        "primary_metric": protocol.evaluation.primary_metric,
        "baseline": protocol.evaluation.baseline,
        **summary,
        "splits": {
            source: _score_all(protocol, items, positive)
            for source, items in sorted(predictions.items())
        },
        "split_sizes": {name: len(samples) for name, samples in sorted(parts.items())},
    }

    (directory / "protocol.toml").write_text(
        dumps_toml(resolved_document(protocol)), encoding="utf-8"
    )
    write_json(directory / "dataset.json", snapshot.identity())
    write_json(directory / "environment.json", environment_record())
    write_json(directory / "metrics.json", metrics)
    write_predictions(directory / "predictions.csv", predictions)
    write_json(
        directory / "run.json",
        run_record(
            run_id,
            protocol,
            status="completed",
            started_at=started_at,
            splits_reused=splits_source is not None,
        ),
    )
    manifest = write_manifest(directory)
    return {"directory": directory, "metrics": metrics, "manifest": manifest}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="check a protocol and dataset snapshot without training"
    )
    validate.add_argument("protocol", type=Path)
    validate.add_argument("--dataset", type=Path, required=True)

    run = subparsers.add_parser("run", help="train, evaluate, and write a run bundle")
    run.add_argument("protocol", type=Path)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="reuse an existing splits.csv instead of deriving one",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory",
    )
    return parser


def cli() -> None:
    """Run the training protocol command-line interface."""
    args = _parser().parse_args()
    protocol = load_protocol(args.protocol)
    snapshot = load_snapshot(args.dataset, protocol)

    if args.command == "validate":
        assignment = assign_splits(snapshot, protocol)
        sizes = {}
        for name in assignment.values():
            sizes[name] = sizes.get(name, 0) + 1
        summary = ", ".join(f"{name}={sizes[name]}" for name in sorted(sizes))
        print(
            f"Valid protocol {protocol.id} against "
            f"{snapshot.record_id}@{snapshot.snapshot_version}: "
            f"{len(snapshot.samples)} samples ({summary})"
        )
        return

    result = execute(
        protocol,
        snapshot,
        args.output,
        splits_source=args.splits,
        overwrite=args.overwrite,
    )
    primary = protocol.evaluation.primary_metric
    scores = result["metrics"]["splits"]
    print(
        f"Wrote run bundle {result['directory']} "
        f"(test {primary}: model {scores['model']['test'][primary]:.6g}, "
        f"baseline {scores['baseline']['test'][primary]:.6g})"
    )


if __name__ == "__main__":
    cli()
