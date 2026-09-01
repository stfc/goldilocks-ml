"""Seal a snapshot, validate a protocol against it, and run one training job."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import goldilocks_ml.baselines  # noqa: F401  (registers reference trainers)
import goldilocks_ml.tabular  # noqa: F401  (registers the tabular contract)
from goldilocks_ml import artifacts as artifact_store
from goldilocks_ml.evaluation import (
    Prediction,
    default_positive_label,
    evaluate,
    label_at,
    select_threshold,
    train_majority,
    train_median,
)
from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.protocol import TrainingProtocol, load_protocol
from goldilocks_ml.registry import (
    FittedModel,
    QuantileFittedModel,
    TrainingContext,
    TrainingPartition,
    get_feature_contract,
    get_trainer,
)
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
from goldilocks_ml.snapshot import (
    FEATURES_NAME,
    ID_PROP_NAME,
    MANIFEST_NAME,
    Sample,
    Snapshot,
    load_snapshot,
    read_sample_ids,
)
from goldilocks_ml.splitting import (
    assign_splits,
    partition,
    read_splits,
    write_splits,
)

# The test split is scored once, after every choice has already been made.
SELECTION_SPLIT = "validation"


def seal(
    directory: Path,
    *,
    record_id: str,
    snapshot_version: str,
    structure_suffix: str,
    target_name: str,
    target_contract: str,
    target_definition: str,
    target_units: str | None,
) -> dict[str, Any]:
    """Write the manifest that turns a converted directory into a snapshot."""
    directory = directory.resolve()
    nested = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if nested:
        raise ValueError(
            "snapshot directories must be flat; found subdirectory: " + nested[0]
        )
    for field, value in (
        ("record_id", record_id),
        ("snapshot_version", snapshot_version),
        ("target", target_name),
        ("target_contract", target_contract),
        ("target_definition", target_definition),
    ):
        if not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    if target_units is not None and not target_units.strip():
        raise ValueError("target_units must be null or a non-empty string")
    if (
        not structure_suffix.startswith(".")
        or Path(f"sample{structure_suffix}").name != f"sample{structure_suffix}"
    ):
        raise ValueError("structure_suffix must start with '.' and contain no path")
    id_prop = directory / ID_PROP_NAME
    if not id_prop.is_file():
        raise FileNotFoundError(
            f"{id_prop} is missing; a snapshot needs {ID_PROP_NAME} with "
            "sample_id,target and an optional group column"
        )
    sample_ids = read_sample_ids(id_prop)

    features_file = None
    if (directory / FEATURES_NAME).is_file():
        features_file = FEATURES_NAME

    structures = [f"{sample_id}{structure_suffix}" for sample_id in sample_ids]
    present = [name for name in structures if (directory / name).is_file()]
    if present and len(present) != len(structures):
        missing = sorted(set(structures) - set(present))
        raise FileNotFoundError(
            f"{len(missing)} structure file(s) are missing, starting with "
            f"{missing[0]}; every sample needs one or none may have one"
        )
    names = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != MANIFEST_NAME
    )

    manifest = {
        "schema_version": 1,
        "record_id": record_id,
        "snapshot_version": snapshot_version,
        "target": {
            "name": target_name,
            "contract": target_contract,
            "definition": target_definition,
            "units": target_units,
        },
        "structure_suffix": structure_suffix if present else None,
        "features_file": features_file,
        "files": [
            {
                "name": name,
                "size_bytes": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
            }
            for name in names
        ],
    }
    path = directory / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "manifest_sha256": sha256_file(path)}


def _regression_predictions(
    parts: dict[str, tuple[Sample, ...]],
    values: dict[str, list[float]],
    intervals: dict[str, list[tuple[float, float]]] | None = None,
) -> list[Prediction]:
    return [
        Prediction(
            sample_id=sample.sample_id,
            truth=float(sample.target),
            prediction=value,
            split=name,
            lower=(intervals[name][index][0] if intervals is not None else None),
            upper=(intervals[name][index][1] if intervals is not None else None),
        )
        for name, samples in parts.items()
        for index, (sample, value) in enumerate(zip(samples, values[name], strict=True))
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
    quantiles: Sequence[float] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        name: evaluate(
            protocol.task,
            items,
            protocol.evaluation.metrics,
            positive_label=positive,
            quantiles=quantiles,
        )
        for name, items in sorted(_by_split(predictions).items())
    }


def _train_regression(
    parts: dict[str, tuple[Sample, ...]], model: FittedModel, features: Any
) -> tuple[dict[str, list[Prediction]], dict[str, Any]]:
    constant = train_median(parts["train"])
    baseline = _regression_predictions(
        parts, {name: [constant] * len(samples) for name, samples in parts.items()}
    )
    if isinstance(model, QuantileFittedModel):
        quantiles = {
            name: model.predict_quantiles(samples, features.subset(samples))
            for name, samples in parts.items()
        }
        fitted = _regression_predictions(
            parts,
            {
                name: [median for _, median, _ in rows]
                for name, rows in quantiles.items()
            },
            {
                name: [(lower, upper) for lower, _, upper in rows]
                for name, rows in quantiles.items()
            },
        )
    else:
        fitted = _regression_predictions(
            parts,
            {
                name: model.predict(samples, features.subset(samples))
                for name, samples in parts.items()
            },
        )
    return {"baseline": baseline, "model": fitted}, {"baseline_constant": constant}


def _train_classification(
    protocol: TrainingProtocol,
    parts: dict[str, tuple[Sample, ...]],
    model: FittedModel,
    features: Any,
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

    scores = {
        name: model.predict(samples, features.subset(samples))
        for name, samples in parts.items()
    }
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
        threshold = select_threshold(
            candidates,
            metric,
            positive,
            negative,
            min_recall=protocol.evaluation.min_recall,
        )
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
                "min_recall": protocol.evaluation.min_recall,
                "selected_on": selected_on,
            },
        },
    )


def build_features(
    protocol: TrainingProtocol, snapshot: Snapshot, artifact_dir: Path
) -> tuple[Any, dict[str, Path]]:
    """Resolve pinned artifacts, then build features for the whole snapshot."""
    resolved = artifact_store.resolve(protocol.features.depends_on, artifact_dir)
    contract = get_feature_contract(protocol.features.schema)
    features = contract(protocol, snapshot, resolved)
    features.validate(snapshot)
    return features, resolved


def _check_runtime(protocol: TrainingProtocol, model: FittedModel) -> None:
    """Reject a trainer whose serving runtime the release name does not claim.

    A release name's first three parts are its runtime, so a protocol naming
    one setting cannot quietly be fitted by a trainer that serves another.
    Reference trainers declare no runtime and are exempt.
    """
    runtime = model.describe().get("runtime")
    if runtime is None:
        return
    produced = runtime.get("id")
    if produced != protocol.release.runtime:
        raise ValueError(
            f"protocol.id names runtime {protocol.release.runtime!r} but "
            f"trainer {protocol.trainer!r} produces {produced!r}"
        )


def execute(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    output: Path,
    *,
    artifact_dir: Path,
    splits_source: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run one protocol end to end and write its complete bundle."""
    started_at = datetime.now(UTC)
    features, resolved = build_features(protocol, snapshot, artifact_dir)
    directory = prepare_directory(output, overwrite=overwrite)
    run_id = directory.name

    if splits_source is None:
        assignment = assign_splits(snapshot, protocol)
    else:
        assignment = read_splits(splits_source, snapshot, protocol)
    write_splits(directory / "splits.csv", assignment)
    parts = partition(assignment, snapshot)

    context = TrainingContext(
        train=TrainingPartition(
            samples=parts["train"], features=features.subset(parts["train"])
        ),
        validation=(
            TrainingPartition(
                samples=parts["validation"],
                features=features.subset(parts["validation"]),
            )
            if "validation" in parts
            else None
        ),
        calibration=(
            TrainingPartition(
                samples=parts["calibration"],
                features=features.subset(parts["calibration"]),
            )
            if "calibration" in parts
            else None
        ),
        artifacts=resolved,
        output_dir=directory / "model",
    )
    # Test samples, labels, and features never reach the trainer.
    model = get_trainer(protocol.trainer)(protocol, context)
    _check_runtime(protocol, model)
    model.save(directory / "model")

    quantiles = (
        tuple(model.quantiles) if isinstance(model, QuantileFittedModel) else None
    )
    if protocol.task == "regression":
        predictions, summary = _train_regression(parts, model, features)
        positive = None
    else:
        predictions, summary = _train_classification(protocol, parts, model, features)
        positive = summary["positive_label"]
        quantiles = None

    metrics = {
        "task": protocol.task,
        "target": protocol.dataset.target,
        "primary_metric": protocol.evaluation.primary_metric,
        "baseline": protocol.evaluation.baseline,
        **summary,
        "splits": {
            # A baseline predicts a point, so quantile scoring applies to the
            # model alone.
            source: _score_all(
                protocol, items, positive, quantiles if source == "model" else None
            )
            for source, items in sorted(predictions.items())
        },
        "split_sizes": {name: len(samples) for name, samples in sorted(parts.items())},
    }

    (directory / "protocol.toml").write_text(
        dumps_toml(resolved_document(protocol)), encoding="utf-8"
    )
    write_json(
        directory / "dataset.json",
        {
            **snapshot.identity(),
            "pinned_by_protocol": protocol.dataset.pinned is not None,
            "feature_schema": protocol.features.schema,
            "feature_columns": list(features.columns),
            "artifacts": {
                dependency.name: {
                    "record_id": dependency.record_id,
                    "file": dependency.file,
                    "sha256": dependency.sha256,
                }
                for dependency in protocol.features.depends_on
            },
        },
    )
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


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Register the ``train`` group on the shared command line."""
    parser = groups.add_parser(
        "train",
        help="seal a snapshot, validate a protocol, and run a training job",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(handler=_run)
    subparsers = parser.add_subparsers(dest="train_command", required=True)

    seal_parser = subparsers.add_parser(
        "seal", help="write manifest.json for a converted snapshot directory"
    )
    seal_parser.add_argument("snapshot", type=Path)
    seal_parser.add_argument("--record-id", required=True)
    seal_parser.add_argument("--version", required=True, dest="snapshot_version")
    seal_parser.add_argument("--structure-suffix", default=".cif")
    seal_parser.add_argument("--target", required=True, dest="target_name")
    seal_parser.add_argument("--target-contract", required=True)
    seal_parser.add_argument("--target-definition", required=True)
    seal_parser.add_argument("--target-units", default=None)

    validate = subparsers.add_parser(
        "validate", help="check a protocol and snapshot without training"
    )
    validate.add_argument("protocol", type=Path)
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--artifact-directory", type=Path, default=None)

    run = subparsers.add_parser("run", help="train, evaluate, and write a run bundle")
    run.add_argument("protocol", type=Path)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--artifact-directory", type=Path, default=None)
    run.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="reuse an existing splits.csv instead of deriving one",
    )
    run.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    return parser


def _run(args: argparse.Namespace) -> None:

    if args.train_command == "seal":
        result = seal(
            args.snapshot,
            record_id=args.record_id,
            snapshot_version=args.snapshot_version,
            structure_suffix=args.structure_suffix,
            target_name=args.target_name,
            target_contract=args.target_contract,
            target_definition=args.target_definition,
            target_units=args.target_units,
        )
        count = len(result["manifest"]["files"])
        print(
            f"Sealed {args.snapshot} as "
            f"{args.record_id}@{args.snapshot_version}: {count} file(s), "
            f"manifest SHA-256 {result['manifest_sha256']}"
        )
        return

    protocol = load_protocol(args.protocol)
    snapshot = load_snapshot(args.dataset, protocol)
    artifact_dir = artifact_store.artifact_directory(args.artifact_directory)

    if args.train_command == "validate":
        features, _ = build_features(protocol, snapshot, artifact_dir)
        assignment = assign_splits(snapshot, protocol)
        sizes: dict[str, int] = {}
        for name in assignment.values():
            sizes[name] = sizes.get(name, 0) + 1
        summary = ", ".join(f"{name}={sizes[name]}" for name in sorted(sizes))
        print(
            f"Valid protocol {protocol.id} against "
            f"{snapshot.record_id}@{snapshot.snapshot_version}: "
            f"{len(snapshot.samples)} samples, {len(features.columns)} features "
            f"({summary})"
        )
        return

    result = execute(
        protocol,
        snapshot,
        args.output,
        artifact_dir=artifact_dir,
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
