"""Shared metrics and train-derived baselines for every protocol."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from goldilocks_ml.datasets import Sample

THRESHOLD_METRICS = frozenset(
    {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"}
)
RANKING_METRICS = frozenset({"roc_auc", "pr_auc"})
BINARY_ONLY_METRICS = (
    THRESHOLD_METRICS - {"accuracy", "balanced_accuracy"} | RANKING_METRICS
)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model or baseline prediction for a single sample."""

    sample_id: str
    truth: float | str
    prediction: float | str
    score: float | None = None
    split: str = ""


def default_positive_label(labels: Sequence[str]) -> str:
    """Return the deterministic positive class when a protocol names none."""
    unique = sorted(set(labels))
    if len(unique) != 2:
        raise ValueError("a positive label is only defined for binary targets")
    return unique[-1]


def train_median(samples: Sequence[Sample]) -> float:
    """Return the regression baseline constant fitted on training data only."""
    values = [float(sample.target) for sample in samples]
    if not values:
        raise ValueError("the train split is empty")
    return statistics.median(values)


def train_majority(samples: Sequence[Sample]) -> tuple[str, float]:
    """Return the majority label and its training frequency."""
    counts: dict[str, int] = {}
    for sample in samples:
        label = str(sample.target)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("the train split is empty")
    label = min(counts, key=lambda name: (-counts[name], name))
    return label, counts[label] / sum(counts.values())


def _regression_metric(name: str, truth: list[float], predicted: list[float]) -> float:
    errors = [t - p for t, p in zip(truth, predicted, strict=True)]
    if name == "mae":
        return sum(abs(error) for error in errors) / len(errors)
    if name == "rmse":
        return math.sqrt(sum(error * error for error in errors) / len(errors))
    if name == "r2":
        mean = sum(truth) / len(truth)
        total = sum((value - mean) ** 2 for value in truth)
        residual = sum(error * error for error in errors)
        if total == 0:
            raise ValueError("r2 is undefined when every true value is identical")
        return 1.0 - residual / total
    raise ValueError(f"unsupported regression metric: {name}")


def _counts(
    truth: list[str], predicted: list[str], positive: str
) -> tuple[int, int, int, int]:
    true_positive = true_negative = false_positive = false_negative = 0
    for actual, guess in zip(truth, predicted, strict=True):
        if guess == positive:
            if actual == positive:
                true_positive += 1
            else:
                false_positive += 1
        elif actual == positive:
            false_negative += 1
        else:
            true_negative += 1
    return true_positive, true_negative, false_positive, false_negative


def _roc_auc(truth: list[str], scores: list[float], positive: str) -> float:
    positives = [
        score for actual, score in zip(truth, scores, strict=True) if actual == positive
    ]
    negatives = [
        score for actual, score in zip(truth, scores, strict=True) if actual != positive
    ]
    if not positives or not negatives:
        raise ValueError("roc_auc needs both classes present in the evaluated split")
    ordered = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(ordered):
        end = position
        while (
            end + 1 < len(ordered)
            and scores[ordered[end + 1]] == scores[ordered[position]]
        ):
            end += 1
        shared = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[ordered[index]] = shared
        position = end + 1
    rank_sum = sum(
        rank for actual, rank in zip(truth, ranks, strict=True) if actual == positive
    )
    count_positive, count_negative = len(positives), len(negatives)
    return (rank_sum - count_positive * (count_positive + 1) / 2) / (
        count_positive * count_negative
    )


def _pr_auc(truth: list[str], scores: list[float], positive: str) -> float:
    total_positive = sum(1 for actual in truth if actual == positive)
    if total_positive == 0 or total_positive == len(truth):
        raise ValueError("pr_auc needs both classes present in the evaluated split")
    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], truth[index] == positive),
    )
    true_positive = false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for index in order:
        if truth[index] == positive:
            true_positive += 1
        else:
            false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_positive
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return area


def _classification_metric(
    name: str,
    truth: list[str],
    predicted: list[str],
    scores: list[float] | None,
    positive: str,
) -> float:
    if name == "accuracy":
        correct = sum(
            1 for actual, guess in zip(truth, predicted, strict=True) if actual == guess
        )
        return correct / len(truth)
    if name in RANKING_METRICS:
        if scores is None:
            raise ValueError(f"{name} needs prediction scores")
        return (
            _roc_auc(truth, scores, positive)
            if name == "roc_auc"
            else _pr_auc(truth, scores, positive)
        )

    true_positive, true_negative, false_positive, false_negative = _counts(
        truth, predicted, positive
    )
    if name == "balanced_accuracy":
        actual_positive = true_positive + false_negative
        actual_negative = true_negative + false_positive
        if actual_positive == 0 or actual_negative == 0:
            raise ValueError(
                "balanced_accuracy needs both classes present in the evaluated split"
            )
        return (true_positive / actual_positive + true_negative / actual_negative) / 2
    if name == "precision":
        denominator = true_positive + false_positive
        return true_positive / denominator if denominator else 0.0
    if name == "recall":
        denominator = true_positive + false_negative
        return true_positive / denominator if denominator else 0.0
    if name == "f1":
        denominator = 2 * true_positive + false_positive + false_negative
        return 2 * true_positive / denominator if denominator else 0.0
    if name == "mcc":
        numerator = true_positive * true_negative - false_positive * false_negative
        denominator = math.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        return numerator / denominator if denominator else 0.0
    raise ValueError(f"unsupported classification metric: {name}")


def evaluate(
    task: str,
    predictions: Sequence[Prediction],
    metrics: Sequence[str],
    *,
    positive_label: str | None = None,
) -> dict[str, Any]:
    """Score one split, reporting every metric a protocol requested."""
    if not predictions:
        raise ValueError("cannot evaluate an empty split")
    result: dict[str, Any] = {"count": len(predictions)}
    if task == "regression":
        truth = [float(item.truth) for item in predictions]
        predicted = [float(item.prediction) for item in predictions]
        for name in metrics:
            result[name] = _regression_metric(name, truth, predicted)
        return result

    truth_labels = [str(item.truth) for item in predictions]
    predicted_labels = [str(item.prediction) for item in predictions]
    scores = (
        [float(item.score) for item in predictions]
        if all(item.score is not None for item in predictions)
        else None
    )
    positive = positive_label or default_positive_label(truth_labels)
    classes = sorted(set(truth_labels) | set(predicted_labels))
    if len(classes) > 2:
        unsupported = sorted(set(metrics) & BINARY_ONLY_METRICS)
        if unsupported:
            raise ValueError(
                f"metric(s) require a binary target: {', '.join(unsupported)}"
            )
    for name in metrics:
        result[name] = _classification_metric(
            name, truth_labels, predicted_labels, scores, positive
        )
    result["positive_label"] = positive
    result["confusion_matrix"] = {
        actual: {
            guess: sum(
                1
                for a, g in zip(truth_labels, predicted_labels, strict=True)
                if a == actual and g == guess
            )
            for guess in classes
        }
        for actual in classes
    }
    return result


def label_at(score: float, threshold: float, positive: str, negative: str) -> str:
    """Apply a decision threshold to one positive-class score."""
    return positive if score >= threshold else negative


def select_threshold(
    predictions: Sequence[Prediction],
    metric: str,
    positive: str,
    negative: str,
) -> float:
    """Choose the decision threshold maximising one metric on a single split."""
    if metric not in THRESHOLD_METRICS:
        raise ValueError(f"{metric} does not depend on a decision threshold")
    scores = [item.score for item in predictions]
    if any(score is None for score in scores):
        raise ValueError("threshold selection needs prediction scores")
    values = sorted({float(score) for score in scores if score is not None})
    candidates = [values[0]]
    candidates.extend(
        (lower + upper) / 2 for lower, upper in zip(values, values[1:], strict=False)
    )
    candidates.append(math.nextafter(values[-1], math.inf))
    truth = [str(item.truth) for item in predictions]
    best_threshold = candidates[0]
    best_value = -math.inf
    for threshold in candidates:
        predicted = [
            label_at(float(item.score), threshold, positive, negative)
            for item in predictions
            if item.score is not None
        ]
        value = _classification_metric(metric, truth, predicted, None, positive)
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return best_threshold
