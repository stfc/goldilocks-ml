"""Shared metrics and train-derived baselines for every protocol."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from goldilocks_ml.snapshot import Sample

# Metrics that score the rounded prediction rather than the continuous one.
# They only mean anything where the target lives on an integer grid, such as a
# rung on a k-mesh ladder, so they check that before scoring.
INTEGER_TARGET_METRICS = frozenset(
    {"rounded_accuracy", "within_one", "underprediction_rate", "mean_excess"}
)
# Metrics a regression protocol may optimise when it chooses which quantile of
# a model's distribution becomes the single value it publishes, and whether a
# larger or smaller value is the better one.
DECISION_METRICS: dict[str, str] = {
    "mean_excess": "min",
    "mae": "min",
    "rounded_accuracy": "max",
}
# r2 compares a band's residuals to that band's own variance, which binning
# deliberately shrinks, so it says nothing once the range is cut up.
UNBANDED_METRICS = frozenset({"r2"})
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
    lower: float | None = None
    upper: float | None = None


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


def _integer_metric(name: str, truth: list[float], predicted: list[float]) -> float:
    """Score an integer-valued target on the grid it is actually consumed on.

    A ladder rung is only ever acted on whole, so the decision a consumer makes
    is the rounded prediction, not the continuous estimate behind it. Mean
    absolute error scores the estimate; these score the decision, and they
    separate the two directions of being wrong, which for a mesh cost very
    different things.

    Halves round up, towards the denser mesh. That is a policy, not a law: a
    consumer is free to apply another one, and then these numbers describe a
    decision it is not making.
    """
    if any(value != math.floor(value) for value in truth):
        raise ValueError(f"{name} requires an integer-valued target")
    pairs = [
        (math.floor(estimate + 0.5), int(actual))
        for estimate, actual in zip(predicted, truth, strict=True)
    ]
    if name == "rounded_accuracy":
        return sum(guess == actual for guess, actual in pairs) / len(pairs)
    if name == "within_one":
        return sum(abs(guess - actual) <= 1 for guess, actual in pairs) / len(pairs)
    if name == "underprediction_rate":
        return sum(guess < actual for guess, actual in pairs) / len(pairs)
    if name == "mean_excess":
        # Signed on purpose: a negative mean says the model is systematically
        # recommending something coarser than the truth.
        return sum(guess - actual for guess, actual in pairs) / len(pairs)
    raise ValueError(f"unsupported integer-target metric: {name}")


def _metrics_by_bin(
    truth: list[float],
    predicted: list[float],
    intervals: list[tuple[float, float]] | None,
    metrics: Sequence[str],
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    """Score each band of the target's range separately.

    One number over a skewed target hides where a model fails: it can be right
    on average and systematically too coarse on the samples whose answers are
    the most expensive to get wrong. Bands cut on the *true* value, which makes
    this a diagnostic and not a guarantee a consumer could condition on -- at
    prediction time the true value is exactly what is missing.
    """
    ordered = list(edges)
    if not ordered or any(
        later <= earlier for earlier, later in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("coverage bins must be a non-empty increasing sequence")
    banded = [name for name in metrics if name not in UNBANDED_METRICS]
    bounds = [-math.inf, *ordered, math.inf]
    bands: list[dict[str, Any]] = []
    for low, high in zip(bounds, bounds[1:], strict=False):
        if low == -math.inf:
            label = f"<{high:g}"
        elif high == math.inf:
            label = f">={low:g}"
        else:
            label = f"[{low:g},{high:g})"
        rows = [index for index, actual in enumerate(truth) if low <= actual < high]
        band: dict[str, Any] = {"band": label, "count": len(rows)}
        if rows:
            band_truth = [truth[index] for index in rows]
            band_predicted = [predicted[index] for index in rows]
            for name in banded:
                band[name] = _regression_metric(name, band_truth, band_predicted)
            if intervals is not None:
                pairs = [intervals[index] for index in rows]
                band["interval_coverage"] = sum(
                    lower <= actual <= upper
                    for actual, (lower, upper) in zip(band_truth, pairs, strict=True)
                ) / len(rows)
                band["mean_interval_width"] = sum(
                    upper - lower for lower, upper in pairs
                ) / len(rows)
        bands.append(band)
    return bands


def select_decision_level(
    truth: Sequence[float],
    values_by_level: Mapping[float, Sequence[float]],
    *,
    metric: str,
    max_underprediction: float | None = None,
) -> dict[str, Any]:
    """Choose which quantile of a model's distribution becomes its one value.

    A regression model publishes a number, not a distribution, and which number
    is a modelling decision with a cost attached. Every symmetric metric --
    mae, rmse, the rounded hit rate -- rewards the level that sits in the
    middle of the distribution, because they price both directions of being
    wrong the same. Where a protocol does not, ``max_underprediction`` states
    the error it refuses to make and the metric then chooses among the levels
    that honour it. This is the regression counterpart of a recall floor.
    """
    if metric not in DECISION_METRICS:
        supported = ", ".join(sorted(DECISION_METRICS))
        raise ValueError(f"{metric} cannot choose a decision level; try {supported}")
    if not values_by_level:
        raise ValueError("decision selection needs at least one candidate level")
    if max_underprediction is not None and not 0.0 <= max_underprediction < 1.0:
        raise ValueError("max_underprediction must lie in [0, 1)")
    direction = DECISION_METRICS[metric]
    trials: list[dict[str, Any]] = []
    best: tuple[float, float] | None = None
    lowest_rate = math.inf
    for level in sorted(values_by_level):
        values = list(values_by_level[level])
        rate = _regression_metric("underprediction_rate", list(truth), values)
        score = _regression_metric(metric, list(truth), values)
        trials.append({"level": level, "underprediction_rate": rate, metric: score})
        lowest_rate = min(lowest_rate, rate)
        if max_underprediction is not None and rate > max_underprediction:
            continue
        better = best is None or (
            score < best[1] if direction == "min" else score > best[1]
        )
        if better:
            best = (level, score)
    if best is None:
        raise ValueError(
            f"no decision level keeps underprediction at or below "
            f"{max_underprediction}; the lowest available is {lowest_rate:.4f}"
        )
    return {
        "rule": "quantile",
        "level": best[0],
        "metric": metric,
        "max_underprediction": max_underprediction,
        metric: best[1],
        "trials": trials,
    }


def _regression_metric(name: str, truth: list[float], predicted: list[float]) -> float:
    if name in INTEGER_TARGET_METRICS:
        return _integer_metric(name, truth, predicted)
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


def pinball_loss(
    truth: Sequence[float], predicted: Sequence[float], quantile: float
) -> float:
    """Return the pinball loss of one quantile estimate.

    This is the proper scoring rule for quantile regression: it is minimised in
    expectation exactly when the prediction is the true conditional quantile.
    Mean absolute error only scores the median, so a model selected on it is
    selected on none of its interval behaviour.
    """
    if not 0 < quantile < 1:
        raise ValueError("a quantile must lie strictly between zero and one")
    errors = [
        actual - estimate for actual, estimate in zip(truth, predicted, strict=True)
    ]
    return sum(max(quantile * error, (quantile - 1) * error) for error in errors) / len(
        errors
    )


def _band_rate(rows: Sequence[tuple[float, int]], offset: int) -> float:
    return sum(guess + offset < actual for actual, guess in rows) / len(rows)


def select_band_offsets(
    truth: Sequence[float],
    values: Sequence[float],
    edges: Sequence[float],
    *,
    max_underprediction: float,
    max_offset: int = 8,
) -> list[dict[str, Any]]:
    """Return the whole-step lift each band needs to honour the floor.

    A single quantile honours a floor on average and still misses it inside the
    part of the range where the model is weakest. Bands cut on the model's own
    rounded value, not on the truth, because that is the only thing a consumer
    has at prediction time.

    Offsets only ever add. A band rule that *lowered* a value where the floor
    looked slack would be buying machine time with safety estimated on a finite
    sample, and the estimate is worst exactly where the samples are fewest.
    """
    if not 0.0 <= max_underprediction < 1.0:
        raise ValueError("max_underprediction must lie in [0, 1)")
    ordered = list(edges)
    if not ordered or any(
        later <= earlier for earlier, later in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("decision bands must be a non-empty increasing sequence")
    rounded = [math.floor(value + 0.5) for value in values]
    bounds = [-math.inf, *ordered, math.inf]
    bands: list[dict[str, Any]] = []
    for low, high in zip(bounds, bounds[1:], strict=False):
        rows = [
            (actual, guess)
            for actual, guess in zip(truth, rounded, strict=True)
            if low <= guess < high
        ]
        band: dict[str, Any] = {
            "upper": None if high == math.inf else high,
            "count": len(rows),
        }
        offset = 0
        if rows:
            rate = _band_rate(rows, offset)
            while rate > max_underprediction and offset < max_offset:
                offset += 1
                rate = _band_rate(rows, offset)
            band["underprediction_rate"] = rate
        band["offset"] = offset
        bands.append(band)
    return bands


def evaluate(
    task: str,
    predictions: Sequence[Prediction],
    metrics: Sequence[str],
    *,
    positive_label: str | None = None,
    quantiles: Sequence[float] | None = None,
    coverage_bins: Sequence[float] | None = None,
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
        banded_intervals: list[tuple[float, float]] | None = None
        bounded = [item for item in predictions if item.lower is not None]
        if bounded:
            if len(bounded) != len(predictions) or any(
                item.upper is None for item in predictions
            ):
                raise ValueError("regression intervals must be present for every row")
            intervals = [
                (float(item.lower), float(item.upper))
                for item in predictions
                if item.lower is not None and item.upper is not None
            ]
            if any(lower > upper for lower, upper in intervals):
                raise ValueError("regression interval lower bound exceeds upper bound")
            banded_intervals = intervals
            result["interval_coverage"] = sum(
                lower <= actual <= upper
                for actual, (lower, upper) in zip(truth, intervals, strict=True)
            ) / len(truth)
            result["mean_interval_width"] = sum(
                upper - lower for lower, upper in intervals
            ) / len(intervals)
            if quantiles is not None:
                if len(quantiles) != 3:
                    raise ValueError("three quantile levels are required")
                columns = (
                    [lower for lower, _ in intervals],
                    predicted,
                    [upper for _, upper in intervals],
                )
                losses = [
                    pinball_loss(truth, column, level)
                    for column, level in zip(columns, quantiles, strict=True)
                ]
                for level, loss in zip(quantiles, losses, strict=True):
                    result[f"pinball_loss_q{level:g}"] = loss
                result["pinball_loss"] = sum(losses) / len(losses)
        if coverage_bins is not None:
            result["metrics_by_bin"] = _metrics_by_bin(
                truth, predicted, banded_intervals, metrics, coverage_bins
            )
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
    *,
    min_recall: float | None = None,
) -> float:
    """Choose the decision threshold maximising one metric on a single split.

    A ``min_recall`` floor restricts the search to thresholds that miss no more
    of the positive class than the protocol accepts. Metrics such as MCC weigh
    both error directions equally; a floor states that this protocol does not,
    and it survives retraining in a way a hardcoded threshold does not.
    """
    if metric not in THRESHOLD_METRICS:
        raise ValueError(f"{metric} does not depend on a decision threshold")
    if min_recall is not None and not 0.0 < min_recall <= 1.0:
        raise ValueError("min_recall must lie in (0, 1]")
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
    best_threshold: float | None = None
    best_value = -math.inf
    best_recall = 0.0
    for threshold in candidates:
        predicted = [
            label_at(float(item.score), threshold, positive, negative)
            for item in predictions
            if item.score is not None
        ]
        if min_recall is not None:
            recall = _classification_metric("recall", truth, predicted, None, positive)
            best_recall = max(best_recall, recall)
            if recall < min_recall:
                continue
        value = _classification_metric(metric, truth, predicted, None, positive)
        if value > best_value:
            best_value = value
            best_threshold = threshold
    if best_threshold is None:
        raise ValueError(
            f"no decision threshold reaches a recall of {min_recall}; "
            f"the best available is {best_recall:.4f}"
        )
    return best_threshold
