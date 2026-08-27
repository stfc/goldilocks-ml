"""Tests for shared metrics, baselines, and threshold selection."""

from __future__ import annotations

import math

import pytest

from goldilocks_ml.datasets import Sample
from goldilocks_ml.evaluation import (
    Prediction,
    default_positive_label,
    evaluate,
    label_at,
    select_threshold,
    train_majority,
    train_median,
)


def _sample(sample_id: str, target: float | str) -> Sample:
    return Sample(sample_id=sample_id, target=target, group=None, features=())


def _regression(pairs: list[tuple[float, float]]) -> list[Prediction]:
    return [
        Prediction(sample_id=f"s{index}", truth=truth, prediction=predicted)
        for index, (truth, predicted) in enumerate(pairs)
    ]


def _classification(
    pairs: list[tuple[str, str, float]],
) -> list[Prediction]:
    return [
        Prediction(
            sample_id=f"s{index}", truth=truth, prediction=predicted, score=score
        )
        for index, (truth, predicted, score) in enumerate(pairs)
    ]


def test_regression_metrics_match_hand_computed_values() -> None:
    predictions = _regression([(1.0, 2.0), (2.0, 2.0), (3.0, 5.0)])

    result = evaluate("regression", predictions, ["mae", "rmse", "r2"])

    assert result["count"] == 3
    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert result["r2"] == pytest.approx(1.0 - 5 / 2)


def test_r2_is_rejected_when_the_target_is_constant() -> None:
    predictions = _regression([(2.0, 1.0), (2.0, 3.0)])

    with pytest.raises(ValueError, match="r2 is undefined"):
        evaluate("regression", predictions, ["r2"])


def test_evaluate_rejects_an_empty_split() -> None:
    with pytest.raises(ValueError, match="empty split"):
        evaluate("regression", [], ["mae"])


def test_classification_metrics_match_hand_computed_values() -> None:
    # Two true positives, one false positive, one false negative, one true negative.
    predictions = _classification(
        [
            ("metal", "metal", 0.9),
            ("metal", "metal", 0.8),
            ("insulator", "metal", 0.7),
            ("metal", "insulator", 0.2),
            ("insulator", "insulator", 0.1),
        ]
    )

    result = evaluate(
        "classification",
        predictions,
        ["accuracy", "precision", "recall", "f1", "mcc", "balanced_accuracy"],
        positive_label="metal",
    )

    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert result["mcc"] == pytest.approx(1 / 6)
    assert result["confusion_matrix"]["metal"]["metal"] == 2
    assert result["confusion_matrix"]["insulator"]["metal"] == 1


def test_roc_auc_is_one_for_a_perfect_ranking() -> None:
    predictions = _classification(
        [
            ("metal", "metal", 0.9),
            ("metal", "metal", 0.8),
            ("insulator", "insulator", 0.2),
            ("insulator", "insulator", 0.1),
        ]
    )

    result = evaluate(
        "classification", predictions, ["roc_auc", "pr_auc"], positive_label="metal"
    )

    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)


def test_roc_auc_is_one_half_for_constant_scores() -> None:
    predictions = _classification(
        [
            ("metal", "metal", 0.5),
            ("insulator", "metal", 0.5),
            ("metal", "metal", 0.5),
            ("insulator", "metal", 0.5),
        ]
    )

    result = evaluate(
        "classification", predictions, ["roc_auc", "pr_auc"], positive_label="metal"
    )

    assert result["roc_auc"] == pytest.approx(0.5)
    # Ties must never be resolved in the model's favour.
    assert result["pr_auc"] < 1.0


def test_ranking_metrics_need_both_classes() -> None:
    predictions = _classification([("metal", "metal", 0.9), ("metal", "metal", 0.8)])

    with pytest.raises(ValueError, match="both classes"):
        evaluate("classification", predictions, ["roc_auc"], positive_label="metal")


def test_binary_only_metrics_are_rejected_for_three_classes() -> None:
    predictions = _classification([("a", "a", 0.1), ("b", "b", 0.2), ("c", "c", 0.3)])

    with pytest.raises(ValueError, match="require a binary target"):
        evaluate("classification", predictions, ["mcc"], positive_label="a")


def test_accuracy_still_works_for_three_classes() -> None:
    predictions = _classification([("a", "a", 0.1), ("b", "c", 0.2), ("c", "c", 0.3)])

    result = evaluate("classification", predictions, ["accuracy"], positive_label="a")

    assert result["accuracy"] == pytest.approx(2 / 3)


def test_default_positive_label_is_the_last_sorted_class() -> None:
    assert default_positive_label(["metal", "insulator", "metal"]) == "metal"


def test_default_positive_label_needs_a_binary_target() -> None:
    with pytest.raises(ValueError, match="binary targets"):
        default_positive_label(["a", "b", "c"])


def test_train_median_uses_training_data_only() -> None:
    samples = [_sample("a", 1.0), _sample("b", 3.0), _sample("c", 10.0)]

    assert train_median(samples) == pytest.approx(3.0)


def test_train_majority_breaks_ties_deterministically() -> None:
    samples = [_sample("a", "metal"), _sample("b", "insulator")]

    label, frequency = train_majority(samples)

    assert label == "insulator"
    assert frequency == pytest.approx(0.5)


@pytest.mark.parametrize("baseline", [train_median, train_majority])
def test_baselines_reject_an_empty_train_split(baseline) -> None:
    with pytest.raises(ValueError, match="train split is empty"):
        baseline([])


def test_select_threshold_maximises_the_requested_metric() -> None:
    predictions = _classification(
        [
            ("insulator", "", 0.1),
            ("insulator", "", 0.4),
            ("metal", "", 0.6),
            ("metal", "", 0.9),
        ]
    )

    threshold = select_threshold(predictions, "mcc", "metal", "insulator")

    assert 0.4 < threshold <= 0.6
    labels = [
        label_at(float(item.score), threshold, "metal", "insulator")
        for item in predictions
    ]
    assert labels == ["insulator", "insulator", "metal", "metal"]


def test_select_threshold_rejects_ranking_metrics() -> None:
    predictions = _classification([("metal", "", 0.9), ("insulator", "", 0.1)])

    with pytest.raises(ValueError, match="does not depend on a decision threshold"):
        select_threshold(predictions, "roc_auc", "metal", "insulator")


def test_select_threshold_needs_scores() -> None:
    predictions = [Prediction(sample_id="a", truth="metal", prediction="metal")]

    with pytest.raises(ValueError, match="needs prediction scores"):
        select_threshold(predictions, "mcc", "metal", "insulator")
