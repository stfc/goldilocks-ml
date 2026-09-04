"""Tests for shared metrics, baselines, and threshold selection."""

from __future__ import annotations

import math

import pytest

from goldilocks_ml.evaluation import (
    DECISION_METRICS,
    Prediction,
    default_positive_label,
    evaluate,
    label_at,
    select_band_offsets,
    select_decision_level,
    select_threshold,
    train_majority,
    train_median,
)
from goldilocks_ml.protocol import DECISION_METRICS as PROTOCOL_DECISION_METRICS
from goldilocks_ml.snapshot import Sample


def _sample(sample_id: str, target: float | str) -> Sample:
    return Sample(sample_id=sample_id, target=target, group=None, structure_path=None)


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


def test_regression_interval_metrics_are_reported() -> None:
    predictions = [
        Prediction("a", 1.0, 1.1, lower=0.5, upper=1.5),
        Prediction("b", 3.0, 2.8, lower=2.0, upper=2.9),
    ]

    result = evaluate("regression", predictions, ["mae"])

    assert result["interval_coverage"] == pytest.approx(0.5)
    assert result["mean_interval_width"] == pytest.approx(0.95)


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


def test_select_threshold_honours_a_recall_floor() -> None:
    predictions = _classification(
        [
            ("insulator", "", 0.1),
            ("metal", "", 0.2),
            ("insulator", "", 0.4),
            ("metal", "", 0.6),
            ("metal", "", 0.9),
        ]
    )

    unconstrained = select_threshold(predictions, "mcc", "metal", "insulator")
    floored = select_threshold(predictions, "mcc", "metal", "insulator", min_recall=1.0)

    def recall(threshold: float) -> float:
        caught = sum(
            1
            for item in predictions
            if item.truth == "metal" and float(item.score) >= threshold
        )
        return caught / 3

    # MCC alone leaves the 0.2 metal behind; the floor moves the line below it.
    assert recall(unconstrained) < 1.0
    assert recall(floored) == 1.0
    assert floored < unconstrained


def test_select_threshold_reports_an_unreachable_recall_floor() -> None:
    predictions = _classification([("insulator", "", 0.9), ("insulator", "", 0.1)])

    with pytest.raises(ValueError, match="no decision threshold reaches a recall"):
        select_threshold(predictions, "mcc", "metal", "insulator", min_recall=0.9)


def test_select_threshold_rejects_an_out_of_range_recall_floor() -> None:
    predictions = _classification([("metal", "", 0.9), ("insulator", "", 0.1)])

    with pytest.raises(ValueError, match=r"min_recall must lie in \(0, 1\]"):
        select_threshold(predictions, "mcc", "metal", "insulator", min_recall=1.5)


def test_select_threshold_rejects_ranking_metrics() -> None:
    predictions = _classification([("metal", "", 0.9), ("insulator", "", 0.1)])

    with pytest.raises(ValueError, match="does not depend on a decision threshold"):
        select_threshold(predictions, "roc_auc", "metal", "insulator")


def test_select_threshold_needs_scores() -> None:
    predictions = [Prediction(sample_id="a", truth="metal", prediction="metal")]

    with pytest.raises(ValueError, match="needs prediction scores"):
        select_threshold(predictions, "mcc", "metal", "insulator")


def test_integer_target_metrics_score_the_rounded_decision() -> None:
    predictions = _regression([(2, 2.4), (2, 2.5), (5, 3.0), (1, 0.6), (4, 6.2)])

    result = evaluate(
        "regression",
        predictions,
        ["rounded_accuracy", "within_one", "underprediction_rate"],
    )

    # Rounded: 2, 3, 3, 1, 6 against truths 2, 2, 5, 1, 4. The second row pins
    # the halves-round-up policy: 2.5 is scored as the denser mesh, not as a hit.
    assert result["rounded_accuracy"] == pytest.approx(2 / 5)
    assert result["within_one"] == pytest.approx(3 / 5)
    assert result["underprediction_rate"] == pytest.approx(1 / 5)


def test_integer_target_metrics_refuse_a_continuous_target() -> None:
    predictions = _regression([(0.35, 0.4), (0.21, 0.2)])

    with pytest.raises(ValueError, match="integer-valued target"):
        evaluate("regression", predictions, ["rounded_accuracy"])


def test_metrics_by_bin_shows_where_the_model_is_systematically_wrong() -> None:
    predictions = [
        Prediction("a", 1.0, 1.0, lower=0.0, upper=2.0),
        Prediction("b", 2.0, 2.0, lower=1.5, upper=2.5),
        Prediction("c", 4.0, 4.0, lower=3.0, upper=5.0),
        Prediction("d", 5.0, 5.0, lower=4.5, upper=5.5),
        Prediction("e", 8.0, 6.0, lower=5.0, upper=7.0),
        Prediction("f", 9.0, 6.0, lower=5.0, upper=7.0),
    ]

    result = evaluate(
        "regression",
        predictions,
        ["mae", "r2", "mean_excess", "underprediction_rate"],
        coverage_bins=[3, 6],
    )

    bands = result["metrics_by_bin"]
    assert [band["band"] for band in bands] == ["<3", "[3,6)", ">=6"]
    assert [band["count"] for band in bands] == [2, 2, 2]
    # Overall the model looks two thirds covered and mildly wrong. The bands
    # say it is exactly right below 6 and always too coarse above it.
    assert result["interval_coverage"] == pytest.approx(4 / 6)
    assert [band["mae"] for band in bands] == pytest.approx([0.0, 0.0, 2.5])
    assert bands[2]["mean_excess"] == pytest.approx(-2.5)
    assert bands[2]["underprediction_rate"] == pytest.approx(1.0)
    assert bands[2]["interval_coverage"] == pytest.approx(0.0)
    assert "r2" not in bands[0]


def test_metrics_by_bin_leaves_an_empty_band_unscored() -> None:
    predictions = [Prediction("a", 1.0, 1.0, lower=0.0, upper=2.0)]

    result = evaluate("regression", predictions, ["mae"], coverage_bins=[3])

    bands = result["metrics_by_bin"]
    assert [band["count"] for band in bands] == [1, 0]
    assert "mae" not in bands[1]


def test_mean_excess_signs_the_direction_of_the_error() -> None:
    predictions = _regression([(5, 3.0), (5, 8.0), (5, 5.0)])

    result = evaluate("regression", predictions, ["mean_excess", "mae"])

    # Errors of -2 and +3 cancel to +1/3 here, while mae reports 5/3. One says
    # which way the model leans; the other says how far off it is.
    assert result["mean_excess"] == pytest.approx(1 / 3)
    assert result["mae"] == pytest.approx(5 / 3)


def _levels() -> dict[float, list[float]]:
    return {0.5: [2, 2, 3, 3], 0.9: [3, 3, 5, 5], 0.95: [4, 4, 6, 6]}


def test_a_decision_level_without_a_floor_picks_the_coarsest() -> None:
    """Left unconstrained, mean_excess is smallest where the model under-calls."""
    chosen = select_decision_level([2, 2, 5, 5], _levels(), metric="mean_excess")

    assert chosen["level"] == 0.5
    assert chosen["mean_excess"] == pytest.approx(-1.0)


def test_the_underprediction_floor_moves_the_decision_level() -> None:
    chosen = select_decision_level(
        [2, 2, 5, 5], _levels(), metric="mean_excess", max_underprediction=0.1
    )

    # 0.5 misses the floor at a rate of 0.5; of the two that honour it, 0.9 is
    # the cheaper, at half a rung of deliberate excess.
    assert chosen["level"] == 0.9
    assert chosen["mean_excess"] == pytest.approx(0.5)
    assert chosen["max_underprediction"] == 0.1
    assert [trial["level"] for trial in chosen["trials"]] == [0.5, 0.9, 0.95]


def test_an_unreachable_floor_reports_the_best_available() -> None:
    with pytest.raises(ValueError, match="the lowest available is 0.5000"):
        select_decision_level(
            [2, 2, 5, 5],
            {0.5: [2, 2, 3, 3]},
            metric="mean_excess",
            max_underprediction=0.1,
        )


def test_a_metric_that_cannot_choose_a_level_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot choose a decision level"):
        select_decision_level([2, 2], {0.5: [2, 2]}, metric="rmse")


def test_the_two_decision_metric_registries_agree() -> None:
    """protocol.py cannot import evaluation.py, so a test keeps them in step."""
    assert set(DECISION_METRICS) == PROTOCOL_DECISION_METRICS


def test_coverage_bins_must_increase() -> None:
    predictions = [Prediction("a", 1.0, 1.0, lower=0.0, upper=2.0)]

    with pytest.raises(ValueError, match="increasing"):
        evaluate("regression", predictions, ["mae"], coverage_bins=[6, 3])


def test_band_offsets_lift_only_the_band_that_misses_the_floor() -> None:
    """Bands cut on the model's own rung, and a band that is fine is left alone."""
    truth = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 13, 14, 15, 16, 12, 12, 12, 12, 12]
    values = [1.0] * 10 + [11.0] * 10

    bands = select_band_offsets(truth, values, [6], max_underprediction=0.1)

    # The low band never comes in under the truth, so it is not touched. The
    # high band sits at rung 11 against truths of 12 to 16, and has to climb
    # four whole steps before at most one in ten is still short.
    assert [band["upper"] for band in bands] == [6.0, None]
    assert bands[0]["offset"] == 0
    assert bands[1]["offset"] == 4
    assert bands[1]["underprediction_rate"] <= 0.1
    assert [band["count"] for band in bands] == [10, 10]


def test_a_band_offset_never_lowers_a_value() -> None:
    """Slack in a band is not spent, because the estimate of it is finite."""
    truth = [1] * 20
    values = [8.0] * 20

    bands = select_band_offsets(truth, values, [6], max_underprediction=0.05)

    assert [band["offset"] for band in bands] == [0, 0]


def test_band_offsets_stop_at_the_ceiling() -> None:
    """An unreachable floor gives the largest allowed lift, not a hang."""
    truth = [40] * 10
    values = [1.0] * 10

    bands = select_band_offsets(
        truth, values, [6], max_underprediction=0.0, max_offset=3
    )

    assert bands[0]["offset"] == 3
    assert bands[0]["underprediction_rate"] == pytest.approx(1.0)
