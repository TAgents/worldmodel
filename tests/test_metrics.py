import json
import math

import pytest

from world_model.evaluation.metrics import losses, paired_interval, summarize


def test_proper_scores_match_hand_calculations():
    brier, log_loss = losses(0.75, True)
    assert brier == 0.0625
    assert log_loss == pytest.approx(-math.log(0.75))
    assert losses(0.25, False) == (brier, log_loss)
    report = summarize([0.75, 0.25], [True, False])
    assert report["mean_brier"] == brier
    assert report["mean_log_loss"] == log_loss


def test_impossible_predictions_keep_infinite_loss_without_invalid_json():
    assert losses(0.0, True) == (1, math.inf)
    assert losses(1.0, False) == (1, math.inf)
    assert losses(0.0, False) == (0, 0)
    assert losses(1.0, True) == (0, 0)
    report = summarize([0, 1], [True, False])
    assert report["infinite_log_loss_count"] == 2
    assert report["mean_log_loss"] is None
    json.dumps(report, allow_nan=False)


def test_calibration_bins_include_boundaries_and_empty_bins():
    report = summarize([0.0, 0.1, 0.9, 1.0], [False, True, False, True])
    bins = report["calibration"]
    assert [bucket["count"] for bucket in bins] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert bins[9]["mean_forecast"] == 0.95
    assert bins[9]["observed_frequency"] == 0.5
    assert bins[9]["upper_inclusive"]
    assert bins[2]["mean_forecast"] is None
    assert bins[2]["observed_frequency"] is None


def test_paired_bootstrap_preserves_known_episode_differences():
    constant = paired_interval([-0.2, -0.2, -0.2])
    assert constant == pytest.approx({"mean_difference": -0.2, "lower_95": -0.2, "upper_95": -0.2})
    varied = paired_interval([-0.5, 0.5])
    assert varied == {"mean_difference": 0.0, "lower_95": -0.5, "upper_95": 0.5}
    assert varied == paired_interval([-0.5, 0.5])


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf, True])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        losses(probability, True)


@pytest.mark.parametrize("differences", [[], [0.1], [0, math.inf], [math.nan, 0]])
def test_invalid_bootstrap_inputs_are_rejected(differences):
    with pytest.raises(ValueError):
        paired_interval(differences)


def test_invalid_metric_shapes_and_nonboolean_labels_are_rejected():
    with pytest.raises(ValueError):
        summarize([], [])
    with pytest.raises(ValueError):
        summarize([0.5], [True, False])
    with pytest.raises(ValueError):
        losses(0.5, 1)
