"""Proper predictive scores, calibration counts, and paired episode bootstrap."""

import math
import random
from collections.abc import Sequence

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261027


def losses(probability: float, outcome: bool) -> tuple[float, float]:
    if isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("forecast probability must be finite and between zero and one")
    if type(outcome) is not bool:
        raise ValueError("outcome must be a boolean")
    brier = (probability - int(outcome)) ** 2
    if outcome:
        log_loss = -math.log(probability) if probability else math.inf
    else:
        log_loss = -math.log1p(-probability) if probability < 1 else math.inf
    return brier, log_loss


def summarize(probabilities: Sequence[float], outcomes: Sequence[bool]) -> dict:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("forecasts and outcomes must be nonempty and equally sized")
    scores = [losses(p, y) for p, y in zip(probabilities, outcomes, strict=True)]
    infinite_count = sum(math.isinf(score[1]) for score in scores)
    calibration = []
    for index in range(10):
        selected = [(p, y) for p, y in zip(probabilities, outcomes, strict=True) if min(9, int(p * 10)) == index]
        count = len(selected)
        calibration.append({
            "lower": index / 10, "upper": (index + 1) / 10,
            "upper_inclusive": index == 9, "count": count,
            "mean_forecast": math.fsum(p for p, _ in selected) / count if count else None,
            "observed_frequency": sum(y for _, y in selected) / count if count else None,
        })
    return {
        "forecast_count": len(scores),
        "mean_brier": math.fsum(score[0] for score in scores) / len(scores),
        "mean_log_loss": None if infinite_count else math.fsum(score[1] for score in scores) / len(scores),
        "infinite_log_loss_count": infinite_count,
        "calibration": calibration,
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_interval(
    differences: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Resample paired per-episode means, retaining temporal dependence within each."""
    if len(differences) < 2 or any(not math.isfinite(value) for value in differences):
        raise ValueError("bootstrap needs at least two finite episode differences")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("bootstrap needs at least two resamples")
    rng = random.Random(seed)
    count = len(differences)
    draws = sorted(
        math.fsum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    return {
        "mean_difference": math.fsum(differences) / count,
        "lower_95": _quantile(draws, 0.025), "upper_95": _quantile(draws, 0.975),
    }
