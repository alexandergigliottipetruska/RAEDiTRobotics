"""Evaluation metrics for binary success-rate experiments.

Provides:
- success_rate: fraction of successful episodes
- wilson_ci: 95% Wilson score confidence interval (preferred over normal
  approximation for rates near 0 or 1)
- aggregate_seeds: pool results across seeds and compute mean + CI
"""

import math
import numpy as np


def success_rate(outcomes: list[bool] | np.ndarray) -> float:
    """Compute success rate from a list of binary outcomes."""
    outcomes = np.asarray(outcomes, dtype=bool)
    if len(outcomes) == 0:
        return 0.0
    return float(np.mean(outcomes))


def wilson_ci(
    n_successes: int,
    n_total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Preferred over normal approximation when success rate is near 0 or 1.
    Formula: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval

    Args:
        n_successes: Number of successes.
        n_total: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if n_total == 0:
        return (0.0, 0.0)

    # z-score for the confidence level (two-tailed)
    # For 95%: z ≈ 1.96
    alpha = 1.0 - confidence
    # Use inverse normal approximation: z = Φ^{-1}(1 - α/2)
    # For common values, this is precise enough without scipy
    z = _probit(1.0 - alpha / 2.0)

    p_hat = n_successes / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    centre = (p_hat + z2 / (2.0 * n_total)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / n_total + z2 / (4.0 * n_total * n_total)
    )

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (lower, upper)


def aggregate_seeds(
    per_seed_outcomes: list[list[bool]],
    confidence: float = 0.95,
) -> dict:
    """Pool binary outcomes across seeds and compute summary statistics.

    Args:
        per_seed_outcomes: List of outcome lists, one per seed.
        confidence: Confidence level for Wilson CI.

    Returns:
        Dict with keys: success_rate, ci_lower, ci_upper, n_successes,
        n_total, per_seed_rates.
    """
    all_outcomes = []
    per_seed_rates = []
    for outcomes in per_seed_outcomes:
        arr = np.asarray(outcomes, dtype=bool)
        per_seed_rates.append(float(np.mean(arr)) if len(arr) > 0 else 0.0)
        all_outcomes.extend(arr.tolist())

    n_total = len(all_outcomes)
    n_successes = sum(all_outcomes)
    sr = n_successes / n_total if n_total > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(n_successes, n_total, confidence)

    return {
        "success_rate": sr,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_successes": n_successes,
        "n_total": n_total,
        "per_seed_rates": per_seed_rates,
    }


# ---------------------------------------------------------------------------
# Internal: rational approximation of the probit (inverse normal CDF)
# ---------------------------------------------------------------------------

def _probit(p: float) -> float:
    """Approximate inverse of the standard normal CDF.

    Uses the Abramowitz & Stegun rational approximation (formula 26.2.23).
    Accurate to ~4.5e-4 absolute error, which is more than sufficient
    for confidence interval computation.
    """
    assert 0.0 < p < 1.0, f"p must be in (0, 1), got {p}"

    if p < 0.5:
        return -_probit(1.0 - p)

    # Coefficients for the rational approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return z
