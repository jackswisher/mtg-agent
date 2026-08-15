"""Statistical primitives for evaluation aggregation.

All functions are pure NumPy / SciPy and are unit-test friendly. They are used
by ``scripts/research/aggregate.py`` to produce comparison reports.

Conventions
-----------
- Significance level ``alpha`` defaults to ``0.05`` (95% intervals).
- Paired tests assume the two arrays are aligned by seed/episode/etc.
- Bootstrap functions accept ``rng`` for deterministic results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CIResult:
    """Result of a confidence-interval computation."""

    mean: float
    lo: float
    hi: float
    n: int

    @property
    def margin(self) -> float:
        """Half-width of the symmetric CI (max distance from mean to lo/hi)."""
        return max(self.mean - self.lo, self.hi - self.mean)

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable representation."""
        return {"mean": self.mean, "lo": self.lo, "hi": self.hi, "n": self.n}


@dataclass(frozen=True)
class TestResult:
    """Result of a hypothesis test or paired-comparison bootstrap."""

    name: str
    statistic: float
    p_value: float
    mean_diff: float
    ci_low: float
    ci_high: float
    n: int

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
        }


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> CIResult:
    """Compute a Wilson score interval for a binomial proportion.

    Wilson is preferred over the normal approximation because it has correct
    coverage even for small ``n`` and proportions near 0 or 1.

    Args:
        successes: Number of successes (e.g. wins).
        n: Total trials.
        alpha: Significance level (default 0.05 -> 95% CI).

    Returns:
        :class:`CIResult` containing mean (point estimate), lo, hi, n.

    """
    if n <= 0:
        return CIResult(mean=0.0, lo=0.0, hi=0.0, n=0)
    p = successes / n
    z = float(stats.norm.ppf(1 - alpha / 2))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return CIResult(mean=p, lo=max(0.0, centre - half), hi=min(1.0, centre + half), n=n)


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CIResult:
    """Percentile bootstrap CI for the mean.

    Args:
        values: 1-D array-like of observations.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 -> 95% CI).
        rng: Optional NumPy generator for reproducibility.

    Returns:
        :class:`CIResult` with mean, percentile lo/hi, and sample size.

    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return CIResult(mean=0.0, lo=0.0, hi=0.0, n=0)
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return CIResult(mean=float(arr.mean()), lo=lo, hi=hi, n=int(n))


def paired_bootstrap_test(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Paired bootstrap test on the mean difference ``a - b``.

    The two arrays must be aligned (e.g. same seeds / same episodes). The
    p-value is two-sided and computed via the bootstrap distribution of the
    paired mean difference, comparing against zero.

    Args:
        a: Observations from method A.
        b: Observations from method B (same shape as ``a``).
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (default 0.05).
        rng: Optional NumPy generator for reproducibility.

    Returns:
        :class:`TestResult` with mean diff, percentile CI on diff, and p-value.

    Raises:
        ValueError: If ``a`` and ``b`` have different lengths.

    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: a={arr_a.shape}, b={arr_b.shape}")
    diff = arr_a - arr_b
    n = diff.size
    if n == 0:
        return TestResult("paired_bootstrap", 0.0, 1.0, 0.0, 0.0, 0.0, 0)
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx].mean(axis=1)
    mean_diff = float(diff.mean())
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))
    # Two-sided p-value: probability that the bootstrap mean is at least
    # as extreme (in either direction) as 0 under H0: mean_diff = 0.
    centred = boot_means - mean_diff
    p_value = float(np.mean(np.abs(centred) >= abs(mean_diff)))
    p_value = max(p_value, 1.0 / n_resamples)
    return TestResult(
        name="paired_bootstrap",
        statistic=mean_diff,
        p_value=p_value,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
    )


def _welch_satterthwaite_df(var_a: float, var_b: float, n_a: int, n_b: int) -> float:
    """Welch-Satterthwaite degrees-of-freedom for two-sample unequal-variance t.

    Returns ``1.0`` whenever the formula would otherwise be undefined
    (zero variance in both samples, or fewer than two observations on
    one side). Callers should use this only when at least one variance
    estimate is finite and positive.
    """
    if n_a < 2 and n_b < 2:
        return 1.0
    se_a_sq = var_a / max(1, n_a)
    se_b_sq = var_b / max(1, n_b)
    num = (se_a_sq + se_b_sq) ** 2
    denom = 0.0
    if n_a > 1:
        denom += se_a_sq**2 / (n_a - 1)
    if n_b > 1:
        denom += se_b_sq**2 / (n_b - 1)
    if denom <= 0.0 or not math.isfinite(num) or not math.isfinite(denom):
        return 1.0
    return max(1.0, num / denom)


def welch_ttest(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
) -> TestResult:
    """Welch's t-test (unequal variance, unpaired) for ``a`` vs ``b``.

    The CI on ``mean_diff`` uses the Welch-Satterthwaite degrees of
    freedom so it is consistent with the test's p-value. Using the
    pooled ``n_a + n_b - 2`` df here would silently widen or shrink the
    interval relative to what the Welch test actually covers.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.size == 0 or arr_b.size == 0:
        return TestResult("welch_t", 0.0, 1.0, 0.0, 0.0, 0.0, 0)
    res = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    mean_diff = float(arr_a.mean() - arr_b.mean())
    var_a = float(arr_a.var(ddof=1)) if arr_a.size > 1 else 0.0
    var_b = float(arr_b.var(ddof=1)) if arr_b.size > 1 else 0.0
    se = math.sqrt(var_a / max(1, arr_a.size) + var_b / max(1, arr_b.size))
    df = _welch_satterthwaite_df(var_a, var_b, int(arr_a.size), int(arr_b.size))
    tcrit = float(stats.t.ppf(0.975, df))
    return TestResult(
        name="welch_t",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        mean_diff=mean_diff,
        ci_low=mean_diff - tcrit * se,
        ci_high=mean_diff + tcrit * se,
        n=int(arr_a.size + arr_b.size),
    )


def _hodges_lehmann_estimate(diff: np.ndarray) -> float:
    """Hodges-Lehmann estimator: median of all Walsh averages of ``diff``.

    Walsh averages are ``(d_i + d_j) / 2`` over all ``i <= j``. This is
    the location estimator paired with the Wilcoxon signed-rank test.
    """
    if diff.size == 0:
        return 0.0
    walsh = (diff[:, None] + diff[None, :]) / 2.0
    iu = np.triu_indices(diff.size)
    return float(np.median(walsh[iu]))


def _hodges_lehmann_ci(
    diff: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 5_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI on the Hodges-Lehmann location estimator of ``diff``.

    The exact rank-based interval is O(n log n) in scipy but is not
    available for arbitrary tied data; resampling the Walsh-average
    median gives a valid distribution-free CI that is consistent with
    the Wilcoxon test's location interpretation.
    """
    n = diff.size
    if n == 0:
        return 0.0, 0.0
    alpha = (1.0 - confidence) / 2.0
    boot = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        sample = diff[rng.integers(0, n, size=n)]
        boot[k] = _hodges_lehmann_estimate(sample)
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    return lo, hi


def wilcoxon_signed_rank(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    rng: np.random.Generator | None = None,
    n_resamples: int = 5_000,
) -> TestResult:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric; recommended when the difference distribution is
    non-normal or when seed-level results are available but n is very
    small. The reported ``ci_low``/``ci_high`` cover the
    Hodges-Lehmann location estimator (median of Walsh averages of the
    paired differences) via bootstrap, which is the location target
    paired with the Wilcoxon test rather than empirical quantiles of
    the raw differences.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: a={arr_a.shape}, b={arr_b.shape}")
    diff = arr_a - arr_b
    if diff.size == 0 or np.all(diff == 0):
        return TestResult("wilcoxon", 0.0, 1.0, 0.0, 0.0, 0.0, int(diff.size))
    res = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox", correction=False)
    hl = _hodges_lehmann_estimate(diff)
    rng = rng or np.random.default_rng(0)
    ci_low, ci_high = _hodges_lehmann_ci(diff, rng, n_resamples=n_resamples)
    return TestResult(
        name="wilcoxon",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        mean_diff=hl,
        ci_low=ci_low,
        ci_high=ci_high,
        n=int(diff.size),
    )


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Algorithm: sort p-values ascending, multiply the i-th smallest by ``(n-i)``,
    take a cumulative max so adjusted p-values are monotone, clip to 1.0,
    and return in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []
    p = np.array(p_values, dtype=float)
    order = np.argsort(p)
    sorted_p = p[order]
    multipliers = (n - np.arange(n)).astype(float)
    adjusted_sorted = np.maximum.accumulate(sorted_p * multipliers)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted_sorted
    return out.tolist()


__all__ = [
    "CIResult",
    "TestResult",
    "bootstrap_mean_ci",
    "holm_bonferroni",
    "paired_bootstrap_test",
    "welch_ttest",
    "wilcoxon_signed_rank",
    "wilson_ci",
]
