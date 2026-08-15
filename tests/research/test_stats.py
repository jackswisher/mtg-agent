"""Tests for ``scripts/research/stats.py``.

Covers the small numerical primitives that the rest of the research
pipeline depends on:

* ``welch_ttest`` reports its CI under the Welch-Satterthwaite degrees
  of freedom that match its p-value.
* ``wilcoxon_signed_rank`` reports the Hodges-Lehmann location
  estimator and a bootstrap CI on it (rather than empirical quantiles
  of the raw paired differences).
* ``holm_bonferroni`` and ``paired_bootstrap_test`` keep the
  invariants the aggregator relies on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.research.stats import (
    _hodges_lehmann_estimate,
    _welch_satterthwaite_df,
    bootstrap_mean_ci,
    holm_bonferroni,
    paired_bootstrap_test,
    welch_ttest,
    wilcoxon_signed_rank,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Welch's t-test: CI uses Satterthwaite df, not pooled df
# ---------------------------------------------------------------------------


def test_welch_satterthwaite_df_matches_textbook_value() -> None:
    """Closed-form Satterthwaite df agrees with the standard formula."""
    var_a, var_b = 4.0, 16.0
    n_a, n_b = 10, 20
    se_a_sq = var_a / n_a
    se_b_sq = var_b / n_b
    expected = (se_a_sq + se_b_sq) ** 2 / (se_a_sq**2 / (n_a - 1) + se_b_sq**2 / (n_b - 1))
    assert _welch_satterthwaite_df(var_a, var_b, n_a, n_b) == pytest.approx(expected)


def test_welch_satterthwaite_df_handles_degenerate_inputs() -> None:
    """Zero variance + tiny samples must not blow up; df clamps at 1.0."""
    assert _welch_satterthwaite_df(0.0, 0.0, 1, 1) == 1.0


def test_welch_ttest_ci_uses_satterthwaite_df_not_pooled() -> None:
    """The Welch CI half-width uses Satterthwaite df, not ``n_a + n_b - 2``.

    For markedly unequal variances the two tcrits differ enough that
    swapping df materially shifts the half-width, so this regression
    test catches the previous mistake.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=1.0, size=10)
    b = rng.normal(loc=0.5, scale=5.0, size=40)
    res = welch_ttest(a, b)
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    df = _welch_satterthwaite_df(var_a, var_b, 10, 40)
    se = math.sqrt(var_a / 10 + var_b / 40)
    from scipy.stats import t as student_t

    tcrit = float(student_t.ppf(0.975, df))
    expected_half_width = tcrit * se
    half_width = (res.ci_high - res.ci_low) / 2
    assert half_width == pytest.approx(expected_half_width, rel=1e-9)


# ---------------------------------------------------------------------------
# Wilcoxon: Hodges-Lehmann location estimate + bootstrap CI
# ---------------------------------------------------------------------------


def test_hodges_lehmann_estimate_known_case() -> None:
    """HL of (1, 3, 5) is 3.0 (median of Walsh averages 1, 2, 3, 3, 4, 5)."""
    diff = np.array([1.0, 3.0, 5.0])
    assert _hodges_lehmann_estimate(diff) == pytest.approx(3.0)


def test_wilcoxon_reports_hodges_lehmann_not_mean_diff() -> None:
    """``mean_diff`` field carries the HL estimator (robust to outliers)."""
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.5, scale=1.0, size=20)
    b = rng.normal(loc=0.0, scale=1.0, size=20)
    a[0] = 100.0  # massive outlier should NOT inflate the HL estimator
    res = wilcoxon_signed_rank(a, b)
    diff = a - b
    hl = _hodges_lehmann_estimate(diff)
    assert res.mean_diff == pytest.approx(hl)
    # The HL estimator should be much less than the raw mean difference,
    # because the latter is dragged by the outlier.
    assert abs(res.mean_diff) < abs(diff.mean())


def test_wilcoxon_bootstrap_ci_brackets_location_estimate() -> None:
    """The reported CI must contain the Hodges-Lehmann point estimate."""
    rng = np.random.default_rng(2)
    a = rng.normal(loc=0.5, scale=1.0, size=15)
    b = rng.normal(loc=0.0, scale=1.0, size=15)
    res = wilcoxon_signed_rank(a, b, rng=np.random.default_rng(2), n_resamples=2_000)
    assert res.ci_low <= res.mean_diff <= res.ci_high


# ---------------------------------------------------------------------------
# Sanity for the rest of the public surface
# ---------------------------------------------------------------------------


def test_paired_bootstrap_pvalue_is_strictly_positive() -> None:
    """``paired_bootstrap_test`` must never return a literal-zero p-value."""
    rng = np.random.default_rng(3)
    a = rng.normal(loc=0.0, size=12)
    b = a + rng.normal(loc=10.0, size=12)  # huge effect: p should be tiny but >0
    res = paired_bootstrap_test(a, b, n_resamples=1_000)
    assert res.p_value > 0.0
    assert res.ci_low <= res.mean_diff <= res.ci_high


def test_holm_bonferroni_is_monotone_and_capped_at_one() -> None:
    """Holm-corrected p-values must be monotone in the original ranking and bounded."""
    p = [0.01, 0.04, 0.03, 0.5, 0.001]
    adj = holm_bonferroni(p)
    assert all(0.0 <= q <= 1.0 for q in adj)
    sorted_pairs = sorted(zip(p, adj, strict=False), key=lambda x: x[0])
    sorted_adj = [q for _, q in sorted_pairs]
    for prev, curr in zip(sorted_adj, sorted_adj[1:], strict=False):
        assert curr >= prev - 1e-12  # monotone in sorted-p order


def test_wilson_ci_is_strictly_inside_unit_interval() -> None:
    """Wilson CI must lie in (0, 1) for any 0 < successes < n."""
    ci = wilson_ci(7, 10, alpha=0.05)
    assert 0.0 < ci.lo < ci.mean < ci.hi < 1.0


def test_bootstrap_mean_ci_is_deterministic_with_default_rng() -> None:
    """Default RNG seed=0 must produce the same CI on repeated calls."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a = bootstrap_mean_ci(data, n_resamples=2_000)
    b = bootstrap_mean_ci(data, n_resamples=2_000)
    assert a.lo == pytest.approx(b.lo)
    assert a.hi == pytest.approx(b.hi)
