"""Fractional differentiation (Phase 4).

Goal: make the series sufficiently stationary while retaining as much predictive memory as
possible (Lopez de Prado, "Advances in Financial Machine Learning", ch. 5).

CRITICAL discipline:
  d is NEVER selected once on the full dataset. For every training fold:
    1. fit/derive the transformation using TRAIN data only
    2. search candidate d in {0.0, 0.05, ..., 1.0}
    3. evaluate stationarity (ADF) + information retention
    4. select the minimum acceptable d
    5. freeze d (and the weight series, which depends only on d and the data length)
    6. apply to validation/test WITHOUT refitting on future observations

Implementation notes:
  * Weight series w_k = (-1)^k * C(d, k) with C(d,k) = prod_{i=1..k} (d - i + 1)/i.
  * Standard fracdiff expands the weights over the WHOLE series length, which makes every
    transformed point a function of ALL history (including the test set for test points).
    That is fine for point-in-time values IF weights are frozen from train, but it couples
    train/test windows. The implementation below therefore supports a fixed, finite window of
    weights (``window``) which is the causal, memory-bounded form used in the feature engine:
    each transformed value uses at most ``window`` prior observations. The ADF diagnostics
    used for d-selection are computed on the same bounded transform.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from statsmodels.tsa.stattools import adfuller, kpss

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - statsmodels optional
    _HAS_STATSMODELS = False

# KPSS p-values outside the lookup table range are common for strongly stationary /
# non-stationary series; the returned (floored) p is conservative for our decision rule.
warnings.filterwarnings("ignore", message="The test statistic is outside of the range.*")

D_CANDIDATES = np.round(np.arange(0.0, 1.01, 0.05), 2).tolist()


def fracdiff_weights(d: float, window: int) -> np.ndarray:
    """Finite-memory fractional differentiation weights of length ``window``.

    w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k   for k = 1..window-1
    (expanded from the binomial series so that the transform is
     (1 - L)^d applied over a trailing window).
    """
    w = np.ones(window)
    for k in range(1, window):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def fracdiff_series(
    s: Series, d: float, window: int = 256, min_periods: int | None = None
) -> Series:
    """Causal finite-window fractional differentiation of a series.

    out[t] = sum_{k=0}^{min(W-1, t)} w_k * s[t-k],   with w from fracdiff_weights(d, W).

    Every reported value uses ONLY trailing observations (bounded memory of W bars).
    Rows with fewer than ``min_periods`` real lookback observations are NaN (warm-up mask),
    so no zero-padding bias ever enters a reported value.
    """
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"d must be in [0, 1], got {d}")
    if d == 0.0:
        return s.copy()
    w = fracdiff_weights(d, window)
    min_periods = min_periods if min_periods is not None else window
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for t in range(min_periods - 1, n):
        # lookback window [t-W+1 .. t] clipped to series start
        lo = max(0, t - window + 1)
        # newest observation (at t) carries w[0]; w is applied newest->oldest
        seg = vals[t : lo - 1 : -1] if lo > 0 else vals[t::-1]
        out[t] = float(np.dot(w[: len(seg)], seg))
    return Series(out, index=s.index, name=s.name)


def fracdiff_returns(
    df: DataFrame, d: float, window: int = 256, min_periods: int | None = None
) -> Series:
    """Fractional differentiation of log-price (equivalent to FD of returns at d-1).

    Returns the FD series named 'fd_price'. For a feature engine, differentiating log price
    with d in [0,1) yields an approximately stationary series retaining memory.
    """
    lp = np.log(df["close"])
    return fracdiff_series(lp, d, window=window, min_periods=min_periods).rename("fd_price")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class FDResult:
    d: float
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    information_retention: float  # corr(original, fd) on the overlapping sample
    effective_sample_size: int
    autocorr_lag1: float
    stationarity_verdict: str  # "stationary" | "not_stationary" | "borderline" | "no_statsmodels"


def _adf(s: Series) -> tuple[float, float]:
    if not _HAS_STATSMODELS:
        return np.nan, np.nan
    s = s.dropna()
    if len(s) < 30:
        return np.nan, np.nan
    try:
        res = adfuller(s, autolag="AIC", maxlag=20, result_object=True)
        return float(res[0]), float(res[1])
    except Exception:
        return np.nan, np.nan


def _kpss(s: Series) -> tuple[float, float]:
    if not _HAS_STATSMODELS:
        return np.nan, np.nan
    s = s.dropna()
    if len(s) < 30:
        return np.nan, np.nan
    try:
        res = kpss(s, regression="c", nlags="auto", result_object=True)
        # p-value is floored at the lookup table boundary; report conservatively
        return float(res[0]), float(res[1])
    except Exception:
        return np.nan, np.nan


def evaluate_d(
    s: Series,
    d: float,
    window: int = 256,
    adf_alpha: float = 0.05,
    kpss_alpha: float = 0.05,
) -> FDResult:
    """Evaluate one candidate d on a series (TRAIN-ONLY usage)."""
    fd = fracdiff_series(s, d, window=window).dropna()
    eff_n = len(fd)
    if eff_n < 50:
        return FDResult(d, np.nan, np.nan, np.nan, np.nan, np.nan, eff_n, np.nan, "insufficient")
    aligned = s.loc[fd.index]
    info_ret = float(np.corrcoef(aligned.to_numpy(), fd.to_numpy())[0, 1]) if eff_n > 2 else np.nan
    ac1 = float(fd.autocorr(lag=1)) if eff_n > 3 else np.nan
    adf_stat, adf_p = _adf(fd)
    kpss_stat, kpss_p = _kpss(fd)
    if not _HAS_STATSMODELS or np.isnan(adf_p):
        verdict = "no_statsmodels"
    elif adf_p < adf_alpha and kpss_p > kpss_alpha:
        verdict = "stationary"
    elif adf_p < adf_alpha or kpss_p > kpss_alpha:
        verdict = "borderline"
    else:
        verdict = "not_stationary"
    return FDResult(d, adf_stat, adf_p, kpss_stat, kpss_p, info_ret, eff_n, ac1, verdict)


def select_d(
    s: Series,
    *,
    window: int = 256,
    min_d: float = 0.0,
    prefer_minimal_d: bool = True,
    verbose: bool = False,
) -> tuple[float, FDResult]:
    """Select the minimum acceptable d on TRAIN data only.

    Search d in candidates [min_d..1.0]. The minimum d that achieves stationarity
    (ADF p < 0.05 AND KPSS p > 0.05, or ADF-only when statsmodels absent) is selected:
    lower d retains more memory. If no candidate is stationary, the highest-d candidate
    with the best (lowest) ADF p-value is returned with its verdict.
    """
    cands = [c for c in D_CANDIDATES if c >= min_d]
    results: list[FDResult] = []
    best: FDResult | None = None
    for d in cands:
        r = evaluate_d(s, d, window=window)
        results.append(r)
        if verbose:
            print(f"  d={d:.2f} ADFp={r.adf_pvalue:.4f} KPSSp={r.kpss_pvalue:.4f} "
                  f"IR={r.information_retention:.3f} {r.stationarity_verdict}")
        if best is None or (not np.isnan(r.adf_pvalue) and (np.isnan(best.adf_pvalue) or r.adf_pvalue < best.adf_pvalue)):
            best = r
        if r.stationarity_verdict == "stationary" and prefer_minimal_d:
            # first (lowest-d) stationary candidate is the minimum acceptable d
            return d, r
    # Fallback: return the most stationary candidate (lowest ADF p) with its own d.
    assert best is not None
    return best.d, best


@dataclass
class FracdiffTransform:
    """A frozen fractional-differentiation transform (fitted on TRAIN only)."""

    d: float
    window: int
    train_len: int
    diagnostic: FDResult | None = None
    metadata: dict = field(default_factory=dict)

    def transform(self, s: Series) -> Series:
        """Apply the frozen transform to any later series (validation/test/live)."""
        return fracdiff_series(s, self.d, window=self.window)

    def to_dict(self) -> dict:
        return {
            "d": self.d,
            "window": self.window,
            "train_len": self.train_len,
            "diagnostic": {
                "adf_stat": self.diagnostic.adf_stat if self.diagnostic else None,
                "adf_pvalue": self.diagnostic.adf_pvalue if self.diagnostic else None,
                "kpss_stat": self.diagnostic.kpss_stat if self.diagnostic else None,
                "kpss_pvalue": self.diagnostic.kpss_pvalue if self.diagnostic else None,
                "information_retention": (
                    self.diagnostic.information_retention if self.diagnostic else None
                ),
                "effective_sample_size": (
                    self.diagnostic.effective_sample_size if self.diagnostic else None
                ),
            },
            "metadata": self.metadata,
        }


def fit_fracdiff_transform(
    train_series: Series,
    *,
    window: int = 256,
    min_d: float = 0.0,
    verbose: bool = False,
) -> FracdiffTransform:
    """Fit (derive + freeze) the FD transform using TRAIN data only."""
    d, diag = select_d(train_series, window=window, min_d=min_d, verbose=verbose)
    return FracdiffTransform(d=d, window=window, train_len=len(train_series), diagnostic=diag)
