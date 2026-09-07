"""Hurst exponent via Detrended Fluctuation Analysis (DFA) and rescaled range (R/S).

DFA is the primary method (robust to non-stationarities). Hurst is used ONLY as a candidate
filter in this project - never as sole proof of mean reversion (per spec, the OU engine must
additionally validate 0<b<1, half-life reasonableness, residual stationarity, sample size and
parameter stability before mean reversion is enabled).

Causal use: a rolling window of past observations only. No future data enters the estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import Series


def dfa_hurst(x: np.ndarray, min_box: int = 8, max_box: int | None = None) -> float:
    """Estimate H via DFA on a 1-D series.

    Returns NaN if the series is too short or the fit is degenerate.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(2 * min_box, 32):
        return np.nan
    y = np.cumsum(x - x.mean())
    n = len(y)
    max_box = max_box or n // 4
    if max_box < min_box + 1:
        return np.nan
    box_sizes = np.unique(np.geomspace(min_box, max_box, num=24).astype(int))
    box_sizes = box_sizes[box_sizes >= min_box]
    if len(box_sizes) < 4:
        return np.nan
    flucts: list[float] = []
    for box in box_sizes:
        n_boxes = n // box
        if n_boxes < 1:
            continue
        # reshape into boxes, detrend each via least-squares line fit, RMS the residuals
        trimmed = y[: n_boxes * box].reshape(n_boxes, box)
        t = np.arange(box, dtype=float)
        # design matrix [1, t]
        A = np.vstack([np.ones(box), t]).T
        coef, *_ = np.linalg.lstsq(A, trimmed.T, rcond=None)  # shape (2, n_boxes)
        fit = A @ coef  # (box, n_boxes)
        resid = trimmed.T - fit
        f = np.sqrt(np.mean(resid**2))
        if f > 0:
            flucts.append((float(box), float(f)))
    if len(flucts) < 4:
        return np.nan
    boxes = np.array([b for b, _ in flucts])
    fs = np.array([f for _, f in flucts])
    slope, *_ = np.polyfit(np.log(boxes), np.log(fs), 1)
    return float(slope)


def rolling_hurst(
    s: Series, window: int = 500, min_box: int = 8, step: int = 1
) -> Series:
    """Causal rolling DFA-Hurst over trailing windows.

    Output is aligned to the LAST index of each window, so value at t uses only s[..t].
    """
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for t in range(window - 1, n, step):
        out[t] = dfa_hurst(vals[t - window + 1 : t + 1], min_box=min_box)
    idx = np.arange(n)
    # fill the skipped rows (step>1) forward from the last computed value (causal)
    valid = np.where(~np.isnan(out))[0]
    if len(valid) > 0:
        out = _forward_fill(out, valid)
    return Series(out, index=s.index, name="hurst_dfa")


def _forward_fill(arr: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    last = np.nan
    j = 0
    for i in range(len(arr)):
        if j < len(valid_idx) and valid_idx[j] == i:
            last = arr[i]
            j += 1
        out[i] = last
    return out


def rolling_hurst_rs(s: Series, window: int = 500) -> Series:
    """R/S rescaled-range Hurst (secondary method, causal rolling window)."""
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        seg = vals[t - window + 1 : t + 1]
        out[t] = _rs_hurst(seg)
    return Series(out, index=s.index, name="hurst_rs")


def _rs_hurst(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 64:
        return np.nan
    # split into 2^k blocks, average R/S per block size (classic method)
    sub_len = n
    halves = []
    while sub_len >= 32:
        halves.append(sub_len)
        sub_len //= 2
    if len(halves) < 3:
        return np.nan
    rs_vals: list[float] = []
    for length in halves:
        n_blocks = n // length
        block_rs = []
        for b in range(n_blocks):
            seg = x[b * length : (b + 1) * length]
            mean = seg.mean()
            dev = np.cumsum(seg - mean)
            r = dev.max() - dev.min()
            sd = seg.std(ddof=1)
            if sd > 0 and r > 0:
                block_rs.append(r / sd)
        if block_rs:
            rs_vals.append((float(np.log(length)), float(np.log(np.mean(block_rs)))))
    if len(rs_vals) < 3:
        return np.nan
    xs = np.array([a for a, _ in rs_vals])
    ys = np.array([b for _, b in rs_vals])
    slope, *_ = np.polyfit(xs, ys, 1)
    return float(slope)
