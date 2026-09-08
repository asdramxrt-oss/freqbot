"""Pairs trading engine (blueprint Phase 1.1) - market-neutral spread trading.

Key design decisions (driven by the honest pre-screen):
  * Whole-window OU on crypto log-ratios FAILS: every candidate ratio (ETH/BTC, SOL/BTC,
    BNB/BTC, SOL/ETH) trended strongly across 2022-2025 (e.g. ETH/BTC -2.53 -> -3.22).
    A static cointegration fit over the full history is therefore not a viable model and
    would repeat the OU failure.
  * This engine therefore estimates a LOCAL mean/spread model on a trailing estimation
    window (classic pairs-desk design): z_t = (spread_t - mean_t) / sd_t where mean/sd are
    estimated causally on the trailing window ONLY. This accommodates slow drift while
    capturing local mean reversion, and it is what must be validated OOS.
  * Estimation is walk-forward: parameters are re-estimated on a rolling basis and each
    bar's z-score uses ONLY data before it. No future observations enter any estimate.
  * Optional OU validation gate (reuses research_lib.strategies.ou gates): when enabled,
    trades are only taken if the trailing window passes the OU acceptance tests (0<b<1,
    residual stationarity, sane half-life) - a causal filter on tradability.
  * Execution is two-leg and dollar-neutral: long A / short B with equal notional.
    Costs are applied on BOTH legs (fee + half-spread + slippage each fill, taker default;
    a maker/post-only variant is provided for the execution study).

This module deliberately does NOT size positions: it emits signals; the risk engine and
position sizer own quantities (research_lib/risk).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel


@dataclass
class PairsParams:
    est_window_bars: int = 672   # trailing estimation window (7 days of 15m bars)
    z_entry: float = 2.0
    z_exit: float = 0.0
    ou_gate: bool = True          # require a valid OU fit on the trailing window to trade
    ou_min_sample: int = 300
    max_hold_bars: int = 672 * 2  # hard time stop in bars (else forced exit)
    cost: CostModel = field(default_factory=CostModel)


@dataclass
class SpreadModel:
    """Frozen local spread model (estimated on a trailing window)."""

    mean: float = 0.0
    std: float = 1.0
    ou_valid: bool = False
    half_life: float = np.nan
    n: int = 0


def log_ratio_series(
    df_a: DataFrame, df_b: DataFrame, close_col: str = "close"
) -> DataFrame:
    """Build an aligned frame with log(A/B) spread and both legs' OHLC.

    Returns a frame with columns date, open_a/high_a/.../close_a, open_b/.../close_b,
    spread = log(close_a/close_b). Only timestamps present in BOTH legs are kept
    (inner join); rows are sorted by date.
    """
    a = df_a[["date", "open", "high", "low", "close"]].rename(
        columns={"open": "open_a", "high": "high_a", "low": "low_a", "close": "close_a"}
    )
    b = df_b[["date", "open", "high", "low", "close"]].rename(
        columns={"open": "open_b", "high": "high_b", "low": "low_b", "close": "close_b"}
    )
    m = pd.merge(a, b, on="date", how="inner").sort_values("date").reset_index(drop=True)
    m["spread"] = np.log(m["close_a"] / m["close_b"])
    return m


def causal_local_z(frame: DataFrame, params: PairsParams) -> np.ndarray:
    """Causal z-score of the spread vs a trailing-window local mean/std.

    z[t] uses ONLY spread values in (t-est_window, t]. The first est_window bars are NaN
    (no estimation yet -> no trade, which is the safe default).
    """
    s = frame["spread"]
    roll = s.rolling(params.est_window_bars, min_periods=params.est_window_bars // 2)
    mu = roll.mean()
    sd = roll.std()
    z = (s - mu) / sd.replace(0, np.nan)
    return z.to_numpy(dtype=float)


def _quick_ou_valid(spread_window: np.ndarray, min_sample: int) -> tuple[bool, float]:
    """Fast OU sanity on a window: 0<b<1 with b-CI inside (0,1) and residual ADF stationarity.

    Used causally as a tradability filter. Returns (valid, half_life_bars).
    """
    from research_lib.strategies.ou import fit_ou, OUParams

    x = pd.Series(spread_window)
    model = fit_ou(x, OUParams(min_sample=min_sample, rolling_window=min(600, len(x) // 2)))
    if model.valid:
        return True, model.half_life
    return False, np.nan


@dataclass
class PairsSignalResult:
    z: np.ndarray
    target: np.ndarray   # +1 = long A/short B (spread cheap), -1 = short A/long B
    ou_valid: np.ndarray  # causal OU-tradability flag per bar
    events: list[dict]


def generate_signals(
    frame: DataFrame,
    params: PairsParams,
    *,
    ou_valid_arr: np.ndarray | None = None,
) -> PairsSignalResult:
    """Generate causal pairs signals over the frame.

    CONVENTION (matches EventBacktester / TurtleSim): a decision made at bar i uses only
    information available at the CLOSE of bar i (z[i], which is built from data <= i) and
    is written to target[i+1], so the earliest executable fill is the OPEN of bar i+1.
    Filling target[t] at open[t] from z[t] would be a same-bar look-ahead - this engine
    never does that.

    Entry when |z| crosses z_entry (spread cheap -> long spread). Exit when z crosses back
    through z_exit or the max-hold time stop fires. Events are logged at the bar whose open
    executes them.
    """
    n = len(frame)
    z = causal_local_z(frame, params)
    ou_ok = (
        ou_valid_arr
        if ou_valid_arr is not None
        else np.full(n, not params.ou_gate, dtype=bool)
    )
    target = np.zeros(n, dtype=int)
    pos = 0
    entry_bar = -1
    events: list[dict] = []
    dates = frame["date"].to_numpy()
    for i in range(1, n - 1):  # decision index i; executed at bar i+1 open
        j = i + 1  # bar whose open executes the decision made at close of i
        zi = z[i]
        if not np.isnan(zi):
            tradable = ou_ok[i] if params.ou_gate else True
            if pos == 0:
                if tradable and zi <= -params.z_entry:
                    pos = 1
                    entry_bar = j
                    events.append({"date": dates[j], "action": "entry", "side": 1, "z": float(zi)})
                elif tradable and zi >= params.z_entry:
                    pos = -1
                    entry_bar = j
                    events.append({"date": dates[j], "action": "entry", "side": -1, "z": float(zi)})
            elif pos == 1:
                if zi >= params.z_exit or (j - entry_bar) >= params.max_hold_bars:
                    pos = 0
                    events.append({"date": dates[j], "action": "exit", "side": 1, "z": float(zi)})
            elif pos == -1:
                if zi <= params.z_exit or (j - entry_bar) >= params.max_hold_bars:
                    pos = 0
                    events.append({"date": dates[j], "action": "exit", "side": -1, "z": float(zi)})
        target[j] = pos
    return PairsSignalResult(z=z, target=target, ou_valid=ou_ok, events=events)
