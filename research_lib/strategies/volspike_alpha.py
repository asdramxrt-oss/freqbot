"""Vol-spike alpha strategy (DeepSeek brief) - the volatility spike IS the signal.

Architecture:
  * A causal walk-forward XGBoost model predicts P(realized vol over the next horizon
    exceeds a train-only percentile threshold) - the VALIDATED capability (AUC 0.82-0.90).
  * The strategy trades the predicted spike: it enters only on a fresh probability crossing
    (P rises above p_entry), holds while P stays above p_exit, and exits on decay, a time
    stop, or a volatility-scaled stop-loss.
  * Direction is the open scientific question. Three rule families are exposed (H1/H2/H3):
      follow  = enter in the direction of pre-spike momentum (momentum ignition)
      fade    = enter AGAINST pre-spike momentum (vol-crush fade)
      dipbuy  = long only after a spike + down-move (asymmetric, strongest study cell)
    The walk-forward engine adjudicates between them on unseen windows - never on the fit
    window.

CAUSALITY (critical): a decision made at bar t uses only information available at the
CLOSE of bar t (p_spike[t], momentum to t, ATR to t). The resulting target is emitted for
bar t+1, so the fill happens at open[t+1] - the earliest executable price after the signal
was observable. This matches every other engine in the project.

Shorts require a margin/futures account - modeled explicitly, not assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class VolSpikeAlphaParams:
    direction: str = "follow"        # follow | fade | dipbuy | breakout
    p_entry: float = 0.50            # fresh-crossing entry threshold
    p_exit: float = 0.30             # exit when probability decays below this
    mom_bars: int = 8                # momentum lookback for direction (2h of 15m)
    breakout_bars: int = 32          # range lookback for 'breakout' mode (8h of 15m)
    max_hold_bars: int = 96          # time stop (1 day of 15m)
    stop_atr_mult: float = 3.0       # stop-loss as multiple of ATR(20) if provided


def direction_from_momentum(mom: float, rule: str) -> int:
    """Map momentum + rule to a desired side: +1 long, -1 short, 0 flat."""
    if not np.isfinite(mom) or mom == 0:
        return 0
    if rule == "follow":
        return 1 if mom > 0 else -1
    if rule == "fade":
        return -1 if mom > 0 else 1
    if rule == "dipbuy":
        return 1 if mom < 0 else 0
    raise ValueError(f"unknown direction rule {rule}")


def _breakout_side(
    high: np.ndarray, low: np.ndarray, close: float, i: int, lookback: int
) -> int:
    """Range breakout direction at bar i using a SHIFTED prior range.

    The current bar can never break its own range: the reference range is the max high /
    min low over bars [i-lookback, i-1] (strictly prior). Entry occurs when bar i's close
    exceeds the prior range high (long) or breaks below the prior range low (short).
    """
    lo = max(0, i - lookback)
    prior_high = float(np.max(high[lo:i])) if i > lo else np.nan
    prior_low = float(np.min(low[lo:i])) if i > lo else np.nan
    if np.isnan(prior_high) or np.isnan(prior_low):
        return 0
    if close > prior_high:
        return 1
    if close < prior_low:
        return -1
    return 0


def generate_alpha_signals(
    df: DataFrame,
    p_spike: np.ndarray,
    params: VolSpikeAlphaParams,
    atr: np.ndarray | None = None,
) -> DataFrame:
    """Generate target/stop arrays aligned to df for EventBacktester.

    Contract with the backtester: target[t] is acted at OPEN[t]. Therefore target[t] is
    decided from p_spike[t-1] / momentum[t-1] / atr[t-1] (all known at close[t-1]).
    p_spike[i] must itself be computable from data <= i (causal walk-forward output).
    """
    n = len(df)
    if len(p_spike) != n:
        raise ValueError("p_spike length mismatch")
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float) if "high" in df else close
    low = df["low"].to_numpy(dtype=float) if "low" in df else close
    logc = np.log(close)
    mom = np.full(n, np.nan)
    mom[params.mom_bars:] = logc[params.mom_bars:] - logc[:-params.mom_bars]

    target = np.zeros(n, dtype=int)
    stop_arr = np.full(n, np.nan)
    pos = 0          # position held during the CURRENT bar (decided at its open)
    entry_bar = -1   # bar index at which pos was opened (for time stop)
    prev_high = False

    def _side_at(i: int) -> int:
        if params.direction == "breakout":
            return _breakout_side(high, low, close[i], i, params.breakout_bars)
        return direction_from_momentum(mom[i], params.direction)

    for t in range(1, n):
        # ---- decide the position for bar t from info at close[t-1] ----
        i = t - 1
        p = p_spike[i]
        if np.isfinite(p):
            high_p = p > params.p_entry
            fresh = high_p and not prev_high
            if pos == 0:
                # breakout mode: enter whenever a range breakout occurs WHILE a spike is
                # predicted (not only on the fresh crossing - breakout events recur within
                # the predicted window)
                if params.direction == "breakout":
                    side = _side_at(i) if high_p else 0
                    entered = side != 0 and (pos == 0)
                else:
                    side = _side_at(i) if fresh else 0
                    entered = side != 0
                if entered and pos == 0 and side != 0:
                    pos = side
                    entry_bar = t
                    if atr is not None and np.isfinite(atr[i]) and atr[i] > 0:
                        stop_arr[t] = close[i] - side * params.stop_atr_mult * atr[i]
            elif pos != 0:
                # exits: probability decay, time stop, or opposite signal
                if p < params.p_exit or (t - entry_bar) >= params.max_hold_bars:
                    pos = 0
                else:
                    side_now = _side_at(i)
                    if params.direction == "breakout":
                        # exit on an opposite breakout while still in a predicted spike
                        if side_now != 0 and side_now != pos:
                            pos = 0
                    elif side_now != 0 and side_now != pos and fresh:
                        pos = 0
            prev_high = high_p
        else:
            prev_high = False
        target[t] = pos
    out = DataFrame({"date": df["date"], "target": target, "stop": stop_arr, "p": p_spike,
                     "mom": mom})
    return out
