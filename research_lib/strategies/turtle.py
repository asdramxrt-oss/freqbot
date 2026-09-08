"""Turtle trend-following engine (Phase 6) with pyramiding state machine (Phase 9 add-ons).

Spec-critical correctness points:
  * Breakout levels use SHIFTED high/low so the current candle cannot influence its own
    breakout decision:
        donchian_high = high.shift(1).rolling(N).max()
        donchian_low  = low.shift(1).rolling(N).min()
  * N = ATR(20) (Wilder), computed causally.
  * System 1: 20-bar breakout; System 2: 55-bar breakout (configurable).
  * A signal on bar t is actionable at bar t+1 open (engine-level shift).
  * Position sizing: Q = 0.01 * E / (N * value_per_point); handled by the risk engine /
    backtester, NOT here. This module emits *intents* (side + N + entry ref + stop), never
    final quantities.
  * Pyramiding is a state machine: up to ``max_units`` units, add ~0.5N in the favorable
    direction, never twice at the same threshold, stop = 2N from the last entry for the
    latest unit (turtle rule: stop on the whole position is 2N from the most recent add).

This engine produces a unit-level target/stop stream consumable by EventBacktester and,
separately, a signal/event log used for triple-barrier meta-labeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from research_lib.constants import TURTLE_DEFAULTS
from research_lib.features.engine import atr


@dataclass
class TurtleParams:
    system: Literal[1, 2] = 1
    entry_n: int = 20            # breakout lookback (system 1)
    atr_n: int = 20              # N = ATR
    stop_mult: float = 2.0       # stop distance in N
    pyramid_max_units: int = 4
    pyramid_step_n: float = 0.5  # add every 0.5N
    use_pyramiding: bool = True
    exit_on_opposite_breakout: bool = True  # classic turtle: exit long on 20d low break


def donchian_channels(df: DataFrame, n: int) -> DataFrame:
    """Shifted Donchian channels (upper/lower/mid). CRITICAL: uses shift(1)."""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    upper = prev_high.rolling(n).max()
    lower = prev_low.rolling(n).min()
    out = DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": (upper + lower) / 2.0})
    return out


@dataclass
class PyramidState:
    """State machine for a single turtle position (possibly multi-unit)."""

    side: int
    units: int = 1
    entry_prices: list[float] = field(default_factory=list)
    entry_times: list = field(default_factory=list)
    n_at_entry: float = 0.0

    @property
    def avg_entry(self) -> float:
        if not self.entry_prices:
            return 0.0
        return float(np.mean(self.entry_prices))

    @property
    def last_entry(self) -> float:
        return self.entry_prices[-1] if self.entry_prices else 0.0

    @property
    def next_add_price(self) -> float:
        if self.side == 1:
            return self.last_entry + 0.5 * self.n_at_entry
        return self.last_entry - 0.5 * self.n_at_entry

    def can_add(self, price: float, max_units: int) -> bool:
        if self.units >= max_units:
            return False
        # price must have moved a fresh 0.5N beyond the LAST add (no repeat at same level)
        if self.side == 1:
            return price >= self.next_add_price * (1 + 1e-12)
        return price <= self.next_add_price * (1 - 1e-12)


def _wilder_atr_series(df: DataFrame, n: int) -> Series:
    return atr(df, n)


class TurtleEngine:
    """Emits target/stop arrays for the EventBacktester plus a pyramiding state log.

    Convention: arrays are aligned to df; target[t] reflects the position the strategy wants
    to hold DURING bar t, having observed bars up to t-1 (entry at t open). This matches the
    backtester's shift model directly (target at index t => fill at open[t]).
    """

    def __init__(self, params: TurtleParams | None = None):
        self.p = params or TurtleParams()
        if self.p.system == 2:
            self.p.entry_n = 55  # system 2 breakout length

    # ------------------------------------------------------------------
    def compute_levels(self, df: DataFrame) -> DataFrame:
        dc = donchian_channels(df, self.p.entry_n)
        n_series = _wilder_atr_series(df, self.p.atr_n)
        dc["N"] = n_series
        # long breakout occurs when PREVIOUS close exceeds PREVIOUS donchian high
        # => signal evaluated at t using data <= t is actionable at t+1. We emit target
        # aligned to the signal bar and let the backtester act one bar later.
        dc["long_breakout"] = (df["close"] > dc["dc_upper"]).astype(int)
        dc["short_breakout"] = (df["close"] < dc["dc_lower"]).astype(int)
        return dc

    def generate(
        self, df: DataFrame
    ) -> tuple[DataFrame, list[dict]]:
        """Generate unit-level state.

        Returns (signals_df, pyramid_log). signals_df has columns:
          date, side (desired at close of bar), N, entry_ref (next open), initial_stop,
          breakout_level, signal_time (the bar whose close triggered), and a unit counter.
        pyramid_log records each unit add as a state-machine event for testing/audit.
        """
        lvl = self.compute_levels(df)
        closes = df["close"].to_numpy()
        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        n_vals = lvl["N"].to_numpy()
        dc_up = lvl["dc_upper"].to_numpy()
        dc_lo = lvl["dc_lower"].to_numpy()
        dates = df["date"].to_numpy()
        n = len(df)

        side = np.zeros(n, dtype=int)
        stop_arr = np.full(n, np.nan)
        pyramid_log: list[dict] = []
        # iterate bars; a breakout decided at close t -> target becomes active from t+1 on
        state: PyramidState | None = None
        # remember which bar the position became active (target index)
        # fresh-crossing rule: a breakout at close prev is an ENTRY only when the prior close
        # was NOT already beyond the channel (prevents stop-out -> immediate re-entry churn).
        for t in range(1, n):
            # decide the position to hold during bar t from info <= t-1 (close t-1)
            prev = t - 1
            prev2 = t - 2
            long_break = lvl["long_breakout"].iloc[prev] == 1
            short_break = lvl["short_breakout"].iloc[prev] == 1
            if state is None:
                # fresh cross required for a new entry
                if prev2 >= 0:
                    long_fresh = long_break and not (lvl["long_breakout"].iloc[prev2] == 1)
                    short_fresh = short_break and not (lvl["short_breakout"].iloc[prev2] == 1)
                else:
                    long_fresh, short_fresh = long_break, short_break
            else:
                long_fresh = short_fresh = False
            desired_side = 0
            if state is None:
                if long_fresh:
                    desired_side = 1
                elif short_fresh:
                    desired_side = -1
            else:
                # existing position: manage pyramiding / exits
                px = opens[t]  # reference for pyramid add is the OPEN we act at
                if state.side == 1:
                    # exit on opposite breakout (classic system 1 20d low break)
                    if self.p.exit_on_opposite_breakout and lvl["short_breakout"].iloc[prev] == 1:
                        desired_side = 0  # exit long
                    else:
                        desired_side = 1  # hold
                        if self.p.use_pyramiding and state.can_add(px, self.p.pyramid_max_units):
                            state.units += 1
                            state.entry_prices.append(px)
                            state.entry_times.append(dates[t])
                            pyramid_log.append(
                                {
                                    "date": dates[t],
                                    "action": "add",
                                    "side": 1,
                                    "unit": state.units,
                                    "price": px,
                                    "avg_entry": state.avg_entry,
                                    "N": n_vals[prev],
                                }
                            )
                elif state.side == -1:
                    if self.p.exit_on_opposite_breakout and lvl["long_breakout"].iloc[prev] == 1:
                        desired_side = 0
                    else:
                        desired_side = -1
                        if self.p.use_pyramiding and state.can_add(px, self.p.pyramid_max_units):
                            state.units += 1
                            state.entry_prices.append(px)
                            state.entry_times.append(dates[t])
                            pyramid_log.append(
                                {
                                    "date": dates[t],
                                    "action": "add",
                                    "side": -1,
                                    "unit": state.units,
                                    "price": px,
                                    "avg_entry": state.avg_entry,
                                    "N": n_vals[prev],
                                }
                            )
            # side[ t ] is the position to hold during bar t (entered at open t)
            side[t] = desired_side
            if desired_side != 0:
                if state is None:
                    state = PyramidState(
                        side=desired_side,
                        entry_prices=[opens[t]],
                        entry_times=[dates[t]],
                        n_at_entry=float(n_vals[prev]) if np.isfinite(n_vals[prev]) else opens[t] * 0.01,
                    )
                    pyramid_log.append(
                        {
                            "date": dates[t],
                            "action": "entry",
                            "side": desired_side,
                            "unit": 1,
                            "price": opens[t],
                            "avg_entry": opens[t],
                            "N": n_vals[prev],
                        }
                    )
                # update the stop for the whole position: 2N from the latest entry
                n_now = n_vals[prev] if np.isfinite(n_vals[prev]) else (state.n_at_entry or opens[t] * 0.01)
                if state.side == 1:
                    stop_arr[t] = state.last_entry - self.p.stop_mult * n_now
                else:
                    stop_arr[t] = state.last_entry + self.p.stop_mult * n_now
            else:
                if state is not None:
                    # position ended at open t (exit fill); log the exit
                    pyramid_log.append(
                        {
                            "date": dates[t],
                            "action": "exit",
                            "side": state.side,
                            "unit": state.units,
                            "price": opens[t],
                            "avg_entry": state.avg_entry,
                            "N": n_vals[prev],
                        }
                    )
                    state = None
                    stop_arr[t] = np.nan

        out = DataFrame(
            {
                "date": dates,
                "target": side,
                "stop": stop_arr,
                "N": n_vals,
                "dc_upper": dc_up,
                "dc_lower": dc_lo,
            }
        )
        return out, pyramid_log
