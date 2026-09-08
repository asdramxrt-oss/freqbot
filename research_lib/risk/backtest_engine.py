"""Vectorized event-driven backtester (single-instrument, single-position core).

Execution model (realistic, shared by ALL strategies so results are comparable):
  * A strategy computes, at bar t (using data up to and including t's close), a DESIRED
    position target[t] in {-1, 0, +1}. The engine fills at bar t+1's open - the signal is
    therefore only actionable one bar after it is observable (no same-candle execution).
  * Entry/exit fills use the CostModel adverse price at the open.
  * Stops/take-profits are evaluated intra-bar against high/low with a conservative rule:
    if the bar opens through the level, the fill is at the open (worse); otherwise the fill
    is at the stop/TP level. Same-bar evaluation is legitimate for open fills because the
    open is the first tick of the bar.
  * Position size is risk-based: units = (equity * risk_per_trade) / (entry - stop).
  * One position at a time (no stacking). Pyramiding is handled by higher-level state
    machines (Turtle) that emit their own target/stop arrays per unit.

Output: trade log DataFrame (entry/exit time, side, prices, units, pnl_net, ret_net, fees,
MFE, MAE, reason, bars_held) and an equity curve array aligned to the input bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel


@dataclass
class EngineConfig:
    equity: float = 100_000.0
    cost: CostModel = field(default_factory=CostModel)
    risk_per_trade: float = 0.01
    max_leverage: float = 1.0  # spot default; caps notional at equity * max_leverage
    allow_short: bool = True


def risk_sized_units(
    equity: float,
    risk_frac: float,
    entry: float,
    stop_dist: float,
    max_leverage: float,
) -> float:
    """Units sized to risk ``risk_frac`` of equity, capped by the leverage/notional limit.

    units_risk = (equity * risk_frac) / stop_dist
    units_cap  = (equity * max_leverage) / entry   (notional <= leverage * equity)
    Returns min(units_risk, units_cap) so 1x accounts never hold notional above equity.
    """
    if stop_dist <= 0 or entry <= 0 or equity <= 0:
        return 0.0
    units_risk = (equity * risk_frac) / stop_dist
    units_cap = (equity * max_leverage) / entry
    return max(min(units_risk, units_cap), 0.0)


@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    units: float
    notional: float
    pnl_net: float
    ret_net: float
    fees_paid: float
    mfe: float
    mae: float
    exit_reason: str
    bars_held: int

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "units": self.units,
            "notional": self.notional,
            "pnl_net": self.pnl_net,
            "ret_net": self.ret_net,
            "fees_paid": self.fees_paid,
            "mfe": self.mfe,
            "mae": self.mae,
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
        }


class EventBacktester:
    """Bar-by-bar backtester.

    Inputs (all length n, aligned to df rows):
      target[t]    desired position computed from data <= t (acted at t+1 open).
      stop[t]      active stop price DURING bar t for an open position (may be NaN).
      tp[t]        active take-profit price during bar t (may be NaN).
    """

    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()

    # ------------------------------------------------------------------
    def _fill(self, ref: float, side: int) -> float:
        """Adverse fill for a buy (side=+1) or sell (side=-1) at ref."""
        c = self.cfg.cost
        adv = 1.0 + c.half_spread() + c.slippage()
        if side == 1:
            return ref * adv
        return ref * (2.0 - adv)

    def _fees(self, notional: float) -> float:
        c = self.cfg.cost
        return notional * (c.fee() + c.half_spread() + c.slippage())

    # ------------------------------------------------------------------
    def run(
        self,
        df: DataFrame,
        target: np.ndarray,
        stop: np.ndarray | None = None,
        tp: np.ndarray | None = None,
    ) -> tuple[DataFrame, np.ndarray]:
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        dates = df["date"].to_numpy()
        n = len(df)
        if len(target) != n:
            raise ValueError("target length mismatch")
        stop = np.full(n, np.nan) if stop is None else np.asarray(stop, dtype=float)
        tp = np.full(n, np.nan) if tp is None else np.asarray(tp, dtype=float)

        equity = self.cfg.equity
        trades: list[TradeRecord] = []
        pos_side = 0
        pos_units = 0.0
        pos_entry = 0.0
        pos_bar = -1
        pos_date = None
        pos_stop = np.nan
        pos_tp = np.nan
        pos_fees_entry = 0.0
        eq_curve = np.full(n, np.nan)

        def close_position(t: int, price: float, reason: str) -> None:
            nonlocal equity, pos_side, pos_units, pos_entry, pos_bar, pos_date, pos_stop, pos_tp
            gross = (price - pos_entry) * pos_units * pos_side
            fees = self._fees(pos_units * price)
            pnl = gross - fees
            if pos_side == 1:
                mfe = (highs[pos_bar : t + 1].max() - pos_entry) * pos_units
                mae = (pos_entry - lows[pos_bar : t + 1].min()) * pos_units
            else:
                mfe = (pos_entry - lows[pos_bar : t + 1].min()) * pos_units
                mae = (highs[pos_bar : t + 1].max() - pos_entry) * pos_units
            trades.append(
                TradeRecord(
                    entry_time=pos_date,
                    exit_time=dates[t],
                    side=pos_side,
                    entry_price=pos_entry,
                    exit_price=float(price),
                    units=pos_units,
                    notional=pos_units * pos_entry,
                    pnl_net=float(pnl),
                    ret_net=float(pnl / (pos_units * pos_entry)) if pos_units else 0.0,
                    fees_paid=float(fees),
                    mfe=float(mfe),
                    mae=float(mae),
                    exit_reason=reason,
                    bars_held=t - pos_bar,
                )
            )
            equity += pnl
            pos_side = 0
            pos_units = 0.0
            pos_entry = 0.0
            pos_bar = -1
            pos_date = None
            pos_stop = np.nan
            pos_tp = np.nan

        prev_desired = 0  # previous bar's desired side - enables edge-triggered entries
        for t in range(n):
            # ---- signal-triggered action decided at previous close, filled now ----
            desired = int(target[t]) if t > 0 else 0  # no action on the first bar
            fresh_entry = pos_side == 0 and desired != 0 and prev_desired == 0
            if fresh_entry:
                entry = self._fill(opens[t], desired)
                sd = abs(entry - stop[t]) if not np.isnan(stop[t]) else entry * 0.02
                sd = max(sd, entry * 1e-4)
                units = risk_sized_units(equity, self.cfg.risk_per_trade, entry, sd, self.cfg.max_leverage)
                if units > 0:
                    fee_e = self._fees(units * entry)
                    equity -= fee_e
                    pos_side = desired
                    pos_units = units
                    pos_entry = entry
                    pos_bar = t
                    pos_date = dates[t]
                    pos_stop = stop[t] if not np.isnan(stop[t]) else entry - desired * sd
                    pos_tp = tp[t] if not np.isnan(tp[t]) else np.nan
                    pos_fees_entry = fee_e
            elif pos_side != 0 and desired == 0:
                close_position(t, self._fill(opens[t], -pos_side), "signal")
            elif pos_side != 0 and desired != 0 and desired != pos_side and prev_desired == 0:
                # direct reversal: close then open in the new direction at this bar's open
                close_position(t, self._fill(opens[t], -pos_side), "reversal_close")
                entry = self._fill(opens[t], desired)
                sd = abs(entry - stop[t]) if not np.isnan(stop[t]) else entry * 0.02
                sd = max(sd, entry * 1e-4)
                units = risk_sized_units(equity, self.cfg.risk_per_trade, entry, sd, self.cfg.max_leverage)
                if units > 0:
                    fee_e = self._fees(units * entry)
                    equity -= fee_e
                    pos_side = desired
                    pos_units = units
                    pos_entry = entry
                    pos_bar = t
                    pos_date = dates[t]
                    pos_stop = stop[t] if not np.isnan(stop[t]) else entry - desired * sd
                    pos_tp = tp[t] if not np.isnan(tp[t]) else np.nan
            prev_desired = desired

            # ---- manage open position intra-bar (stop/TP), including same-bar entry ----
            if pos_side != 0:
                # TP first (favorable) then SL (conservative); if the bar opens beyond a
                # level the fill is the open (worse than the level).
                exit_price = np.nan
                reason = ""
                if pos_side == 1:
                    if not np.isnan(pos_tp) and highs[t] >= pos_tp:
                        exit_price = pos_tp if opens[t] <= pos_tp else opens[t]
                        reason = "take_profit"
                    if np.isnan(exit_price) and lows[t] <= pos_stop:
                        exit_price = min(pos_stop, opens[t]) if opens[t] < pos_stop else pos_stop
                        reason = "stop"
                else:
                    if not np.isnan(pos_tp) and lows[t] <= pos_tp:
                        exit_price = pos_tp if opens[t] >= pos_tp else opens[t]
                        reason = "take_profit"
                    if np.isnan(exit_price) and highs[t] >= pos_stop:
                        exit_price = max(pos_stop, opens[t]) if opens[t] > pos_stop else pos_stop
                        reason = "stop"
                if not np.isnan(exit_price):
                    close_position(t, exit_price, reason)

            # ---- equity mark at close ----
            if pos_side != 0:
                eq_curve[t] = equity + (closes[t] - pos_entry) * pos_units * pos_side
            else:
                eq_curve[t] = equity

        if pos_side != 0:
            close_position(n - 1, self._fill(closes[-1], -pos_side), "force_close")

        out = DataFrame([t.to_dict() for t in trades])
        return out, eq_curve
