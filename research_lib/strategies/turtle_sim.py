"""Pyramiding-aware turtle simulator with realistic fills (same cost model as EventBacktester).

Faithful turtle execution:
  * Unit size: each unit risks 1% of CURRENT equity per 1N move
        qty_i = 0.01 * E / N_i          (spec section 8: Q = 0.01*Equity / (N * ValuePerPoint))
    With a 2N initial stop each first unit risks ~2%; the risk engine caps units and exposure.
  * Entry of unit 1 at next open after the 20/55-bar breakout (signal at close t-1).
  * Pyramiding: add one unit each time price moves 0.5N beyond the LAST entry (state machine),
    up to ``max_units``; never twice at the same level.
  * Stop: 2N from the most recent entry (whole-position stop), checked intra-bar; if the bar
    opens through the stop the fill is the open (conservative).
  * Exit: opposite 20/55-bar breakout (classic turtle system) at next open.

Output: trades DataFrame (per POSITION, with unit breakdown summary), an equity curve, and the
pyramid event log for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel


@dataclass
class TurtleSimParams:
    entry_n: int = 20
    atr_n: int = 20
    stop_mult: float = 2.0
    risk_per_unit: float = 0.01
    pyramid_step_n: float = 0.5
    max_units: int = 4
    max_leverage: float = 1.0  # spot default - caps per-unit notional contribution
    allow_short: bool = False  # spot data cannot short without a margin account
    equity: float = 100_000.0
    cost: CostModel = field(default_factory=CostModel)


@dataclass
class _Unit:
    entry_time: object
    entry_price: float
    qty: float


class TurtleSimulator:
    def __init__(self, params: TurtleSimParams | None = None):
        self.p = params or TurtleSimParams()

    # ------------------------------------------------------------------
    def _wilder_atr(self, df: DataFrame) -> np.ndarray:
        from research_lib.features.engine import atr

        return atr(df, self.p.atr_n).to_numpy()

    def _donchian(self, df: DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
        ph = df["high"].shift(1)
        pl = df["low"].shift(1)
        return ph.rolling(n).max().to_numpy(), pl.rolling(n).min().to_numpy()

    def _fill(self, ref: float, side: int) -> float:
        adv = 1.0 + self.p.cost.half_spread() + self.p.cost.slippage()
        return ref * adv if side == 1 else ref * (2.0 - adv)

    def _fees(self, notional: float) -> float:
        c = self.p.cost
        return notional * (c.fee() + c.half_spread() + c.slippage())

    # ------------------------------------------------------------------
    def run(self, df: DataFrame) -> tuple[DataFrame, np.ndarray, list[dict]]:
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        dates = df["date"].to_numpy()
        n = len(df)
        n_arr = self._wilder_atr(df)
        dc_up_hi, dc_lo_hi = self._donchian(df, self.p.entry_n)
        # exit channel = same as entry channel for classic turtle system 1
        dc_up_ex, dc_lo_ex = dc_up_hi, dc_lo_hi

        equity = self.p.equity
        units: list[_Unit] = []
        side = 0
        stop = np.nan
        trades_list: list[dict] = []
        log: list[dict] = []
        eq_curve = np.full(n, np.nan)
        pos_entry_bar = -1
        if n > 0:
            eq_curve[0] = equity

        def close_all(t: int, price: float, reason: str) -> None:
            nonlocal equity, units, side, stop
            pnl = 0.0
            fees = 0.0
            gross = 0.0
            for u in units:
                g = (price - u.entry_price) * u.qty * side
                gross += g
                f = self._fees(u.qty * price)
                fees += f
                pnl += g - f
            # position-level MFE/MAE (price units * side * total qty approximation on avg)
            avg_entry = float(np.mean([u.entry_price for u in units])) if units else 0.0
            total_qty = float(sum(u.qty for u in units))
            window = df.iloc[pos_entry_bar : t + 1]
            if side == 1:
                mfe = (window["high"].max() - avg_entry) * total_qty
                mae = (avg_entry - window["low"].min()) * total_qty
            else:
                mfe = (avg_entry - window["low"].min()) * total_qty
                mae = (window["high"].max() - avg_entry) * total_qty
            trades_list.append(
                {
                    "entry_time": units[0].entry_time if units else dates[t],
                    "exit_time": dates[t],
                    "side": side,
                    "units": len(units),
                    "avg_entry": avg_entry,
                    "exit_price": float(price),
                    "notional": float(avg_entry * total_qty),
                    "pnl_net": float(pnl),
                    "fees_paid": float(fees),
                    "mfe": float(mfe),
                    "mae": float(mae),
                    "exit_reason": reason,
                    "bars_held": t - pos_entry_bar,
                }
            )
            log.append({"date": dates[t], "action": "exit", "side": side, "units": len(units), "reason": reason, "price": float(price)})
            equity += pnl
            units = []
            side = 0
            stop = np.nan

        for t in range(1, n):
            prev = t - 1
            prev2 = t - 2
            long_entry = np.isfinite(dc_up_hi[prev]) and closes[prev] > dc_up_hi[prev]
            short_entry = np.isfinite(dc_lo_hi[prev]) and closes[prev] < dc_lo_hi[prev]
            # fresh-crossing: entry requires close prev beyond channel AND close prev2 not
            # already beyond it (prevents stop-out -> instant re-entry at the same level)
            if prev2 >= 0:
                long_fresh = long_entry and not (
                    np.isfinite(dc_up_hi[prev2]) and closes[prev2] > dc_up_hi[prev2]
                )
                short_fresh = short_entry and not (
                    np.isfinite(dc_lo_hi[prev2]) and closes[prev2] < dc_lo_hi[prev2]
                )
            else:
                long_fresh, short_fresh = long_entry, short_entry
            long_exit = np.isfinite(dc_lo_ex[prev]) and closes[prev] < dc_lo_ex[prev]
            short_exit = np.isfinite(dc_up_ex[prev]) and closes[prev] > dc_up_ex[prev]
            n_prev = n_arr[prev] if np.isfinite(n_arr[prev]) else opens[t] * 0.01

            # ---- open a position (from previous close signal) at this open ----
            if side == 0:
                new_side = 0
                if long_fresh:
                    new_side = 1
                elif short_fresh and self.p.allow_short:
                    new_side = -1
                if new_side != 0:
                    px = self._fill(opens[t], new_side)
                    if px > 0 and n_prev > 0:
                        qty_risk = (self.p.risk_per_unit * equity) / n_prev
                        # leverage cap: notional of this unit cannot push aggregate notional
                        # above current equity * max_leverage
                        qty_cap = (equity * self.p.max_leverage) / px
                        qty = min(qty_risk, qty_cap)
                        fees = self._fees(qty * px)
                        equity -= fees
                        units = [_Unit(entry_time=dates[t], entry_price=px, qty=qty)]
                        side = new_side
                        stop = px - side * self.p.stop_mult * n_prev
                        pos_entry_bar = t
                        log.append({"date": dates[t], "action": "entry", "side": side, "units": 1, "price": px, "N": n_prev, "qty": qty})
            # ---- manage open position ----
            if side != 0:
                # pyramiding add: 0.5N beyond last entry, once per level
                last = units[-1].entry_price
                open_notional = float(sum(u.qty * u.entry_price for u in units))
                remaining_cap = (equity * self.p.max_leverage - open_notional) / max(opens[t], 1e-9)
                if len(units) < self.p.max_units:
                    if side == 1 and opens[t] >= last + self.p.pyramid_step_n * n_prev:
                        px = self._fill(opens[t], 1)
                        qty_risk = (self.p.risk_per_unit * equity) / n_prev
                        # only add when a FULL risk-sized unit fits inside the leverage
                        # budget (a dust add that moves the stop is an execution artifact)
                        if remaining_cap >= qty_risk:
                            qty = qty_risk
                            fees = self._fees(qty * px)
                            equity -= fees
                            units.append(_Unit(entry_time=dates[t], entry_price=px, qty=qty))
                            stop = px - self.p.stop_mult * n_prev
                            log.append({"date": dates[t], "action": "add", "side": side, "units": len(units), "price": px, "N": n_prev, "qty": qty})
                    elif side == -1 and opens[t] <= last - self.p.pyramid_step_n * n_prev:
                        px = self._fill(opens[t], -1)
                        qty_risk = (self.p.risk_per_unit * equity) / n_prev
                        if remaining_cap >= qty_risk:
                            qty = qty_risk
                            fees = self._fees(qty * px)
                            equity -= fees
                            units.append(_Unit(entry_time=dates[t], entry_price=px, qty=qty))
                            stop = px + self.p.stop_mult * n_prev
                            log.append({"date": dates[t], "action": "add", "side": side, "units": len(units), "price": px, "N": n_prev, "qty": qty})
                # stop check intra-bar
                exit_price = np.nan
                reason = ""
                if side == 1 and lows[t] <= stop:
                    exit_price = stop if opens[t] > stop else opens[t]
                    reason = "stop"
                elif side == -1 and highs[t] >= stop:
                    exit_price = stop if opens[t] < stop else opens[t]
                    reason = "stop"
                # opposite-breakout exit (signal at prev close, act this open)
                if np.isnan(exit_price):
                    if (side == 1 and long_exit) or (side == -1 and short_exit):
                        exit_price = opens[t]
                        reason = "opposite_breakout"
                if not np.isnan(exit_price):
                    close_all(t, exit_price, reason)
            # ---- equity mark ----
            if side != 0:
                total_qty = sum(u.qty for u in units)
                avg_entry = float(np.mean([u.entry_price for u in units]))
                eq_curve[t] = equity + (closes[t] - avg_entry) * total_qty * side
            else:
                eq_curve[t] = equity

        if side != 0:
            close_all(n - 1, self._fill(closes[-1], -side), "force_close")

        trades = DataFrame(trades_list)
        return trades, eq_curve, log
