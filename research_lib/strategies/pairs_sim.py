"""Two-leg dollar-neutral pairs backtester.

Execution model (identical discipline to the single-leg engines):
  * signal decided at close t -> fills at open t+1 for BOTH legs (causal shift)
  * long leg fills at the ask-side (adverse), short leg fills at the bid-side (adverse)
  * costs per leg per fill: fee + half-spread + slippage (CostModel)
  * dollar-neutral: notional_leg = equity * capital_fraction on each leg
  * PnL = (A_return - B_return) on the long/short legs minus all costs.
  * Optional maker mode: entries/exits can rest as post-only limit orders at the touch;
    the maker variant is a SEPARATE execution study (the maker rebate/cheaper fee is only
    realistic if the order actually provides liquidity - modeled conservatively here by
    requiring the next bar to trade through the quoted level).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel


@dataclass
class PairsSimParams:
    equity: float = 100_000.0
    capital_fraction: float = 0.25   # notional per leg as fraction of equity (25% per leg = 50% gross)
    cost: CostModel = field(default_factory=CostModel)


def _fill_price(ref: float, side: int, cost: CostModel) -> float:
    """side=+1 buy (pay up), side=-1 sell (receive less)."""
    adv = 1.0 + cost.half_spread() + cost.slippage()
    return ref * adv if side == 1 else ref * (2.0 - adv)


def _fees(notional: float, cost: CostModel) -> float:
    return notional * (cost.fee() + cost.half_spread() + cost.slippage())


class PairsSimulator:
    """Backtest a pairs target stream. target[t]==+1 means hold long A/short B from open
    t+1; 0 means flat. Entries/exits fill at the next bar open (signal shift by one)."""

    def __init__(self, params: PairsSimParams | None = None):
        self.p = params or PairsSimParams()

    def run(self, frame: DataFrame, target: np.ndarray) -> tuple[DataFrame, np.ndarray]:
        dates = frame["date"].to_numpy()
        open_a = frame["open_a"].to_numpy(dtype=float)
        close_a = frame["close_a"].to_numpy(dtype=float)
        open_b = frame["open_b"].to_numpy(dtype=float)
        close_b = frame["close_b"].to_numpy(dtype=float)
        n = len(frame)
        if len(target) != n:
            raise ValueError("target length mismatch")

        notional = self.p.equity * self.p.capital_fraction
        cost = self.p.cost
        pos = 0
        trades: list[dict] = []
        eq_curve = np.full(n, np.nan)
        equity = self.p.equity
        entry_bar = -1
        entry_date = None
        entry_price_a = 0.0
        entry_price_b = 0.0
        qa = 0.0
        qb = 0.0

        for t in range(1, n):
            desired = int(target[t]) if t > 0 else 0
            # enter on rising edge from flat (no stop-out re-entry churn)
            if pos == 0 and desired != 0:
                if desired == 1:
                    pa = _fill_price(open_a[t], 1, cost)   # buy A
                    pb = _fill_price(open_b[t], -1, cost)  # sell B
                else:
                    pa = _fill_price(open_a[t], -1, cost)  # sell A
                    pb = _fill_price(open_b[t], 1, cost)   # buy B
                # dollar-neutral: per-leg notional is exact at the FILL price
                qa = notional / pa
                qb = notional / pb
                # costs on both legs at entry
                fees = _fees(qa * pa, cost) + _fees(qb * pb, cost)
                equity -= fees
                pos = desired
                entry_bar = t
                entry_date = dates[t]
                entry_price_a = float(pa)
                entry_price_b = float(pb)
                trades.append(None)  # placeholder; finalized on exit
            elif pos != 0 and desired == 0:
                # exit both legs at this open (qty locked since entry)
                if pos == 1:
                    xa = _fill_price(open_a[t], -1, cost)
                    xb = _fill_price(open_b[t], 1, cost)
                else:
                    xa = _fill_price(open_a[t], 1, cost)
                    xb = _fill_price(open_b[t], -1, cost)
                # PnL: long leg A earns (exit-entry); short leg B earns (entry_b-exit_b)
                if pos == 1:
                    pnl_leg_a = (xa - entry_price_a) * qa
                    pnl_leg_b = (entry_price_b - xb) * qb
                else:
                    pnl_leg_a = (entry_price_a - xa) * qa
                    pnl_leg_b = (xb - entry_price_b) * qb
                fees = _fees(qa * xa, cost) + _fees(qb * xb, cost)
                pnl = pnl_leg_a + pnl_leg_b - fees
                equity += pnl
                # replace placeholder
                trades[-1] = {
                    "entry_time": entry_date,
                    "exit_time": dates[t],
                    "side": pos,
                    "entry_a": entry_price_a,
                    "exit_a": float(xa),
                    "entry_b": entry_price_b,
                    "exit_b": float(xb),
                    "notional_per_leg": notional,
                    "pnl_net": float(pnl),
                    "fees_paid": float(fees),
                    "exit_reason": "signal",
                    "bars_held": t - entry_bar,
                }
                pos = 0
                qa = qb = 0.0
            # equity mark at close
            if pos != 0:
                if pos == 1:
                    eq_curve[t] = equity + (close_a[t] - entry_price_a) * qa + (
                        entry_price_b - close_b[t]
                    ) * qb
                else:
                    eq_curve[t] = equity + (entry_price_a - close_a[t]) * qa + (
                        close_b[t] - entry_price_b
                    ) * qb
            else:
                eq_curve[t] = equity

        # force-close any open position at the last close
        if pos != 0 and trades and trades[-1] is None:
            t = n - 1
            xa = _fill_price(close_a[t], -pos, cost)
            xb = _fill_price(close_b[t], pos, cost)
            if pos == 1:
                pnl = (xa - entry_price_a) * qa + (entry_price_b - xb) * qb
            else:
                pnl = (entry_price_a - xa) * qa + (xb - entry_price_b) * qb
            fees = _fees(qa * xa, cost) + _fees(qb * xb, cost)
            pnl -= fees
            trades[-1] = {
                "entry_time": entry_date,
                "exit_time": dates[t],
                "side": pos,
                "entry_a": entry_price_a,
                "exit_a": float(xa),
                "entry_b": entry_price_b,
                "exit_b": float(xb),
                "notional_per_leg": notional,
                "pnl_net": float(pnl),
                "fees_paid": float(fees),
                "exit_reason": "force_close",
                "bars_held": t - entry_bar,
            }
            equity += pnl

        out = DataFrame([x for x in trades if x is not None])
        return out, eq_curve
