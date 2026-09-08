"""Ablation studies (Phase 19/27) and transaction-cost stress (Phase 20).

Ablation answers whether each subsystem adds genuine OOS value:
  BASELINE (single strategy) -> +FD -> +VOL -> +REGIME -> +ML -> +RISK -> FULL SYSTEM
  and removal tests: FULL - ML, FULL - HMM, FULL - OU, FULL - Turtle.

Cost stress answers whether a strategy survives realistic costs:
  base -> 1.5x -> 2x -> 3x cost multipliers. A strategy that only works at ideal costs is
  rejected (spec section 19).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel
from research_lib.validation.metrics import summarize_backtest


@dataclass
class AblationResult:
    variant: str
    trades: DataFrame
    metrics: dict
    equity_curve: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {"variant": self.variant, **self.metrics}


def run_ablation(
    variants: dict[str, Callable[[], tuple[DataFrame, np.ndarray]]],
    timeframe: str = "1h",
    start_equity: float = 100_000.0,
) -> list[AblationResult]:
    """Run named strategy variants and summarize each.

    variants: name -> fn() returning (trades_df, equity_curve).
    """
    out: list[AblationResult] = []
    for name, fn in variants.items():
        trades, eq = fn()
        m = summarize_backtest(trades, None, timeframe=timeframe, start_equity=start_equity)
        out.append(AblationResult(variant=name, trades=trades, metrics=m, equity_curve=eq))
    return out


def ablation_to_frame(results: list[AblationResult]) -> DataFrame:
    return DataFrame([r.to_dict() for r in results])


# ---------------------------------------------------------------------------
# transaction cost stress
# ---------------------------------------------------------------------------
def cost_stress(
    run_fn: Callable[[CostModel], tuple[DataFrame, np.ndarray]],
    timeframe: str = "1h",
    start_equity: float = 100_000.0,
    multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
) -> DataFrame:
    """Run the same strategy under 1x/1.5x/2x/3x cost multipliers.

    run_fn: callable(cost: CostModel) -> (trades_df, equity_curve). The cost model passed
    has its fee/spread/slippage multiplied by the multiplier.
    """
    rows = []
    base = CostModel()
    for mult in multipliers:
        cost = CostModel(
            maker_fee=base.maker_fee * mult,
            taker_fee=base.taker_fee * mult,
            spread_bps=base.spread_bps * mult,
            slippage_bps=base.slippage_bps * mult,
            funding_rate=base.funding_rate,
            latency_bars=base.latency_bars,
        )
        trades, eq = run_fn(cost)
        m = summarize_backtest(trades, None, timeframe=timeframe, start_equity=start_equity)
        rows.append({"cost_multiplier": mult, **m})
    return DataFrame(rows)


def cost_stress_verdict(cost_rows: DataFrame, tolerance_frac: float = 0.3) -> dict:
    """A strategy that collapses under 3x costs is rejected.

    tolerance_frac: acceptable fraction of base net profit retained at max cost.
    """
    if len(cost_rows) == 0:
        return {"accepted": False, "reason": "no data"}
    base = cost_rows[cost_rows["cost_multiplier"] == 1.0].iloc[0]
    worst = cost_rows.iloc[-1]  # highest multiplier
    base_pnl = base.get("total_pnl", np.nan)
    worst_pnl = worst.get("total_pnl", np.nan)
    if not np.isfinite(base_pnl) or not np.isfinite(worst_pnl):
        return {"accepted": False, "reason": "non-finite pnl"}
    if base_pnl <= 0:
        return {"accepted": False, "reason": "unprofitable at base costs"}
    retained = worst_pnl / base_pnl if base_pnl != 0 else np.nan
    accepted = retained >= tolerance_frac
    return {
        "accepted": accepted,
        "base_pnl": float(base_pnl),
        "max_cost_pnl": float(worst_pnl),
        "retained_frac": float(retained) if np.isfinite(retained) else np.nan,
        "reason": "ok" if accepted else "profit collapses under cost stress",
    }
