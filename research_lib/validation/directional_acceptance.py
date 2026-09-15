"""Acceptance gates for directional variants (multi-coin, futures).

Same discipline as the carry acceptance bundle: contiguous time folds over ONE
continuous portfolio run (fold nets are equity-curve differences, costs land when
incurred), then mechanical gates. A directional variant is "successful" only if ALL
gates pass - and the report always includes every variant tested, so a lucky winner
on a multiple-tested grid cannot hide.

Gates:
  1. data sufficiency     (>= min_folds usable folds, >= min_trades closed trades)
  2. walk-forward positive (every fold net > 0)
  3. cost stress           (net > 0 at 1.5x and 2x cost components)
  4. no concentration      (best coin < max_coin_share of total positive coin pnl)
  5. drawdown              (portfolio max drawdown within budget)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel
from research_lib.strategies.directional import DirectionalResult, run_panel_directional


@dataclass
class DirectionalGate:
    gate_id: int
    name: str
    passed: bool
    evidence: str = ""


@dataclass
class DirectionalAcceptance:
    name: str
    folds: DataFrame
    gates: list[DirectionalGate] = field(default_factory=list)
    verdict: str = "FAIL"
    result: DirectionalResult | None = None

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)


def _stressed(cost: CostModel, mult: float) -> CostModel:
    return CostModel(
        maker_fee=cost.maker_fee * mult,
        taker_fee=cost.taker_fee * mult,
        spread_bps=cost.spread_bps * mult,
        slippage_bps=cost.slippage_bps * mult,
    )


def evaluate_directional_folds(result: DirectionalResult, n_folds: int = 6) -> DataFrame:
    """Attribute one continuous portfolio run to ``n_folds`` contiguous bar windows."""
    eq = result.equity
    dates = result.dates
    if len(eq) < n_folds * 10 or len(dates) != len(eq):
        return DataFrame()
    bounds = np.linspace(0, len(eq), n_folds + 1).astype(int)
    rows = []
    for k in range(n_folds):
        a, b = bounds[k], bounds[k + 1] - 1
        rows.append(
            {
                "fold": k,
                "start": str(pd.Timestamp(dates[a]))[:10],
                "end": str(pd.Timestamp(dates[b]))[:10],
                "net_pnl": float(eq[b] - (eq[a - 1] if a > 0 else eq[0])),
            }
        )
    return DataFrame(rows)


def acceptance_gates(
    result: DirectionalResult,
    folds: DataFrame,
    stress: dict[float, DirectionalResult],
    *,
    min_folds: int = 5,
    min_trades: int = 30,
    max_coin_share: float = 0.5,
    max_drawdown: float = 0.25,
) -> list[DirectionalGate]:
    gates: list[DirectionalGate] = []

    ok_folds = folds.dropna(subset=["net_pnl"]) if len(folds) else folds
    g1 = len(ok_folds) >= min_folds and result.n_trades >= min_trades
    gates.append(DirectionalGate(1, "data_sufficiency", g1,
                                 f"{len(ok_folds)} folds, {result.n_trades} trades"))

    g2 = len(ok_folds) >= min_folds and bool((ok_folds["net_pnl"] > 0).all())
    worst = ok_folds["net_pnl"].min() if len(ok_folds) else np.nan
    gates.append(DirectionalGate(2, "walk_forward_positive", g2, f"worst fold net={worst:.0f}"))

    g3 = len(stress) > 0 and all(r.net_pnl > 0 for r in stress.values())
    ev = "; ".join(f"{m}x net={r.net_pnl:.0f}" for m, r in sorted(stress.items()))
    gates.append(DirectionalGate(3, "cost_stress", g3, ev))

    positive = {s: v for s, v in result.per_coin_net.items() if v > 0}
    share = max(positive.values()) / sum(positive.values()) if positive and sum(positive.values()) > 0 else 1.0
    gates.append(DirectionalGate(4, "no_concentration", share < max_coin_share,
                                 f"best coin {share:.0%} of positive coin pnl"))

    gates.append(DirectionalGate(5, "drawdown", result.max_drawdown < max_drawdown,
                                 f"max dd={result.max_drawdown:.1%}"))
    return gates


def run_directional_acceptance(
    name: str,
    panels: dict[str, DataFrame],
    targets: dict[str, np.ndarray],
    cost: CostModel,
    n_folds: int = 6,
    equity: float = 100_000.0,
) -> DirectionalAcceptance:
    """Full bundle for one variant: portfolio run, folds, 1.5x/2x cost stress, gates."""
    result = run_panel_directional(name, panels, targets, cost, equity=equity)
    folds = evaluate_directional_folds(result, n_folds=n_folds)
    stress = {
        m: run_panel_directional(name, panels, targets, _stressed(cost, m), equity=equity)
        for m in (1.5, 2.0)
    }
    gates = acceptance_gates(result, folds, stress)
    verdict = "PASS" if all(g.passed for g in gates) else "FAIL"
    return DirectionalAcceptance(name=name, folds=folds, gates=gates, verdict=verdict, result=result)
