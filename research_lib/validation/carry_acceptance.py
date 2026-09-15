"""Walk-forward validation + acceptance gates for funding-rate carry.

Carry has no fitted parameters, so "walk-forward" here means contiguous out-of-sample
time folds: the trade is replayed independently on each fold and must be net-positive
in EVERY fold, survive cost stress, and not depend on a single period. The gates mirror
the project's acceptance spec (validation/acceptance.py, spec section 41) restricted to
what applies to a parameter-free, non-directional trade:

  1. data sufficiency        (>= min_prints funding prints, >= min_folds folds)
  2. walk-forward positive   (every fold net positive)
  3. cost stress             (net positive at 1.5x and 2x cost components)
  4. no concentration        (best fold < max_fold_share of total net)
  5. tail behaviour          (no liquidation; worst funding drawdown within budget)

Holdout hygiene: this module only ever reads research/data (never research/data_holdout).
The trade is parameter-free, so the full research funding history is usable without
fitting leakage - the folds exist to prove time-consistency, not to tune anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel
from research_lib.strategies.carry import CarryConfig, CarryResult, run_carry


@dataclass
class CarryGate:
    gate_id: int
    name: str
    passed: bool
    evidence: str = ""


@dataclass
class CarryAcceptance:
    symbol: str
    folds: DataFrame
    gates: list[CarryGate] = field(default_factory=list)
    verdict: str = "FAIL"

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "verdict": self.verdict,
            "gates": {g.name: ("PASS" if g.passed else f"FAIL ({g.evidence})") for g in self.gates},
            "n_folds": len(self.folds),
        }


def _stressed(cost: CostModel, mult: float) -> CostModel:
    """Scale every cost component (mirrors the project's stress-harness convention)."""
    return CostModel(
        maker_fee=cost.maker_fee * mult,
        taker_fee=cost.taker_fee * mult,
        spread_bps=cost.spread_bps * mult,
        slippage_bps=cost.slippage_bps * mult,
        funding_rate=cost.funding_rate,
        latency_bars=cost.latency_bars,
    )


def evaluate_carry_folds(
    spot: DataFrame,
    funding: DataFrame,
    cfg: CarryConfig,
    cost: CostModel,
    n_folds: int = 6,
    perp: DataFrame | None = None,
) -> DataFrame:
    """Attribute a single continuous carry run to ``n_folds`` contiguous print windows.

    Folds are windows over one continuous position (funding summed, hedge MtM and costs
    differenced from the equity curve), NOT independent replays - a real carry position
    pays entry/exit once, so this keeps fold cost accounting honest. Fold nets sum to
    the total net pnl by construction.
    """
    f = funding.sort_values("date").reset_index(drop=True).copy()
    # match run_carry's internal normalisation (naive-UTC ns) for window comparisons
    f["date"] = (
        pd.to_datetime(f["date"], utc=True, format="mixed").dt.tz_localize(None).astype("datetime64[ns]")
    )
    if len(f) < n_folds * 2:
        return DataFrame()
    full = run_carry(spot, funding, cfg, cost, perp=perp)
    curve = full.equity_curve
    if curve.empty:
        return DataFrame()
    bounds = np.linspace(0, len(f), n_folds + 1).astype(int)
    rows: list[dict] = []
    for k in range(n_folds):
        chunk = f.iloc[bounds[k] : bounds[k + 1]]
        t0, t1 = chunk["date"].iloc[0], chunk["date"].iloc[-1]
        c = curve[(curve["date"] >= t0) & (curve["date"] <= t1)]
        if c.empty:
            rows.append({"fold": k, "start": str(t0), "end": str(t1), "n_prints": 0, "net_pnl": np.nan})
            continue
        prior = curve[curve["date"] < t0]
        prev_hedge = float(prior["basis_mtm"].iloc[-1]) if len(prior) else 0.0
        prev_costs = float(prior["costs_to_date"].iloc[-1]) if len(prior) else 0.0
        funding_sum = float(c["payment"].sum())
        d_hedge = float(c["basis_mtm"].iloc[-1]) - prev_hedge
        d_costs = float(c["costs_to_date"].iloc[-1]) - prev_costs
        rows.append(
            {
                "fold": k,
                "start": str(t0),
                "end": str(t1),
                "n_prints": int(len(c)),
                "funding": funding_sum,
                "hedge_mtm_change": d_hedge,
                "costs": d_costs,
                "net_pnl": funding_sum + d_hedge - d_costs,
            }
        )
    # exit + liquidation costs are not in costs_to_date - attribute them to the last
    # fold that has prints so fold nets still sum to the total
    residual = full.total_costs - float(curve["costs_to_date"].iloc[-1])
    if residual > 0:
        for i in range(len(rows) - 1, -1, -1):
            if rows[i]["n_prints"] > 0:
                rows[i]["costs"] += residual
                rows[i]["net_pnl"] -= residual
                break
    out = DataFrame(rows)
    out["liquidated"] = full.liquidated
    out["rehedges"] = full.n_rebalances
    return out


def acceptance_gates(
    full: CarryResult,
    folds: DataFrame,
    stress: dict[float, CarryResult],
    *,
    min_folds: int = 4,
    min_prints: int = 500,
    max_fold_share: float = 0.5,
    max_funding_dd: float = 0.05,
) -> list[CarryGate]:
    """Mechanical acceptance gates for carry. ``stress``: {multiplier: CarryResult}."""
    gates: list[CarryGate] = []

    ok_folds = folds.dropna(subset=["net_pnl"]) if len(folds) else folds
    g1_ok = len(ok_folds) >= min_folds and int(ok_folds["n_prints"].sum()) >= min_prints
    gates.append(CarryGate(1, "data_sufficiency", g1_ok,
                           f"{len(ok_folds)} folds, {int(ok_folds['n_prints'].sum()) if len(ok_folds) else 0} prints"))

    g2_ok = len(ok_folds) >= min_folds and bool((ok_folds["net_pnl"] > 0).all())
    worst = ok_folds["net_pnl"].min() if len(ok_folds) else np.nan
    gates.append(CarryGate(2, "walk_forward_positive", g2_ok, f"worst fold net={worst:.0f}"))

    stress_ok = all(r.net_pnl > 0 for r in stress.values()) and len(stress) > 0
    ev = "; ".join(f"{m}x net={r.net_pnl:.0f}" for m, r in sorted(stress.items()))
    gates.append(CarryGate(3, "cost_stress", stress_ok, ev))

    if len(ok_folds) and ok_folds["net_pnl"].sum() > 0:
        share = float(ok_folds["net_pnl"].max() / ok_folds["net_pnl"].sum())
    else:
        share = 1.0
    gates.append(CarryGate(4, "no_concentration", share < max_fold_share,
                           f"best fold {share:.0%} of total net"))

    dd = 0.0
    if len(full.equity_curve):
        cum = full.equity_curve["cum_funding"]
        dd = float((cum.cummax() - cum).max() / full.final_equity) if full.final_equity else 1.0
    g5_ok = (not full.liquidated) and dd < max_funding_dd
    gates.append(CarryGate(5, "tail_behaviour", g5_ok,
                           f"liquidated={full.liquidated}, funding dd={dd:.2%}"))
    return gates


def run_carry_acceptance(
    symbol: str,
    spot: DataFrame,
    funding: DataFrame,
    cfg: CarryConfig | None = None,
    cost: CostModel | None = None,
    n_folds: int = 6,
    perp: DataFrame | None = None,
) -> CarryAcceptance:
    """Full validation bundle for one symbol: full-period run, folds, cost stress, gates."""
    cfg = cfg or CarryConfig()
    cost = cost or CostModel()
    full = run_carry(spot, funding, cfg, cost, perp=perp)
    folds = evaluate_carry_folds(spot, funding, cfg, cost, n_folds=n_folds, perp=perp)
    stress = {m: run_carry(spot, funding, cfg, _stressed(cost, m), perp=perp) for m in (1.5, 2.0)}
    gates = acceptance_gates(full, folds, stress)
    verdict = "PASS" if all(g.passed for g in gates) else "FAIL"
    return CarryAcceptance(symbol=symbol, folds=folds, gates=gates, verdict=verdict)
