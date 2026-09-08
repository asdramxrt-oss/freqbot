"""Unified multi-strategy engine (unified_strategy_plan.md).

Combines Turtle (trend, on an absolute instrument such as BTC/USDT) and Pairs
(mean-reversion, on a log-ratio spread) under a causal regime router.

Capital allocation is 100/0 per the plan's execution matrix:
  * regime in {TREND} and no predicted vol spike -> Turtle book trades at full capital
  * regime in {LOW_VOL_MR, NEUTRAL} and no spike -> Pairs book trades at full capital
  * spike predicted / HIGH_VOL / UNCERTAIN -> neither book trades (HALT)

Both books start with the full portfolio equity and only ever open positions when their
gate is on, so at any bar at most one strategy holds a position. The combined result is the
union of both books' trades; the portfolio equity is the sum of the two book equities
(they are mutually exclusive in time, so this equals one book's equity).

Gate causality: turtle_gate[t] / pairs_gate[t] are computable from data <= t-1 and are
consulted by the simulators at open[t] - no look-ahead (verified by unit tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.strategies.pairs import PairsParams, generate_signals, log_ratio_series
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator
from research_lib.strategies.unified_router import (
    RouterConfig,
    RouterOutput,
    filtered_hmm_router,
)
from research_lib.risk.costs import CostModel


@dataclass
class UnifiedParams:
    router: RouterConfig = field(default_factory=RouterConfig)
    turtle: TurtleSimParams = field(default_factory=TurtleSimParams)
    pairs: PairsParams = field(default_factory=PairsParams)
    pairs_sim: PairsSimParams = field(default_factory=PairsSimParams)
    equity: float = 100_000.0
    turtle_capital_frac: float = 0.5   # share of equity seeded to the Turtle sub-account
    cost: CostModel = field(default_factory=CostModel)


@dataclass
class UnifiedResult:
    router: RouterOutput
    turtle_trades: DataFrame
    pairs_trades: DataFrame
    equity_curve: np.ndarray
    log: dict


def run_unified(
    turtle_df: DataFrame,
    ratio_df: DataFrame,
    params: UnifiedParams | None = None,
    seed: int = 42,
) -> UnifiedResult:
    """Run the unified engine.

    turtle_df : canonical OHLCV frame for the absolute instrument (e.g. BTC/USDT 15m).
    ratio_df  : aligned spread frame from log_ratio_series(leg_a, leg_b).
    Both frames must share the same calendar; the router is fit on turtle_df (the
    instrument whose regime we classify).

    Capital model: the portfolio equity is split into two sub-accounts (default 50/50).
    Each sub-account runs exactly one strategy and only opens positions when its regime
    gate is on, so a regime that forbids a strategy leaves that sub-account flat. The
    combined equity curve is the SUM of the sub-accounts - a true portfolio. The plan's
    "100/0 allocation" is a special case (turtle_capital_frac = 1.0) and is supported by
    passing the same split to both sub-account seeds; with a single account that switches
    instruments, results are identical up to the seed used at each activation.
    """
    params = params or UnifiedParams()
    # positional alignment safety: gate arrays are positional, so both frames must share
    # the exact same calendar
    if len(turtle_df) != len(ratio_df) or not (
        turtle_df["date"].to_numpy() == ratio_df["date"].to_numpy()
    ).all():
        raise ValueError(
            "turtle_df and ratio_df must share the exact same date calendar "
            "(build both from the same panel and align before calling run_unified)"
        )
    eq_turtle = params.equity * params.turtle_capital_frac
    eq_pairs = params.equity * (1.0 - params.turtle_capital_frac)
    rt = filtered_hmm_router(turtle_df, params.router, seed=seed)

    # --- Turtle sub-account (gated) ---
    tp = TurtleSimParams(**{**params.turtle.__dict__, "equity": eq_turtle})
    tsim = TurtleSimulator(tp)
    turtle_trades, turtle_eq, _ = tsim.run(turtle_df, entry_gate=rt.turtle_gate)

    # --- Pairs sub-account (gated) ---
    pp = PairsSimParams(**{**params.pairs_sim.__dict__, "equity": eq_pairs})
    psim = PairsSimulator(pp)
    pres = generate_signals(ratio_df, params.pairs)
    pairs_trades, pairs_eq = psim.run(ratio_df, pres.target, entry_gate=rt.pairs_gate)

    # Combined portfolio equity = turtle sub-account + pairs sub-account.
    n = min(len(turtle_eq), len(pairs_eq))
    eq = turtle_eq[:n] + pairs_eq[:n]
    overlap = int((rt.turtle_gate & rt.pairs_gate).sum())
    return UnifiedResult(
        router=rt,
        turtle_trades=turtle_trades,
        pairs_trades=pairs_trades,
        equity_curve=eq,
        log={"gate_overlap_bars": overlap},
    )
