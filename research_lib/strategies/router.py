"""Regime router (Phase 9): decides which strategy subsystem may trade each bar.

Design constraints:
  * Router input at bar t may ONLY use information available at bar t (regime labels must be
    walk-forward outputs - never hindsight).
  * Routing policy is configurable; the default policy:
        TREND        -> route to Turtle
        LOW_VOL_MR   -> route to OU
        HIGH_VOL     -> NO TRADE (risk-off) or REDUCED exposure
        NEUTRAL      -> allow only a base strategy if the strategy has independent validation,
                        else NO TRADE
        UNCERTAIN    -> NO TRADE
  * Oracle router exists ONLY as a research diagnostic upper bound (never production).

The router never generates orders by itself; it returns a *routing decision* per bar that the
risk/execution layer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

REGIMES = ["TREND", "LOW_VOL_MR", "HIGH_VOL", "NEUTRAL", "UNCERTAIN"]


@dataclass
class RouterPolicy:
    route: dict[str, str] = field(
        default_factory=lambda: {
            "TREND": "turtle",
            "LOW_VOL_MR": "ou",
            "HIGH_VOL": "no_trade",
            "NEUTRAL": "no_trade",
            "UNCERTAIN": "no_trade",
        }
    )
    # turtle allowed when label in turtle_ok_regimes (used by router evals)
    turtle_ok: tuple[str, ...] = ("TREND", "NEUTRAL")
    ou_ok: tuple[str, ...] = ("LOW_VOL_MR", "NEUTRAL")


@dataclass
class RouterResult:
    route: Series  # 'turtle' | 'ou' | 'no_trade' per bar (causal)
    oracle_route: Series  # diagnostic only
    n_active: int


def build_causal_router(
    regime_labels: Series,
    policy: RouterPolicy | None = None,
    *,
    oracle_turtle: Series | None = None,
    oracle_ou: Series | None = None,
) -> RouterResult:
    """Build causal routing + oracle diagnostics.

    regime_labels: walk-forward HMM labels aligned to bars (already causal).
    oracle_turtle/oracle_ou: boolean series (True where the strategy WOULD be profitable with
    hindsight) used ONLY to compute the oracle upper-bound diagnostic.
    """
    policy = policy or RouterPolicy()
    labels = regime_labels.fillna("UNCERTAIN").astype(str)
    route = labels.map(policy.route).fillna("no_trade").astype(str)

    # oracle: best possible routing given hindsight profitability (diagnostic only)
    oracle = pd.Series("no_trade", index=labels.index, dtype=object)
    if oracle_turtle is not None and oracle_ou is not None:
        # where both could be active choose the one with positive hindsight edge
        oracle.loc[oracle_turtle.fillna(False).to_numpy()] = "turtle"
        oracle.loc[oracle_ou.fillna(False).to_numpy() & (oracle == "no_trade")] = "ou"
    return RouterResult(
        route=route,
        oracle_route=oracle,
        n_active=int((route != "no_trade").sum()),
    )


def merge_router_with_targets(
    turtle_target: np.ndarray,
    ou_target: np.ndarray,
    route: Series,
) -> np.ndarray:
    """Merge per-strategy target arrays into one target array gated by the router.

    Where route=='turtle' the turtle target applies; 'ou' the OU target; else flat.
    """
    n = len(route)
    if len(turtle_target) != n or len(ou_target) != n:
        raise ValueError("target/route length mismatch")
    out = np.zeros(n, dtype=int)
    r = route.to_numpy()
    for i in range(n):
        if r[i] == "turtle":
            out[i] = int(turtle_target[i])
        elif r[i] == "ou":
            out[i] = int(ou_target[i])
    return out
