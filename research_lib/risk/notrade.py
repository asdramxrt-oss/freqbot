"""No-trade logic (Phase 29) - a professional system must be allowed to do nothing.

Encapsulates the explicit NO-TRADE conditions from the spec and exposes them as a checklist
the router and meta-model consult before an order intent is even formed:

  data quality poor, spread too wide, volatility abnormal, regime uncertain,
  ML confidence low, risk budget exhausted, portfolio correlation too high,
  daily loss limit reached, broker state inconsistent, market data stale,
  strategy edge unavailable.

NO TRADE is always the default under uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class NoTradeParams:
    max_spread_bps: float = 20.0          # refuse when spread wider than this
    vol_abnormal_pct: float = 0.99        # refuse when vol percentile above this
    min_ml_confidence: float = 0.52
    max_correlation: float = 0.85


class NoTradeGate:
    def __init__(self, params: NoTradeParams | None = None):
        self.p = params or NoTradeParams()

    def evaluate_row(self, row: dict) -> list[str]:
        """Return reasons to NOT trade at a bar; empty list means trade is allowed."""
        reasons: list[str] = []
        # data quality
        if row.get("data_quality_ok") is False:
            reasons.append("data quality poor")
        # spread
        spread = row.get("spread_bps")
        if spread is not None and np.isfinite(spread) and spread > self.p.max_spread_bps:
            reasons.append(f"spread too wide ({spread:.1f} bps)")
        # abnormal volatility
        vp = row.get("volatility_percentile")
        if vp is not None and np.isfinite(vp) and vp > self.p.vol_abnormal_pct:
            reasons.append("volatility abnormal")
        # regime uncertainty
        if row.get("regime") in ("UNCERTAIN", None):
            reasons.append("regime uncertain")
        # ML confidence
        p = row.get("ml_p")
        if p is not None:
            if not np.isfinite(p) or p < self.p.min_ml_confidence:
                reasons.append("ML confidence low")
        # risk budget
        if row.get("risk_budget_exhausted"):
            reasons.append("risk budget exhausted")
        # correlation
        corr = row.get("portfolio_correlation")
        if corr is not None and np.isfinite(corr) and corr > self.p.max_correlation:
            reasons.append("portfolio correlation too high")
        # daily loss
        if row.get("daily_loss_reached"):
            reasons.append("daily loss limit reached")
        # broker/system state
        if row.get("broker_inconsistent"):
            reasons.append("broker state inconsistent")
        if row.get("data_stale"):
            reasons.append("market data stale")
        # strategy edge
        if row.get("strategy_edge_available") is False:
            reasons.append("strategy edge unavailable")
        return reasons

    def trade_allowed(self, row: dict) -> tuple[bool, list[str]]:
        reasons = self.evaluate_row(row)
        return len(reasons) == 0, reasons
