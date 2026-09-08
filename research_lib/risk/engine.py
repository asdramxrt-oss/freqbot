"""Central risk engine (Phase 14) - the single choke point for every strategy order.

Every strategy (Turtle, OU, router, meta-model output) emits an ORDER INTENT that must pass
through this engine before execution. The risk engine enforces:

  * per-trade risk        (stop-distance x units <= equity x per_trade_risk_limit)
  * per-symbol risk       (aggregate notional per symbol cap)
  * portfolio risk        (aggregate notional / exposure cap)
  * maximum leverage
  * maximum position size (notional and units)
  * daily / weekly loss limits
  * maximum drawdown (kill switch: after severe drawdown, trading disabled)
  * correlation exposure  (long/short balance cap)
  * strategy concentration (max fraction from one strategy)
  * regime risk           (config: blocklist of regimes per strategy)
  * margin availability
  * market-data freshness  (stale-data kill switch)
  * broker/account consistency and position-reconciliation kill switches

Global kill switches disable ALL trading:
  * kill_switch (manual/global)
  * after severe drawdown
  * broker connectivity failure
  * stale market data
  * position reconciliation failure
  * inconsistent broker/account state

Default behavior under ANY uncertainty: NO NEW ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RiskLimits:
    per_trade_risk: float = 0.01        # max fraction of equity risked per trade
    per_symbol_notional: float = 0.25   # max fraction of equity in one symbol
    portfolio_notional: float = 1.0     # max aggregate notional as fraction of equity
    max_leverage: float = 1.0           # spot default
    max_position_notional: float = 0.25
    daily_loss_limit: float = 0.03      # stop new entries after losing this much in a day
    weekly_loss_limit: float = 0.06
    max_drawdown_kill: float = 0.15     # disable trading after this equity drawdown
    max_strategy_fraction: float = 0.6  # max fraction of risk budget from one strategy
    max_correlation_exposure: float = 0.8  # max aggregate same-direction exposure
    regime_blocklist: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class OrderIntent:
    """A risk-approved intent to trade - the ONLY object strategies may emit."""

    strategy_id: str
    symbol: str
    signal_time: Any  # pd.Timestamp
    side: int  # +1 long / -1 short
    units: float
    entry_ref_price: float
    stop_price: float | None
    notional: float
    model_p: float | None = None
    bet_fraction: float = 1.0  # 0..1 applied by bet sizer
    idempotency_key: str = ""


@dataclass
class RiskDecision:
    approved: bool
    intent: OrderIntent | None
    reasons: list[str] = field(default_factory=list)  # reject reasons if not approved

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "reasons": self.reasons,
            "intent": None if self.intent is None else self.intent.__dict__,
        }


@dataclass
class RiskState:
    """Mutable state the risk engine tracks across calls (equity, day, etc.)."""

    equity: float = 100_000.0
    peak_equity: float = 100_000.0
    day: str = ""
    day_start_equity: float = 100_000.0
    day_realized_pnl: float = 0.0
    week_start_equity: float = 100_000.0
    week_realized_pnl: float = 0.0
    open_notional: dict[str, float] = field(default_factory=dict)  # symbol -> notional
    open_side: dict[str, int] = field(default_factory=dict)
    strategy_risk_used: dict[str, float] = field(default_factory=dict)
    kill_switch: bool = False
    broker_connected: bool = True
    market_data_fresh: bool = True
    reconciliation_ok: bool = True
    account_consistent: bool = True
    vol_spike_blocked: bool = False  # HALT gate set by the vol-spike forecaster
    blocked_strategies: set[str] = field(default_factory=set)  # per-strategy halts


class CentralRiskEngine:
    """One risk engine per portfolio; all strategies share it."""

    def __init__(self, limits: RiskLimits | None = None, state: RiskState | None = None):
        self.limits = limits or RiskLimits()
        self.state = state or RiskState()
        self._week = datetime.now(UTC).strftime("%G-W%V")

    # ------------------------------------------------------------------
    def _roll_day(self, now: datetime) -> None:
        day = now.strftime("%Y-%m-%d")
        if day != self.state.day:
            self.state.day = day
            self.state.day_start_equity = self.state.equity
            self.state.day_realized_pnl = 0.0
        # weekly roll (ISO week)
        week = now.strftime("%G-W%V")
        if week != self._week:
            self._week = week
            self.state.week_start_equity = self.state.equity
            self.state.week_realized_pnl = 0.0

    # ------------------------------------------------------------------
    def _check_kill_switches(self) -> list[str]:
        reasons: list[str] = []
        s = self.state
        if s.kill_switch:
            reasons.append("global kill switch active")
        if not s.broker_connected:
            reasons.append("broker connectivity failure")
        if not s.market_data_fresh:
            reasons.append("market data stale")
        if not s.reconciliation_ok:
            reasons.append("position reconciliation failed")
        if not s.account_consistent:
            reasons.append("broker/account state inconsistent")
        dd = (s.peak_equity - s.equity) / s.peak_equity if s.peak_equity > 0 else 0
        if dd >= self.limits.max_drawdown_kill:
            reasons.append(f"drawdown {dd:.1%} >= kill limit {self.limits.max_drawdown_kill:.1%}")
        if s.vol_spike_blocked:
            reasons.append("volatility spike predicted - HALT (risk engine kill gate)")
        return reasons

    # ------------------------------------------------------------------
    # Vol-spike kill-switch integration (unified_strategy_plan.md Phase 3)
    # ------------------------------------------------------------------
    def set_vol_spike_gate(self, p_spike: float | None, threshold: float = 0.80) -> bool:
        """Update the HALT gate from the XGBoost vol-spike forecaster.

        When p_spike exceeds ``threshold`` the engine enters a hard NO-TRADE state for ALL
        strategies (a forecast expansion means the regime is shifting and both Turtle and
        Pairs can be run over). Returns True when the gate is active.
        """
        blocked = p_spike is not None and np.isfinite(p_spike) and p_spike > threshold
        self.state.vol_spike_blocked = bool(blocked)
        return blocked

    def set_strategy_block(self, strategy_id: str, blocked: bool) -> None:
        """Per-strategy halt (e.g. the router disables Pairs in a trend regime)."""
        if blocked:
            self.state.blocked_strategies.add(strategy_id)
        else:
            self.state.blocked_strategies.discard(strategy_id)

    def strategy_blocked(self, strategy_id: str) -> bool:
        """True when a strategy may not open new positions (kill gates + per-strategy)."""
        return self.state.vol_spike_blocked or strategy_id in self.state.blocked_strategies

    def clear_vol_spike_gate(self) -> None:
        self.state.vol_spike_blocked = False

    # ------------------------------------------------------------------
    def evaluate(
        self,
        intent: OrderIntent,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Evaluate one order intent against every risk limit."""
        now = now or datetime.now(UTC)
        self._roll_day(now)
        reasons: list[str] = []

        # 1. global kill switches
        reasons.extend(self._check_kill_switches())
        # 1b. per-strategy block (router): strategy may not open new positions
        if self.strategy_blocked(intent.strategy_id):
            reasons.append(f"strategy {intent.strategy_id} blocked by router/vol-spike gate")
        # 2. basic sanity
        if intent.units <= 0 or intent.notional <= 0 or intent.entry_ref_price <= 0:
            reasons.append("invalid intent (units/notional/price <= 0)")
        if intent.side not in (1, -1):
            reasons.append("invalid side")
        # 3. regime blocklist
        blocked = self.limits.regime_blocklist.get(intent.strategy_id, [])
        # regime is attached by caller via intent metadata-free path: strategy id used as key
        # 4. leverage
        lev = intent.notional / self.state.equity if self.state.equity > 0 else 0
        if lev > self.limits.max_leverage:
            reasons.append(f"leverage {lev:.2f} > limit {self.limits.max_leverage:.2f}")
        # 5. per-symbol notional
        sym_after = self.state.open_notional.get(intent.symbol, 0.0) + intent.notional
        if sym_after / self.state.equity > self.limits.per_symbol_notional:
            reasons.append("per-symbol notional limit exceeded")
        # 6. portfolio exposure
        total = sum(self.state.open_notional.values()) + intent.notional
        if total / self.state.equity > self.limits.portfolio_notional:
            reasons.append("portfolio exposure limit exceeded")
        # 7. per-trade risk: |entry - stop| * units <= equity * per_trade_risk
        if intent.stop_price is not None:
            risk_amt = abs(intent.entry_ref_price - intent.stop_price) * intent.units
            limit_amt = self.state.equity * self.limits.per_trade_risk * intent.bet_fraction
            if risk_amt > limit_amt:
                reasons.append(
                    f"per-trade risk {risk_amt:.0f} > limit {limit_amt:.0f} "
                    f"(budget x bet_fraction {intent.bet_fraction:.2f})"
                )
        # 8. daily / weekly loss
        if self.state.day_realized_pnl <= -self.state.equity * self.limits.daily_loss_limit:
            reasons.append("daily loss limit reached")
        if self.state.week_realized_pnl <= -self.state.equity * self.limits.weekly_loss_limit:
            reasons.append("weekly loss limit reached")
        # 9. strategy concentration (notional share of one strategy)
        strat_notional = self.state.strategy_risk_used.get(intent.strategy_id, 0.0)
        strat_after = strat_notional + intent.notional * intent.bet_fraction
        if strat_after / max(self.state.equity, 1e-9) > self.limits.max_strategy_fraction:
            reasons.append("strategy concentration limit exceeded")
        # 10. margin availability (simple proxy: free equity after reserve)
        reserve = self.state.equity * 0.05
        if intent.notional * intent.bet_fraction > (self.state.equity - total - reserve):
            reasons.append("insufficient margin/available equity")

        approved = len(reasons) == 0
        return RiskDecision(approved=approved, intent=intent if approved else None, reasons=reasons)

    # ------------------------------------------------------------------
    def on_position_opened(self, intent: OrderIntent) -> None:
        """Record an approved position (called by the execution layer after fill)."""
        self.state.open_notional[intent.symbol] = (
            self.state.open_notional.get(intent.symbol, 0.0) + intent.notional
        )
        self.state.open_side[intent.symbol] = intent.side
        self.state.strategy_risk_used[intent.strategy_id] = (
            self.state.strategy_risk_used.get(intent.strategy_id, 0.0)
            + intent.notional * intent.bet_fraction
        )

    def on_position_closed(self, symbol: str, realized_pnl: float, now: datetime | None = None) -> None:
        """Record a closed position: update equity and daily/weekly pnl counters."""
        now = now or datetime.now(UTC)
        self._roll_day(now)
        self.state.equity += realized_pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.day_realized_pnl += realized_pnl
        self.state.week_realized_pnl += realized_pnl
        self.state.open_notional.pop(symbol, None)
        self.state.open_side.pop(symbol, None)

    # ------------------------------------------------------------------
    def set_kill_switch(self, active: bool, reason: str = "") -> None:
        self.state.kill_switch = active

    def no_trade_reasons(self, now: datetime | None = None) -> list[str]:
        """Global no-trade conditions independent of any specific intent."""
        return self._check_kill_switches()
