"""Unit tests for Phase 14 (central risk engine) + no-trade logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits, RiskState
from research_lib.risk.notrade import NoTradeGate, NoTradeParams


def _intent(side=1, notional=10_000, stop_dist=None, strategy="turtle", symbol="BTC/USDT"):
    return OrderIntent(
        strategy_id=strategy,
        symbol=symbol,
        signal_time=pd.Timestamp("2022-01-01", tz="UTC"),
        side=side,
        units=notional / 100.0 if stop_dist is None else notional / stop_dist,
        entry_ref_price=100.0,
        stop_price=100.0 - stop_dist if stop_dist else None,
        notional=notional,
    )


class TestRiskEngine:
    def test_clean_intent_approved(self):
        eng = CentralRiskEngine(RiskLimits())
        dec = eng.evaluate(_intent(notional=10_000))
        assert dec.approved, dec.reasons

    def test_kill_switch_blocks(self):
        eng = CentralRiskEngine(RiskLimits())
        eng.set_kill_switch(True)
        dec = eng.evaluate(_intent())
        assert not dec.approved
        assert any("kill switch" in r for r in dec.reasons)

    def test_per_trade_risk_limit(self):
        # stop distance tiny => units huge => risk over limit
        eng = CentralRiskEngine(RiskLimits(per_trade_risk=0.01))
        intent = _intent(notional=100_000, stop_dist=0.1)
        intent.units = 100_000  # |100-99.9| * 100000 = 10,000 risk > 1,000 limit
        dec = eng.evaluate(intent)
        assert not dec.approved
        assert any("per-trade risk" in r for r in dec.reasons)

    def test_daily_loss_blocks(self):
        eng = CentralRiskEngine(RiskLimits(daily_loss_limit=0.03))
        now = datetime(2022, 1, 1, tzinfo=UTC)
        eng.on_position_closed("BTC/USDT", -4_000, now=now)  # -4% on 100k
        dec = eng.evaluate(_intent(), now=now + timedelta(hours=1))
        assert not dec.approved
        assert any("daily loss" in r for r in dec.reasons)

    def test_drawdown_kill_switch(self):
        st = RiskState(equity=80_000, peak_equity=100_000)
        eng = CentralRiskEngine(RiskLimits(max_drawdown_kill=0.15), state=st)
        dec = eng.evaluate(_intent())
        assert not dec.approved
        assert any("drawdown" in r for r in dec.reasons)

    def test_per_symbol_limit(self):
        eng = CentralRiskEngine(RiskLimits(per_symbol_notional=0.1))
        eng.on_position_opened(_intent(notional=9_000))  # 9% used
        # another 9% on same symbol exceeds the 10% cap (total 18%)
        dec = eng.evaluate(_intent(notional=9_000))
        assert not dec.approved
        assert any("per-symbol" in r for r in dec.reasons)

    def test_leverage_limit(self):
        eng = CentralRiskEngine(RiskLimits(max_leverage=1.0))
        dec = eng.evaluate(_intent(notional=150_000))  # 150% of 100k equity
        assert not dec.approved
        assert any("leverage" in r for r in dec.reasons)

    def test_broker_failure_blocks(self):
        st = RiskState()
        st.broker_connected = False
        eng = CentralRiskEngine(state=st)
        assert not eng.evaluate(_intent()).approved


class TestNoTradeGate:
    def test_clean_row_allows(self):
        gate = NoTradeGate()
        allowed, reasons = gate.trade_allowed({"regime": "TREND", "data_quality_ok": True})
        assert allowed
        assert reasons == []

    def test_uncertain_regime_blocks(self):
        gate = NoTradeGate()
        allowed, reasons = gate.trade_allowed({"regime": "UNCERTAIN"})
        assert not allowed
        assert "regime uncertain" in reasons

    def test_wide_spread_blocks(self):
        gate = NoTradeGate(NoTradeParams(max_spread_bps=10))
        allowed, _ = gate.trade_allowed({"spread_bps": 25.0})
        assert not allowed

    def test_low_ml_confidence_blocks(self):
        gate = NoTradeGate()
        allowed, reasons = gate.trade_allowed({"ml_p": 0.50, "regime": "TREND"})
        assert not allowed
        assert "ML confidence low" in reasons

    def test_abnormal_vol_blocks(self):
        gate = NoTradeGate()
        allowed, _ = gate.trade_allowed({"volatility_percentile": 0.995, "regime": "TREND"})
        assert not allowed

    def test_data_stale_blocks(self):
        gate = NoTradeGate()
        allowed, reasons = gate.trade_allowed({"data_stale": True, "regime": "TREND"})
        assert not allowed
