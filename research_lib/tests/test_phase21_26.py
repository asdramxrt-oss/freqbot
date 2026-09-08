"""Unit tests for Phases 21-26: execution engine, Redis locking, idempotency, order state
machine, position reconciliation, paper trading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.execution.engine import ExecutionEngine
from research_lib.execution.order_state import (
    IllegalTransition,
    OrderRecord,
    OrderState,
    OrderStateStore,
    make_idempotency_key,
)
from research_lib.execution.paper import PaperConfig, PaperTradingSimulator, divergence_report
from research_lib.execution.reconciliation import FailureInjector, PositionReconciler
from research_lib.execution.redis_lock import DistributedLock, IdempotencyGuard
from research_lib.risk.costs import CostModel
from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits

try:
    import fakeredis

    _HAS_FAKE = True
except Exception:  # pragma: no cover
    _HAS_FAKE = False


def _intent(strategy="turtle", side=1, signal_time="2022-01-01T00:00:00Z", units=1.0):
    return OrderIntent(
        strategy_id=strategy,
        symbol="BTC/USDT",
        signal_time=signal_time,
        side=side,
        units=units,
        entry_ref_price=100.0,
        stop_price=99.0,
        notional=100.0,
    )


class _FakeBroker:
    def __init__(self):
        self.submits = 0
        self.fetches = 0

    def submit(self, intent):
        self.submits += 1
        return {"order_id": f"B{self.submits}", "status": "open"}

    def fetch(self, order_id):
        self.fetches += 1
        return {"status": "closed", "filled": 1.0, "remaining": 0.0}

    def cancel(self, order_id):
        return {"status": "canceled"}


# ------------------------------------------------------------------ Phase 31
class TestRedisLock:
    @pytest.mark.skipif(not _HAS_FAKE, reason="fakeredis not installed")
    def test_acquire_exclusive(self):
        r = fakeredis.FakeRedis()
        a = DistributedLock(r, "res1", token="tok-a")
        b = DistributedLock(r, "res1", token="tok-b")
        assert a.acquire()
        assert not b.acquire()  # already held
        assert a.is_owner()
        assert not b.is_owner()

    @pytest.mark.skipif(not _HAS_FAKE, reason="fakeredis not installed")
    def test_owner_only_release(self):
        r = fakeredis.FakeRedis()
        a = DistributedLock(r, "res1", token="tok-a")
        b = DistributedLock(r, "res1", token="tok-b")
        a.acquire()
        # b cannot release a's lock
        assert not b.release()
        assert a.is_owner()
        # owner releases
        assert a.release()
        assert not a.is_owner()

    @pytest.mark.skipif(not _HAS_FAKE, reason="fakeredis not installed")
    def test_ttl_expiry(self):
        r = fakeredis.FakeRedis()
        a = DistributedLock(r, "res1", token="tok-a", ttl_ms=50)
        assert a.acquire()
        import time

        time.sleep(0.08)
        assert not a.is_owner()  # expired

    @pytest.mark.skipif(not _HAS_FAKE, reason="fakeredis not installed")
    def test_idempotency_guard_single_claim(self):
        r = fakeredis.FakeRedis()
        g = IdempotencyGuard(r)
        assert g.try_claim("key1")
        assert not g.try_claim("key1")  # second worker cannot claim
        g.release("key1")
        assert g.try_claim("key1")


# ------------------------------------------------------------------ Phase 32
class TestOrderState:
    def test_legal_flow(self):
        rec = OrderRecord("o1", "k1", "turtle", "BTC/USDT", "t", 1, 10.0)
        rec.transition(OrderState.RISK_APPROVED)
        rec.transition(OrderState.ORDER_PENDING)
        rec.transition(OrderState.ORDER_SUBMITTED)
        rec.transition(OrderState.PARTIALLY_FILLED)
        rec.transition(OrderState.FILLED)
        assert rec.state == OrderState.FILLED
        assert len(rec.transitions) == 5

    def test_illegal_transition_raises(self):
        rec = OrderRecord("o1", "k1", "turtle", "BTC/USDT", "t", 1, 10.0)
        with pytest.raises(IllegalTransition):
            rec.transition(OrderState.FILLED)  # cannot jump straight to FILLED

    def test_filled_is_terminal(self):
        rec = OrderRecord("o1", "k1", "t", "X", "t", 1, 1.0)
        rec.transition(OrderState.RISK_APPROVED)
        rec.transition(OrderState.ORDER_PENDING)
        rec.transition(OrderState.ORDER_SUBMITTED)
        rec.transition(OrderState.FILLED)
        with pytest.raises(IllegalTransition):
            rec.transition(OrderState.CANCELLED)

    def test_idempotency_key_deterministic(self):
        a = make_idempotency_key("turtle", "BTC/USDT", "2022-01-01", 1, "v1")
        b = make_idempotency_key("turtle", "BTC/USDT", "2022-01-01", 1, "v1")
        c = make_idempotency_key("turtle", "BTC/USDT", "2022-01-01", -1, "v1")
        assert a == b
        assert a != c

    def test_store_deduplicates(self):
        store = OrderStateStore()
        rec1 = OrderRecord("o1", "same-key", "t", "X", "t", 1, 1.0)
        rec2 = OrderRecord("o2", "same-key", "t", "X", "t", 1, 1.0)
        _, c1 = store.create_or_get(rec1)
        existing, c2 = store.create_or_get(rec2)
        assert c1 and not c2
        assert existing.order_id == "o1"


# ------------------------------------------------------------------ Phase 30
class TestExecutionEngine:
    def test_risk_rejection_never_submits(self):
        risk = CentralRiskEngine(RiskLimits())
        risk.set_kill_switch(True)
        broker = _FakeBroker()
        eng = ExecutionEngine(risk, OrderStateStore(), broker)
        res = eng.execute(_intent())
        assert not res.submitted
        assert broker.submits == 0
        assert res.order.state == OrderState.REJECTED

    def test_successful_submit_requires_confirmation(self):
        risk = CentralRiskEngine(RiskLimits())
        broker = _FakeBroker()
        eng = ExecutionEngine(risk, OrderStateStore(), broker)
        res = eng.execute(_intent())
        assert res.submitted
        assert res.order.state == OrderState.PARTIALLY_FILLED  # ack open, NOT filled
        # fill only after broker confirmation via reconciliation
        state = eng.reconcile_open(res.order)
        assert state == OrderState.FILLED

    def test_duplicate_signal_single_submit(self):
        risk = CentralRiskEngine(RiskLimits())
        broker = _FakeBroker()
        eng = ExecutionEngine(risk, OrderStateStore(), broker)
        r1 = eng.execute(_intent())
        r2 = eng.execute(_intent())  # same signal -> same idempotency key
        assert r1.submitted and not r2.submitted
        assert broker.submits == 1
        assert "duplicate" in r2.message

    def test_broker_error_fails_safe(self):
        risk = CentralRiskEngine(RiskLimits())
        broker = _FakeBroker()
        broker.submit = lambda intent: (_ for _ in ()).throw(TimeoutError("timeout"))
        eng = ExecutionEngine(risk, OrderStateStore(), broker)
        res = eng.execute(_intent())
        assert not res.submitted
        assert res.order.state == OrderState.ERROR


# ------------------------------------------------------------------ Phase 33
class TestReconciliation:
    def test_mismatch_sets_kill_switch(self):
        risk = CentralRiskEngine(RiskLimits())
        rec = PositionReconciler(risk)
        res = rec.reconcile({"BTC/USDT": 1.0}, {"BTC/USDT": 0.5})
        assert not res.consistent
        assert risk.state.kill_switch  # new orders stopped
        assert not risk.state.reconciliation_ok

    def test_match_keeps_trading(self):
        risk = CentralRiskEngine(RiskLimits())
        rec = PositionReconciler(risk)
        res = rec.reconcile({"BTC/USDT": 1.0}, {"BTC/USDT": 1.0})
        assert res.consistent
        assert not risk.state.kill_switch

    def test_resume_after_reconcile(self):
        risk = CentralRiskEngine(RiskLimits())
        rec = PositionReconciler(risk)
        rec.reconcile({"A": 1.0}, {"A": 0.0})
        assert risk.state.kill_switch
        # fix the position, then resume
        rec.reconcile({"A": 1.0}, {"A": 1.0})
        rec.resume()
        assert not risk.state.kill_switch


# ------------------------------------------------------------------ Phase 34
class TestFailureInjection:
    def test_network_down_no_submit(self):
        fx = FailureInjector(network_down=True)
        with pytest.raises(TimeoutError):
            fx.submit(_intent())

    def test_partial_fill_reconciliation(self):
        risk = CentralRiskEngine(RiskLimits())
        broker = _FakeBroker()
        fx = FailureInjector(partial_fill_mode=True)
        eng = ExecutionEngine(risk, OrderStateStore(), broker)
        res = eng.execute(_intent())
        # order is open at broker with partial fill; reconciliation must reflect it
        broker.fetch = fx.fetch
        state = eng.reconcile_open(res.order)
        assert state == OrderState.PARTIALLY_FILLED


# ------------------------------------------------------------------ Phase 24
class TestPaperTrading:
    def _frame(self, n=2000, seed=0):
        rng = np.random.RandomState(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, n)))
        dates = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": 1.0,
            }
        )

    def test_replay_produces_trades(self):
        df = self._frame()
        target = np.where(np.arange(len(df)) % 100 < 50, 1, 0)
        stop = df["close"].to_numpy() * 0.97
        sim = PaperTradingSimulator(PaperConfig(capital_scale=0.1))
        out = sim.replay(df, target, stop=stop)
        assert out["n_fills"] > 0
        assert out["capital_scale"] == 0.1
        assert len(out["equity_curve"]) == len(df)

    def test_divergence_report(self):
        df = self._frame(1500)
        target = np.where(np.arange(len(df)) % 50 < 25, 1, 0)
        sim = PaperTradingSimulator()
        paper = sim.replay(df, target)
        bt_trades = paper["trades"]
        rep = divergence_report(paper, bt_trades)
        assert rep["paper_trades"] == rep["backtest_trades"]
