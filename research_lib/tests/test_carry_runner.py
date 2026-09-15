"""Tests for the production-path carry runner (CarryRunner vs run_carry divergence)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.bot.carry_runner import CarryRunner, CarryRunnerConfig
from research_lib.bot.config import ExecutionConfig
from research_lib.risk.costs import CostModel
from research_lib.risk.engine import RiskDecision
from research_lib.strategies.carry import CarryConfig, run_carry

TAKER_COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def _spot(prices, start="2022-01-01", freq="1h"):
    dates = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0}
    )


def _funding(rates, start="2022-01-01", freq="8h"):
    dates = pd.date_range(start, periods=len(rates), freq=freq, tz="UTC")
    return pd.DataFrame({"date": dates, "funding_rate": rates})


def _flat(n_bars=6000, n_prints=1500, rate=0.0006):
    return _spot([100.0] * n_bars), _funding([rate] * n_prints)


def _exec_cfg():
    return ExecutionConfig(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


class TestProductionChain:
    def test_both_legs_fill_and_funding_flows(self):
        spot, funding = _flat()
        r = CarryRunner(CarryRunnerConfig(post_only=False, notional_fraction=0.6), _exec_cfg())
        res = r.run(spot, funding)
        assert res["gross_funding"] > 0.0
        assert res["net_pnl"] > 0.0
        assert res["total_fees"] > 0.0
        assert res["kill_switch"] is False
        assert res["reconciliation_ok"] is True
        # the run must end perfectly flat - broker and internal state agree
        assert res["book_flat"] is True
        assert r.broker.positions() == {}
        assert r.risk.state.open_notional == {}
        actions = [e["action"] for e in res["log"]]
        assert "entered" in actions and "closed" in actions
        assert "reconciliation_failed" not in actions
        assert "close_failed" not in actions
        assert "order_not_filled" not in actions

    def test_runner_matches_research_backtest_taker(self):
        """Live-vs-backtest divergence: same trade through the production chain must
        land within rounding of the research model on identical (flat) data."""
        spot, funding = _flat()
        res = CarryRunner(
            CarryRunnerConfig(post_only=False, notional_fraction=0.6), _exec_cfg()
        ).run(spot, funding)
        research = run_carry(
            spot, funding, CarryConfig(notional_fraction=0.6, post_only=False), TAKER_COST
        )
        assert res["net_pnl"] == pytest.approx(research.net_pnl, rel=0.02)
        assert res["gross_funding"] == pytest.approx(research.gross_funding, rel=0.02)

    def test_maker_net_beats_taker_net(self):
        spot, funding = _flat()
        taker = CarryRunner(
            CarryRunnerConfig(post_only=False, notional_fraction=0.6), _exec_cfg()
        ).run(spot, funding)
        maker = CarryRunner(
            CarryRunnerConfig(post_only=True, notional_fraction=0.6), _exec_cfg()
        ).run(spot, funding)
        assert maker["total_fees"] < taker["total_fees"]
        assert maker["net_pnl"] > taker["net_pnl"]

    def test_rehedge_on_gradual_rally(self):
        prices = list(np.linspace(100.0, 190.0, 6000))  # +90% over the window
        spot = _spot(prices)
        funding = _funding([0.0005] * 1500)
        r = CarryRunner(
            CarryRunnerConfig(post_only=False, notional_fraction=0.6, leverage=1.0), _exec_cfg()
        )
        res = r.run(spot, funding)
        assert res["liquidated"] is False
        assert res["n_rehedges"] >= 1
        assert res["gross_funding"] > 0.0
        # rehedge closes+reopens must actually fill - no silent divergence anywhere
        assert res["book_flat"] is True
        assert r.broker.positions() == {}
        assert "close_failed" not in [e["action"] for e in res["log"]]

    def test_closing_orders_are_not_blocked_by_exposure_limits(self):
        """Risk engine regression test: a reduce-only intent trading against the open
        side must pass even when the same notional would be rejected as a new position."""
        from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits

        risk = CentralRiskEngine(
            limits=RiskLimits(per_symbol_notional=0.3, portfolio_notional=0.3, max_leverage=0.3)
        )
        risk.state.equity = 100_000.0
        open_intent = OrderIntent(
            strategy_id="carry_spot", symbol="BTC/USDT",
            signal_time=pd.Timestamp("2024-01-01", tz="UTC"), side=1, units=1.0,
            entry_ref_price=20_000.0, stop_price=None, notional=20_000.0,
        )
        risk.on_position_opened(open_intent)
        close_intent = OrderIntent(
            strategy_id="carry_spot", symbol="BTC/USDT",
            signal_time=pd.Timestamp("2024-01-02", tz="UTC"), side=-1, units=1.0,
            entry_ref_price=20_000.0, stop_price=None, notional=20_000.0,
        )
        assert risk.evaluate(close_intent).approved is True
        # a FRESH same-side position with the same notional must still be rejected
        bigger = OrderIntent(
            strategy_id="x", symbol="ETH/USDT",
            signal_time=pd.Timestamp("2024-01-02", tz="UTC"), side=1, units=3.0,
            entry_ref_price=20_000.0, stop_price=None, notional=60_000.0,
        )
        assert risk.evaluate(bigger).approved is False

    def test_explosive_rally_liquidates_before_rehedge_can_help(self):
        prices = [100.0] * 3000 + [260.0] * 3000  # instant +160%
        spot = _spot(prices)
        funding = _funding([0.0005] * 1500)
        res = CarryRunner(
            CarryRunnerConfig(post_only=False, notional_fraction=0.6, leverage=1.0), _exec_cfg()
        ).run(spot, funding)
        assert res["liquidated"] is True
        actions = [e["action"] for e in res["log"]]
        assert "closed" in actions

    def test_failed_second_leg_unwinds_the_first(self):
        """Never run a naked leg: if the perp leg is rejected, the spot leg is unwound."""
        spot, funding = _flat()
        r = CarryRunner(CarryRunnerConfig(post_only=False, notional_fraction=0.6), _exec_cfg())
        orig_evaluate = r.risk.evaluate

        def reject_perp(intent, now=None):
            if intent.symbol.endswith(":PERP") and intent.side < 0 and not r._perp.open:
                return RiskDecision(approved=False, intent=None, reasons=["forced test rejection"])
            return orig_evaluate(intent, now)

        r.risk.evaluate = reject_perp  # type: ignore[method-assign]
        res = r.run(spot, funding)
        assert "entry_aborted_second_leg_failed" in [e["action"] for e in res["log"]]
        assert r.broker.positions() == {}  # nothing left open


class TestRealDataParity:
    def test_btc_runner_tracks_research_backtest(self):
        from pathlib import Path

        from research_lib.strategies.carry import load_funding

        root = Path(__file__).resolve().parents[2]
        funding_path = root / "research" / "data" / "funding" / "BTC_USDT_USDT_funding.csv"
        spot_path = root / "research" / "data" / "BTC_USDT-1h.feather"
        if not funding_path.exists() or not spot_path.exists():
            pytest.skip("research data not available")
        funding = load_funding(funding_path)
        spot = pd.read_feather(spot_path)
        res = CarryRunner(
            CarryRunnerConfig(post_only=False, notional_fraction=0.6), _exec_cfg()
        ).run(spot, funding)
        research = run_carry(
            spot, funding, CarryConfig(notional_fraction=0.6, post_only=False), TAKER_COST
        )
        # entry timing and fee-on-exec-price differences allow loose parity only,
        # but sign, shape and magnitude must agree
        assert res["liquidated"] == research.liquidated
        assert res["net_pnl"] == pytest.approx(research.net_pnl, rel=0.15)
        assert res["gross_funding"] == pytest.approx(research.gross_funding, rel=0.10)
