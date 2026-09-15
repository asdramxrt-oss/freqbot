"""Tests for the production-path directional runner (DirectionalRunner vs
EventBacktester divergence on identical signals)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.bot.config import ExecutionConfig
from research_lib.bot.directional_runner import (
    DirectionalRunner,
    DirectionalRunnerConfig,
    build_trend_signals,
)
from research_lib.risk.backtest_engine import EngineConfig, EventBacktester
from research_lib.risk.costs import CostModel
from research_lib.strategies.directional import resample_panel

COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)
EXEC = ExecutionConfig(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def _trendy_1h(n: int = 9000, seed: int = 5) -> pd.DataFrame:
    """1h perp-like frame with alternating multi-week trends (both directions)."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    seg = 0
    while seg < n:
        length = int(rng.integers(600, 1500))
        drift[seg:seg + length] = rng.choice([0.0006, -0.0006, 0.0])
        seg += length
    prices = 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n)))
    dates = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    op = np.roll(prices, 1)
    op[0] = prices[0]
    return pd.DataFrame(
        {"date": dates, "open": op, "high": np.maximum(op, prices),
         "low": np.minimum(op, prices), "close": prices, "volume": 1.0}
    )


class TestTrendSignals:
    def test_target_and_stop_contract(self):
        cfg = DirectionalRunnerConfig(sma_bars=200)
        df4 = resample_panel(_trendy_1h(), "4h")
        target, stop = build_trend_signals(df4, cfg)
        assert set(np.unique(target)) <= {-1.0, 0.0, 1.0}
        # stop present exactly while in a position
        in_pos = target != 0
        assert np.isnan(stop[~in_pos]).all()
        assert np.isfinite(stop[in_pos]).all()
        # on ENTRY bars (fresh flat -> position transition) the stop must be on the
        # protective side of the fill reference (later bars may trade through it -
        # that is what the stop is for)
        opens = df4["open"].to_numpy(dtype=float)
        prev = np.roll(target, 1)
        prev[0] = 0.0
        entries = in_pos & (prev == 0)
        long_entries = entries & (target == 1)
        short_entries = entries & (target == -1)
        assert (stop[long_entries] < opens[long_entries]).all()
        assert (stop[short_entries] > opens[short_entries]).all()

    def test_truncation_invariance(self):
        cfg = DirectionalRunnerConfig(sma_bars=200)
        df4 = resample_panel(_trendy_1h(), "4h")
        full, _ = build_trend_signals(df4, cfg)
        for cut in (1500, 2500):
            part, _ = build_trend_signals(df4.iloc[:cut], cfg)
            assert (full[:cut] == part).all()


class TestDirectionalRunner:
    def _run(self, cfg=None):
        cfg = cfg or DirectionalRunnerConfig()
        r = DirectionalRunner(cfg, exec_cfg=EXEC)
        return r, r.run(_trendy_1h())

    def test_full_cycle_clean(self):
        r, res = self._run()
        assert res["n_trades"] > 0
        assert res["kill_switch"] is False
        assert res["reconciliation_ok"] is True
        assert res["book_flat"] is True
        assert r.broker.positions() == {}
        actions = [e["action"] for e in res["log"]]
        assert "close_failed_abort" not in actions
        assert "reconciliation_failed" not in actions

    def test_runner_matches_research_backtest(self):
        """Same signals through the production chain must land within ~3% of the
        EventBacktester (taker) - the carry runner holds a 2% bar on the same chain."""
        cfg = DirectionalRunnerConfig()
        df4 = resample_panel(_trendy_1h(), "4h")
        target, stop = build_trend_signals(df4, cfg)

        from research_lib.bot.directional_runner import MARGIN_SAFE_NOTIONAL

        bt = EventBacktester(
            EngineConfig(equity=cfg.equity, cost=COST, risk_per_trade=cfg.risk_per_trade,
                         max_leverage=min(cfg.leverage, MARGIN_SAFE_NOTIONAL),
                         allow_short=True)
        )
        _trades, eq = bt.run(df4, target, stop=stop)
        research_net = float(eq[-1] - cfg.equity)

        r = DirectionalRunner(cfg, exec_cfg=EXEC)
        res = r.run(_trendy_1h())
        assert res["net_pnl"] == pytest.approx(research_net, rel=0.03)

    def test_short_side_trades(self):
        _r, res = self._run()
        sides = set(res["trades"]["side"].unique()) if res["n_trades"] else set()
        assert -1 in sides and 1 in sides  # futures: both directions are traded
