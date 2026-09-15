"""Offline tests for the router-book live loop.

The live loop's network boundary is PerpDataFeed; everything else is tested
with a fake feed + synthetic panels. The replay machinery (build_trend_signals)
needs ~5200+ 1h bars per coin, which runs in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.bot.live_feed import FetchResult, PerpDataFeed
from research_lib.bot.router_book_live import RouterBookLive
from research_lib.bot.router_book_runner import TREND_SYMS, RouterBookConfig
from research_lib.strategies.unified_router import RouterOutput


def _frame(prices, start="2022-01-01"):
    dates = pd.date_range(start, periods=len(prices), freq="1h", tz="UTC")
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"date": dates, "open": prices, "high": prices * 1.001,
         "low": prices * 0.999, "close": prices, "volume": 1.0}
    )


def _fake_router(n: int) -> RouterOutput:
    return RouterOutput(
        regime=np.full(n, "TREND", dtype=object),
        p_spike=np.zeros(n),
        turtle_gate=np.ones(n, dtype=bool),
        pairs_gate=np.ones(n, dtype=bool),
        halt_reason=np.full(n, "", dtype=object),
    )


class _FakeFeed:
    """No network: reports zero new bars (panels already on disk)."""

    def update_panel(self, symbol, path):
        return FetchResult(symbol, 0, None)


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "data" / "futures"
    d.mkdir(parents=True)
    up = np.linspace(100.0, 220.0, 3000)
    down = np.linspace(220.0, 120.0, 3000)
    base = np.concatenate([up, down])
    for i, s in enumerate(TREND_SYMS):
        _frame(base * (1.0 + 0.02 * i)).to_feather(d / f"{s}_USDT-1h-perp.feather")
    return tmp_path / "data"


class TestClosedOnly:
    def test_drops_in_progress_bar(self):
        now = pd.Timestamp("2026-09-14 10:30", tz="UTC")
        df = _frame([1.0, 2.0, 3.0], start="2026-09-14 08:00")
        out = PerpDataFeed.closed_only(df, now=now)
        # bars at 08:00 and 09:00 are closed by 10:30; the 10:00 bar is not
        assert len(out) == 2
        assert out["date"].max() == pd.Timestamp("2026-09-14 09:00", tz="UTC")

    def test_empty_frame(self):
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        assert PerpDataFeed.closed_only(df).empty


class TestLiveCycle:
    def test_first_cycle_derives_and_opens_target(self, data_dir):
        cfg = RouterBookConfig(equity=50_000.0)
        bot = RouterBookLive(cfg, data_dir, feed=_FakeFeed())
        # skip the HMM refit: inject pre-fitted fake routers
        n_4h = 6000 // 4
        bot._routers = {s: _fake_router(n_4h) for s in TREND_SYMS}
        bot._last_router_refit = pd.Timestamp.now(tz="UTC")
        rep = bot.cycle()
        assert rep.new_4h_bar is True
        assert rep.router_refit is False
        # a strong synthetic trend must produce nonzero target positions,
        # opened through the chain and verified by reconciliation
        assert rep.target, "expected nonzero target from a trending panel"
        assert rep.orders_filled > 0
        assert rep.reconciliation_ok is True
        assert rep.kill_switch is False
        # broker now holds the target
        for sym, units in rep.target.items():
            assert bot.broker.positions().get(sym, 0.0) == pytest.approx(units, rel=1e-6)

    def test_second_cycle_without_new_bar_is_noop(self, data_dir):
        cfg = RouterBookConfig(equity=50_000.0)
        bot = RouterBookLive(cfg, data_dir, feed=_FakeFeed())
        n_4h = 6000 // 4
        bot._routers = {s: _fake_router(n_4h) for s in TREND_SYMS}
        bot._last_router_refit = pd.Timestamp.now(tz="UTC")
        bot.cycle()
        rep2 = bot.cycle()
        assert rep2.new_4h_bar is False
        assert rep2.note == "no new 4h bar"
        assert rep2.orders_filled == 0

    def test_diff_orders_closes_removed_positions(self, data_dir):
        cfg = RouterBookConfig(equity=50_000.0)
        bot = RouterBookLive(cfg, data_dir, feed=_FakeFeed())
        # open something, then diff against an empty target -> must close
        mark = 100.0
        bot.broker.set_market(mark)
        from research_lib.risk.engine import OrderIntent

        res = bot.execution.execute(
            OrderIntent(
                strategy_id="router_book_live", symbol="BTC/USDT:PERP",
                signal_time="t", side=1, units=10.0, entry_ref_price=mark,
                stop_price=None, notional=1000.0, post_only=False,
            )
        )
        assert res.submitted
        deltas, filled, failed = bot._diff_orders({}, "t2")
        assert deltas.get("BTC/USDT:PERP") == pytest.approx(-10.0)
        assert filled == 1 and failed == 0
        assert bot.broker.positions() == {}
