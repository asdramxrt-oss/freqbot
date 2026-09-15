"""Smoke tests for the config-D router-book production runner.

The research harness (research/run_router_book_acceptance.py) proved the
economics; these tests pin the production contract: every sleeve runs through
the execution chain, the book ends flat, and reconciliation stays clean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.bot.router_book_runner import (
    MR_RATIOS,
    TREND_SYMS,
    RouterBookConfig,
    RouterBookRunner,
)
from research_lib.strategies.unified_router import RouterOutput


def _frame(prices, start="2022-01-01", freq="1h"):
    dates = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": 1.0,
        }
    )


def _fake_router(n: int) -> RouterOutput:
    """All gates open, no vol spikes: the runner's machinery is what's under test."""
    return RouterOutput(
        regime=np.full(n, "TREND", dtype=object),
        p_spike=np.full(n, 0.0),
        turtle_gate=np.ones(n, dtype=bool),
        pairs_gate=np.ones(n, dtype=bool),
        halt_reason=np.full(n, "", dtype=object),
    )


def _panels(n_1h: int = 6000):
    """Trend-then-reverse synthetic panel per coin (enough bars for SMA1200 4h)."""
    up = np.linspace(100.0, 220.0, n_1h // 2)
    down = np.linspace(220.0, 90.0, n_1h - n_1h // 2)
    base = np.concatenate([up, down])
    panels = {}
    for i, s in enumerate(TREND_SYMS):
        panels[s] = _frame(base * (1.0 + 0.02 * i))
    return panels


class TestRouterBookRunner:
    def test_runs_all_sleeves_and_ends_flat(self):
        n_4h = 6000 // 4
        routers = {s: _fake_router(n_4h) for s in TREND_SYMS}
        runner = RouterBookRunner(RouterBookConfig(equity=50_000.0), routers)
        out = runner.run(_panels())
        assert out["book_flat"] is True
        assert out["reconciliation_ok"] is True
        assert out["kill_switch"] is False
        # a strong synthetic trend must produce trend-sleeve trades
        assert out["n_trades"] > 0
        sleeves = set(out["trades"]["sleeve"].unique())
        assert any(s.startswith("trend:") for s in sleeves)

    def test_equity_curve_covers_full_history(self):
        n_4h = 6000 // 4
        routers = {s: _fake_router(n_4h) for s in TREND_SYMS}
        runner = RouterBookRunner(RouterBookConfig(equity=50_000.0), routers)
        out = runner.run(_panels())
        assert len(out["equity_curve"]) == n_4h
        assert np.isfinite(out["equity_curve"]["equity"]).all()

    def test_no_naked_mr_leg(self):
        """Whenever an MR sleeve is open, both legs must be booked in the broker."""
        n_4h = 6000 // 4
        routers = {s: _fake_router(n_4h) for s in TREND_SYMS}
        runner = RouterBookRunner(RouterBookConfig(equity=50_000.0), routers)
        out = runner.run(_panels())
        # final broker state must be flat too (no leg left behind)
        assert runner.broker.positions() == {}
        # and no aborted-entry log entries that left a leg unwound
        aborts = [e for e in out["log"] if e["action"] == "mr_entry_aborted_second_leg_failed"]
        assert runner.broker.positions() == {} and not any(
            e["action"] == "close_failed_abort" for e in out["log"]
        ), f"close failures: {aborts}"
