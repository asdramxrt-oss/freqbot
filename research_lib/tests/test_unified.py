"""Unit tests for the unified regime router + engine + vol-spike kill gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits
from research_lib.strategies.pairs import log_ratio_series
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator
from research_lib.strategies.unified_engine import UnifiedParams, run_unified
from research_lib.strategies.unified_router import (
    HMMParams,
    RouterConfig,
    RouterOutput,
    filtered_hmm_router,
)

from research_lib.tests.test_pairs import _mk_frame as _mk_pair_frame


def _garch_frame(n=8000, seed=0):
    """Stationary GARCH(1,1) series with regime-ish drift: trend up then down then range."""
    rng = np.random.RandomState(seed)
    burn = 2000
    total = burn + n
    rets = np.zeros(total)
    omega, alpha, beta = 1e-6, 0.1, 0.85
    sigma2 = np.full(total, omega / (1 - alpha - beta))
    # add a slow drift that changes sign to create trend + range phases
    t = np.arange(total)
    drift = 1e-5 * np.sin(t / 400.0) + 2e-5 * np.sin(t / 900.0)
    for i in range(1, total):
        sigma2[i] = omega + alpha * rets[i - 1] ** 2 + beta * sigma2[i - 1]
        rets[i] = drift[i] + rng.normal(0, np.sqrt(sigma2[i]))
    rets = rets[burn:]
    close = 100 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close, "high": close * 1.002, "low": close * 0.998,
            "close": close, "volume": 1.0,
        }
    )


class TestVolSpikeKillGate:
    def test_gate_blocks_when_spike_predicted(self):
        eng = CentralRiskEngine(RiskLimits())
        assert not eng.set_vol_spike_gate(0.5, threshold=0.8)
        assert eng.set_vol_spike_gate(0.9, threshold=0.8)
        assert eng.strategy_blocked("turtle")
        intent = OrderIntent(strategy_id="turtle", symbol="BTC/USDT",
                             signal_time="2022-01-01", side=1, units=1.0,
                             entry_ref_price=100.0, stop_price=99.0, notional=100.0)
        dec = eng.evaluate(intent)
        assert not dec.approved
        assert any("blocked" in r or "HALT" in r for r in dec.reasons)
        eng.clear_vol_spike_gate()
        assert not eng.strategy_blocked("turtle")

    def test_per_strategy_block(self):
        eng = CentralRiskEngine(RiskLimits())
        eng.set_strategy_block("pairs", True)
        assert eng.strategy_blocked("pairs")
        assert not eng.strategy_blocked("turtle")
        eng.set_strategy_block("pairs", False)
        assert not eng.strategy_blocked("pairs")


class TestSimulatorEntryGates:
    def test_turtle_gate_blocks_entries_and_closes_open(self):
        df = _garch_frame(3000)
        # gate on for a while, then off forever -> any open position must close
        gate = np.ones(len(df), dtype=bool)
        gate[1500:] = False
        sim = TurtleSimulator(TurtleSimParams(equity=100_000, allow_short=True))
        trades, eq, log = sim.run(df, entry_gate=gate)
        # no trade may exit after the gate turned off (bar 1500)
        if len(trades):
            last_exit = trades["exit_time"].max()
            last_exit_idx = df["date"].searchsorted(pd.Timestamp(last_exit))
            assert last_exit_idx <= 1501
        # and no entry may happen after 1500
        entries = [x for x in log if x["action"] == "entry"]
        for e in entries:
            idx = df["date"].searchsorted(pd.Timestamp(e["date"]))
            assert idx <= 1501

    def test_turtle_gate_off_from_start_no_trades(self):
        df = _garch_frame(2000)
        gate = np.zeros(len(df), dtype=bool)
        sim = TurtleSimulator(TurtleSimParams(equity=100_000))
        trades, _, _ = sim.run(df, entry_gate=gate)
        assert len(trades) == 0


class TestRouter:
    def test_router_runs_causal_on_realish_data(self):
        df = _garch_frame(6000, seed=1)
        cfg = RouterConfig(
            hmm=HMMParams(n_states=3, n_iter=60),
            refit_bars=600,
            min_train_bars=1200,
        )
        out = filtered_hmm_router(df, cfg)
        assert isinstance(out, RouterOutput)
        # first min_train bars are UNCERTAIN (no model yet)
        assert (out.regime[:1200] == "UNCERTAIN").all()
        # gates are off during the warm-up
        assert not out.turtle_gate[:1200].any()
        # after warmup some non-UNCERTAIN labels appear
        assert (out.regime[1200:] != "UNCERTAIN").sum() > 0

    def test_router_gate_is_causal_one_bar_shift(self):
        """gate[t] must equal a decision made from data <= t-1: forcing a known regime at
        bar K can only affect gate[K+1], never gate[K]."""
        df = _garch_frame(3000, seed=2)
        cfg = RouterConfig(hmm=HMMParams(n_states=3, n_iter=60),
                           refit_bars=500, min_train_bars=1200)
        out = filtered_hmm_router(df, cfg)
        n = len(df)
        # causality invariant: gate[t] is derived from regime[t-1]; changing regime at K
        # must not change gate[K]. We can't inject regimes externally here, but we can
        # verify the structural one-bar shift holds by checking gate never leads regime.
        for t in range(1201, n):
            # turtle_gate[t] True requires regime[t-1] in turtle_regimes
            if out.turtle_gate[t]:
                assert out.regime[t - 1] in cfg.turtle_regimes
            if out.pairs_gate[t]:
                assert out.regime[t - 1] in cfg.pairs_regimes


class TestUnifiedEngine:
    def test_run_end_to_end(self):
        df = _garch_frame(7000, seed=3)
        a, b = _mk_pair_frame(7000, seed=4)
        # align ratio frame to df calendar (build from the same dates)
        b2 = b.copy()
        b2["date"] = df["date"].to_numpy()
        ratio = log_ratio_series(a, b2)
        ratio["date"] = df["date"].to_numpy()
        # ratio frame from a/b has its own dates; force to df's calendar for the test
        up = UnifiedParams(
            router=RouterConfig(hmm=HMMParams(n_states=3, n_iter=50),
                                refit_bars=600, min_train_bars=1500),
        )
        res = run_unified(df, ratio, up, seed=0)
        assert res.equity_curve is not None
        assert len(res.equity_curve) == len(df)
        # equity curve should be finite after warmup
        assert np.isfinite(res.equity_curve[2000:]).all()
        # no gate overlap by construction (disjoint regime sets)
        assert res.log["gate_overlap_bars"] == 0
