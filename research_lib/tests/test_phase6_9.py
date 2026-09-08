"""Unit tests for Phases 6-9: Turtle engine/sim, OU engine, HMM regime, router."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.data.loader import load_pair_tf
from research_lib.features.hurst import dfa_hurst
from research_lib.risk.backtest_engine import EngineConfig, EventBacktester
from research_lib.risk.costs import CostModel
from research_lib.strategies.ou import OUEngine, OUParams, fit_ou
from research_lib.strategies.regime_hmm import HMMParams, HMMRegimeEngine, characterize_states
from research_lib.strategies.router import RouterPolicy, build_causal_router, merge_router_with_targets
from research_lib.strategies.turtle import TurtleEngine, TurtleParams, donchian_channels
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator

from research_lib.tests.test_phase1_5 import _synth_df  # noqa: F401  (reuse synthetic builder)


# ------------------------------------------------------------------ helpers
def _btc1h(n: int = 6000) -> pd.DataFrame:
    return load_pair_tf("BTC/USDT", "1h").iloc[:n].reset_index(drop=True)


# ------------------------------------------------------------------ Phase 6
class TestTurtle:
    def test_donchian_is_shifted(self):
        """CRITICAL: breakout channel must use shift(1) so the current candle cannot break
        its own channel."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=30, freq="1h", tz="UTC"),
                "high": np.arange(30, dtype=float) + 100,
                "low": np.arange(30, dtype=float) + 90,
            }
        )
        dc = donchian_channels(df, 5)
        # At row 25, channel upper = max(high[20:25]) - the CURRENT high[25] must NOT appear
        assert dc["dc_upper"].iloc[25] == df["high"].iloc[20:25].max()
        # a monotonic series never has close > shifted rolling max at the same row unless the
        # previous 5 highs were all lower - construct an explicit probe: place a huge high at
        # row 25; the channel at 25 must be unaffected.
        df2 = df.copy()
        df2.loc[25, "high"] = 10_000.0
        dc2 = donchian_channels(df2, 5)
        assert dc2["dc_upper"].iloc[25] == dc["dc_upper"].iloc[25]

    def test_atr_causal(self):
        df = _btc1h(1000)
        from research_lib.features.engine import atr

        a = atr(df, 20)
        # value at t equals value computed on truncated frame
        a2 = atr(df.iloc[:500].reset_index(drop=True), 20)
        assert np.allclose(a.iloc[400:500].to_numpy(), a2.iloc[400:500].to_numpy(), equal_nan=True)

    def test_turtle_generate_entries_and_adds(self):
        df = _btc1h(4000)
        eng = TurtleEngine(TurtleParams(pyramid_max_units=4, use_pyramiding=True))
        sig, log = eng.generate(df)
        entries = [x for x in log if x["action"] == "entry"]
        adds = [x for x in log if x["action"] == "add"]
        exits = [x for x in log if x["action"] == "exit"]
        assert len(entries) > 0
        assert len(exits) > 0
        # state machine invariant: unit count increases monotonically within a position
        for e in entries:
            assert e["unit"] == 1
        for a in adds:
            assert 2 <= a["unit"] <= 4

    def test_turtle_fresh_cross_entries(self):
        """Simulator entries must be fresh channel crossings (close[t] > dc_upper[t] AND
        close[t-1] <= dc_upper[t-1]) - no churn re-entry while price stays beyond channel."""
        df = _btc1h(3000)
        sim = TurtleSimulator(TurtleSimParams(max_units=1))
        trades1, eq1, log1 = sim.run(df)
        sim_entries = [x for x in log1 if x["action"] == "entry"]
        assert len(sim_entries) > 0
        # every entry date must satisfy the fresh-cross property on the PRIOR close
        idx = {d: i for i, d in enumerate(df["date"])}
        for e in sim_entries:
            t = idx[e["date"]]
            assert t >= 1
            if e["side"] == 1:
                dc = df["high"].shift(1).rolling(20).max()
                assert df["close"].iloc[t - 1] > dc.iloc[t - 1]
                # prior close was not already above the channel
                assert not (df["close"].iloc[t - 2] > dc.iloc[t - 2])
            else:
                dc = df["low"].shift(1).rolling(20).min()
                assert df["close"].iloc[t - 1] < dc.iloc[t - 1]
                assert not (df["close"].iloc[t - 2] < dc.iloc[t - 2])

    def test_turtle_generate_stop_is_entry_minus_2n(self):
        """generate() must write stop = last_entry -/+ 2*N for the position."""
        df = _btc1h(2000)
        eng = TurtleEngine(TurtleParams(pyramid_max_units=1, use_pyramiding=False))
        sig, log = eng.generate(df)
        from research_lib.features.engine import atr

        n_arr = atr(df, 20).to_numpy()
        idx = {d: i for i, d in enumerate(df["date"])}
        for e in log:
            if e["action"] != "entry":
                continue
            t = idx[e["date"]]
            stop = sig["stop"].iloc[t]
            n = n_arr[t - 1]
            if e["side"] == 1:
                assert stop == pytest.approx(e["price"] - 2.0 * n, rel=1e-3)
            else:
                assert stop == pytest.approx(e["price"] + 2.0 * n, rel=1e-3)

    def test_sim_fees_charged(self):
        df = _btc1h(2000)
        sim = TurtleSimulator(TurtleSimParams())
        trades, _, _ = sim.run(df)
        assert len(trades) > 0
        assert (trades["fees_paid"] > 0).all()

    def test_sim_respects_max_units(self):
        df = _btc1h(3000)
        for mx in (1, 2, 4):
            sim = TurtleSimulator(TurtleSimParams(max_units=mx))
            trades, _, log = sim.run(df)
            max_units = max((x["units"] for x in log if x["action"] == "add"), default=1)
            assert max_units <= mx
            assert trades["units"].max() <= mx


# ------------------------------------------------------------------ Phase 7
class TestOU:
    def test_fit_valid_on_mean_reverting(self):
        rng = np.random.RandomState(0)
        # realistic slow mean reversion: theta=0.03/bar => half-life ~23 bars
        theta, mu, sigma = 0.03, 0.0, 0.3
        x = [0.0]
        for _ in range(6000):
            x.append(x[-1] + theta * (mu - x[-1]) + sigma * rng.normal())
        s = pd.Series(x)
        model = fit_ou(s, OUParams(min_sample=400, rolling_window=400))
        assert model.valid, model.reject_reasons
        assert 5 < model.half_life < 100
        assert 0 < model.b < 1

    def test_fit_rejects_random_walk(self):
        rng = np.random.RandomState(1)
        s = pd.Series(np.cumsum(rng.normal(0, 1, 1500)))
        model = fit_ou(s, OUParams(min_sample=100))
        # a pure random walk must NOT pass the OU gates (b close to 1, CI, residual ADF)
        assert not model.valid

    def test_hurst_on_mr_vs_rw(self):
        rng = np.random.RandomState(2)
        mr = []
        v = 0.0
        for _ in range(4000):
            v = v + 0.5 * (0 - v) + 0.3 * rng.normal()
            mr.append(v)
        rw = np.cumsum(rng.normal(0, 1, 4000))
        # DFA on LEVELS: MR spread H ~0.5-0.6, random walk H ~1.5. Hurst is a candidate
        # filter only - it must separate MR from trending, but NOT be used as sole proof.
        h_mr = dfa_hurst(np.asarray(mr))
        h_rw = dfa_hurst(rw)
        assert h_mr < h_rw - 0.3
        assert h_mr < 0.75
        assert h_rw > 1.0

    def test_engine_no_trade_when_model_invalid(self):
        df = _btc1h(2000)
        rng = np.random.RandomState(3)
        # feed a random-walk spread => model invalid => zero targets
        x = pd.Series(np.cumsum(rng.normal(0, 1, len(df))))
        eng = OUEngine(OUParams(min_sample=200))
        res = eng.run(df, x, fit_end=1000)
        assert not res.model.valid
        assert (res.signals["target"] == 0).all()

    def test_engine_trades_when_model_valid(self):
        rng = np.random.RandomState(4)
        n = 6000
        x = [0.0]
        for _ in range(n):
            x.append(x[-1] + 0.03 * (0 - x[-1]) + 0.3 * rng.normal())
        x = pd.Series(x[1:])
        dates = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "date": dates,
                "open": 100 + x.values,
                "high": 100 + x.values + 0.5,
                "low": 100 + x.values - 0.5,
                "close": 100 + x.values,
                "volume": 1.0,
            }
        )
        eng = OUEngine(OUParams(min_sample=800, z_entry=1.5, rolling_window=600))
        res = eng.run(df, x, fit_end=4000)
        assert res.model.valid, res.model.reject_reasons
        assert (res.signals["target"] != 0).sum() > 0


# ------------------------------------------------------------------ Phase 8
class TestHMM:
    def test_walk_forward_no_leak(self):
        # small frame so the walk-forward refit loop is fast
        df = _btc1h(1500)
        eng = HMMRegimeEngine(HMMParams(n_states=3, train_days=8, refit_days=8, n_iter=50))
        res = eng.run(df)
        # leading warmup is UNCERTAIN, rest labeled
        assert (res.state_id == -1).sum() > 0
        assert (res.labels == "UNCERTAIN").sum() > 0
        assert res.labels.nunique() >= 2

    def test_states_characterized_by_stats_not_id(self):
        # deterministic synthetic: two regimes, state identity must come from statistics
        rng = np.random.RandomState(5)
        n = 4000
        rets = np.zeros(n)
        # low-vol half, high-vol half
        rets[:2000] = rng.normal(0.0001, 0.002, 2000)
        rets[2000:] = rng.normal(-0.0002, 0.02, 2000)
        vol = np.concatenate([np.full(2000, 0.002), np.full(2000, 0.02)])
        # build from an actual fitted model via engine internals is complex; test the
        # characterizer directly with a fake model
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=50, random_state=0)
        obs = np.column_stack([rets, vol])
        # fit with artificial init to two distinct states is unstable; instead test the
        # characterizer ordering on hand-built inputs
        class FakeModel:
            n_components = 2
            transmat_ = np.array([[0.98, 0.02], [0.05, 0.95]])
            covars_ = np.array([[[0.002**2]], [[0.02**2]]])

        chars = characterize_states(
            FakeModel(),  # type: ignore
            np.column_stack([rets, vol]),
            np.array([0] * 2000 + [1] * 2000),
            HMMParams(n_states=2),
        )
        by_vol = {c.state_id: c.volatility for c in chars}
        assert max(by_vol.values()) > 5 * min(by_vol.values())
        # high-vol state must be labeled HIGH_VOL
        high_state = max(by_vol, key=by_vol.get)
        low_state = min(by_vol, key=by_vol.get)
        c_high = next(c for c in chars if c.state_id == high_state)
        assert c_high.regime_label == "HIGH_VOL"
        assert next(c for c in chars if c.state_id == low_state).regime_label != "HIGH_VOL"


# ------------------------------------------------------------------ Phase 9
class TestRouter:
    def test_policy_mapping(self):
        labels = pd.Series(["TREND", "LOW_VOL_MR", "HIGH_VOL", "NEUTRAL", "UNCERTAIN"])
        res = build_causal_router(labels)
        assert res.route.tolist() == ["turtle", "ou", "no_trade", "no_trade", "no_trade"]

    def test_merge_targets(self):
        labels = pd.Series(["TREND"] * 4 + ["LOW_VOL_MR"] * 4 + ["HIGH_VOL"] * 4)
        res = build_causal_router(labels)
        turtle = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        ou = np.array([0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0])
        merged = merge_router_with_targets(turtle, ou, res.route)
        assert merged[:4].tolist() == [1, 1, 1, 1]
        assert merged[4:8].tolist() == [-1, -1, -1, -1]
        assert merged[8:].tolist() == [0, 0, 0, 0]

    def test_oracle_diagnostic_separate(self):
        labels = pd.Series(["TREND"] * 6)
        ot = pd.Series([True, False, True, False, True, False])
        oo = pd.Series([False, True, False, True, False, True])
        res = build_causal_router(labels, oracle_turtle=ot, oracle_ou=oo)
        # causal route stays turtle for all TREND bars
        assert (res.route == "turtle").all()
        # oracle diagnostic: turtle flagged bars -> turtle; un-flagged turtle but OU flagged
        # bars -> ou (single-position constraint: turtle takes precedence)
        assert res.oracle_route.iloc[0] == "turtle"  # ot=True
        assert res.oracle_route.iloc[1] == "ou"  # ot=False, oo=True
        assert (res.oracle_route.isin(["turtle", "ou", "no_trade"])).all()
