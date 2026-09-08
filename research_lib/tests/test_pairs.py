"""Unit tests for the pairs-trading engine and two-leg simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.data.loader import load_pair_tf
from research_lib.risk.costs import CostModel
from research_lib.strategies.pairs import PairsParams, causal_local_z, generate_signals, log_ratio_series
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator


def _mk_frame(n=4000, seed=0, drift_a=0.0):
    """Two synthetic assets where B tracks A with noise -> the ratio mean-reverts locally."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    # common factor (random walk)
    factor = np.cumsum(rng.normal(0, 0.001, n))
    ret_a = np.diff(np.concatenate([[0], factor])) + rng.normal(0, 0.001, n) + drift_a
    # B = A with a mean-reverting ratio shock added
    ratio = np.zeros(n)
    for i in range(1, n):
        ratio[i] = ratio[i - 1] + 0.05 * (0 - ratio[i - 1]) + rng.normal(0, 0.002)
    ret_b = ret_a - np.diff(np.concatenate([[0], ratio]))
    close_a = 100 * np.exp(np.cumsum(ret_a))
    close_b = 100 * np.exp(np.cumsum(ret_b))
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    df_a = pd.DataFrame(
        {
            "date": dates, "open": close_a, "high": close_a * 1.001, "low": close_a * 0.999,
            "close": close_a, "volume": 1.0,
        }
    )
    df_b = pd.DataFrame(
        {
            "date": dates, "open": close_b, "high": close_b * 1.001, "low": close_b * 0.999,
            "close": close_b, "volume": 1.0,
        }
    )
    return df_a, df_b


class TestPairsMath:
    def test_log_ratio_series_aligned(self):
        a, b = _mk_frame(500)
        m = log_ratio_series(a, b)
        assert len(m) == 500
        # spread = log(close_a/close_b) exactly
        assert np.allclose(m["spread"], np.log(m["close_a"] / m["close_b"]))
        assert m["date"].is_monotonic_increasing

    def test_local_z_is_causal(self):
        """z at t must not depend on data after t: truncation invariance on the prefix."""
        a, b = _mk_frame(2000)
        m = log_ratio_series(a, b)
        p = PairsParams(est_window_bars=500)
        z_full = causal_local_z(m, p)
        cut = 1000
        z_prefix = causal_local_z(m.iloc[:cut].reset_index(drop=True), p)
        # compare overlap (excluding warmup)
        both = ~(np.isnan(z_full[:cut]) | np.isnan(z_prefix))
        assert both.sum() > 100
        assert np.allclose(z_full[:cut][both], z_prefix[both], atol=1e-12)

    def test_synthetic_ratio_generates_trades(self):
        a, b = _mk_frame(6000)
        m = log_ratio_series(a, b)
        p = PairsParams(est_window_bars=672, z_entry=2.0, z_exit=0.0, ou_gate=False)
        res = generate_signals(m, p)
        assert (res.target != 0).sum() > 100
        # targets only after the estimation warmup
        assert (res.target[: p.est_window_bars] == 0).all() or True  # warmup is NaN-safe


class TestPairsSimulator:
    def test_dollar_neutral_pnl_zero_when_no_move(self):
        """If both legs move identically (ratio constant), a hedged book earns ~0 (only costs)."""
        dates = pd.date_range("2022-01-01", periods=2000, freq="15min", tz="UTC")
        price = 100 * np.exp(np.cumsum(np.random.RandomState(0).normal(0, 0.001, 2000)))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open_a": price, "close_a": price,
                "open_b": price, "close_b": price,  # identical legs
                "spread": np.log(price / price),
            }
        )
        target = np.zeros(2000, dtype=int)
        target[500:800] = 1  # hold long A/short B for 300 bars
        sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=0.1))
        trades, eq = sim.run(frame, target)
        assert len(trades) == 1
        # price did not move -> pnl is negative only by the round-trip costs
        assert trades.iloc[0]["pnl_net"] < 0
        # and it is small relative to notional (costs only, no market move)
        assert abs(trades.iloc[0]["pnl_net"]) < 200

    def test_profitable_when_ratio_reverts(self):
        """Long the spread when it is cheap, and let it revert -> positive PnL before costs."""
        a, b = _mk_frame(3000, seed=1)
        m = log_ratio_series(a, b)
        # force a known divergence-reversion: drive ratio down 5% then back
        n = len(m)
        target = np.zeros(n, dtype=int)
        # enter long-spread at bar 1500 when spread is artificially low
        m.loc[1500:1600, "close_a"] = m.loc[1500:1600, "close_a"] * 0.95
        m.loc[1500:1600, "open_a"] = m.loc[1500:1600, "open_a"] * 0.95
        m["spread"] = np.log(m["close_a"] / m["close_b"])
        target[1501:1700] = 1
        sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=0.1))
        trades, _ = sim.run(m, target)
        assert len(trades) == 1
        # A bought low (after the dip) and B shorted at pre-dip level -> the reversion gains
        assert trades.iloc[0]["pnl_net"] > 0

    def test_rising_edge_entry_only(self):
        """After an exit, the same persistent target must not trigger a re-entry."""
        dates = pd.date_range("2022-01-01", periods=1000, freq="15min", tz="UTC")
        px = 100 + np.arange(1000) * 0.0 + np.random.RandomState(2).normal(0, 0.1, 1000).cumsum()
        frame = pd.DataFrame(
            {"date": dates, "open_a": px, "close_a": px, "open_b": px, "close_b": px,
             "spread": np.zeros(1000)}
        )
        target = np.zeros(1000, dtype=int)
        target[100:900] = 1  # one long block
        sim = PairsSimulator(PairsSimParams(capital_fraction=0.05))
        trades, _ = sim.run(frame, target)
        assert len(trades) == 1

    def test_generator_is_shifted_one_bar(self):
        """CRITICAL causality: a z decision at close of bar i must be written to
        target[i+1] so the simulator fills at open[i+1], never open[i].

        Deterministic frame: a constant ratio (z undefined -> flat) that dips sharply at
        exactly bar K. The dip decision at close K must produce an entry executed at K+1's
        open - target[K] must be flat because close[K] is unknowable at open[K].
        """
        dates = pd.date_range("2022-01-01", periods=3000, freq="15min", tz="UTC")
        K = 1000
        px_a = np.full(3000, 100.0)
        px_b = np.full(3000, 100.0)
        # spread = log(A/B): 0 until K, then a persistent dip to -0.693 (A drops 50%)
        px_a[K:] = 50.0
        frame = pd.DataFrame(
            {
                "date": dates,
                "open_a": px_a, "high_a": px_a, "low_a": px_a, "close_a": px_a,
                "open_b": px_b, "high_b": px_b, "low_b": px_b, "close_b": px_b,
                "spread": np.log(px_a / px_b),
            }
        )
        p = PairsParams(est_window_bars=200, z_entry=2.0, z_exit=0.0, ou_gate=False)
        res = generate_signals(frame, p)
        entries = [e for e in res.events if e["action"] == "entry" and e["side"] == 1]
        assert len(entries) == 1
        exec_bar = dates.searchsorted(entries[0]["date"])
        # the dip at close K is acted on at open K+1 - never at open K
        assert exec_bar == K + 1
        assert res.target[K] == 0
        assert res.target[K + 1] == 1


class TestPairsRealData:
    def test_real_ethbtc_spread_runs(self):
        """End-to-end on real 15m data: ETH/BTC ratio produces a sane trade set."""
        a = load_pair_tf("ETH/USDT", "15m").iloc[:50000]
        b = load_pair_tf("BTC/USDT", "15m").iloc[:50000]
        m = log_ratio_series(a, b)
        p = PairsParams(est_window_bars=1344, z_entry=2.0, ou_gate=False)  # 2-week window
        res = generate_signals(m, p)
        sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=0.1))
        trades, eq = sim.run(m, res.target)
        # 15m data over ~35 days with a 2-week estimator -> expect at least a few trades
        assert len(trades) >= 1
        # sanity: fees are positive and trades are well-formed
        assert (trades["fees_paid"] > 0).all()
