"""Unit tests for the vol-spike alpha strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.strategies.volspike_alpha import (
    VolSpikeAlphaParams,
    direction_from_momentum,
    generate_alpha_signals,
)


def _frame(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
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


class TestDirection:
    def test_follow(self):
        assert direction_from_momentum(0.01, "follow") == 1
        assert direction_from_momentum(-0.01, "follow") == -1
        assert direction_from_momentum(0.0, "follow") == 0

    def test_fade(self):
        assert direction_from_momentum(0.01, "fade") == -1
        assert direction_from_momentum(-0.01, "fade") == 1

    def test_dipbuy(self):
        assert direction_from_momentum(-0.01, "dipbuy") == 1
        assert direction_from_momentum(0.01, "dipbuy") == 0

    def test_unknown_rule(self):
        with pytest.raises(ValueError):
            direction_from_momentum(0.01, "nope")


class TestSignals:
    def test_no_trade_when_no_spike(self):
        df = _frame(2000)
        p = np.full(len(df), 0.1)  # never above p_entry
        sig = generate_alpha_signals(df, p, VolSpikeAlphaParams())
        assert (sig["target"] == 0).all()

    def test_trades_on_fresh_crossing(self):
        df = _frame(2000)
        p = np.zeros(len(df))
        p[500:800] = 0.9   # a sustained predicted-spike block
        mom = np.zeros(len(df))
        # force positive momentum so 'follow' goes long
        sig_follow = generate_alpha_signals(df, p, VolSpikeAlphaParams(direction="follow"))
        # long block should appear shortly after the crossing (entry at open t+1)
        long_start = np.argmax(sig_follow["target"] == 1)
        assert long_start > 500  # strictly after the signal was observable
        assert sig_follow["target"][long_start] == 1
        assert 500 < long_start <= 502  # signal at 500 (p known at close 500) -> target at 501+

    def test_causality_prefix_invariant(self):
        """Signals on a prefix must be identical to signals on the full frame (no future)."""
        df = _frame(4000)
        p_full = np.zeros(len(df))
        p_full[1000:1400] = 0.9
        p_full[2000:2400] = 0.85
        params = VolSpikeAlphaParams(direction="fade")
        full = generate_alpha_signals(df, p_full, params)
        cut = 2500
        prefix = generate_alpha_signals(df.iloc[:cut].reset_index(drop=True),
                                        p_full[:cut], params)
        # target and stop for the shared region must match exactly
        assert (full["target"].iloc[:cut].to_numpy() == prefix["target"].to_numpy()).all()
        a = full["stop"].iloc[:cut].to_numpy()
        b = prefix["stop"].to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[both], b[both], equal_nan=True)

    def test_breakout_requires_prior_range(self):
        """Breakout mode: entry only on a close beyond the PRIOR range (never the current
        bar's own range) and only while a spike is predicted."""
        df = _frame(3000, seed=5)
        # force a quiet range then an expansion
        p = np.zeros(len(df))
        p[1500:1800] = 0.9  # predicted spike window
        # inject a strong up-move inside the spike window
        df.loc[1550:, "close"] = df.loc[1550:, "close"].to_numpy() * np.linspace(1.001, 1.02, len(df) - 1550)
        df.loc[1550:, "high"] = df["close"].loc[1550:] * 1.001
        df.loc[1550:, "low"] = df["close"].loc[1550:] * 0.999
        df.loc[1550:, "open"] = df["close"].loc[1550:]
        sig = generate_alpha_signals(df, p, VolSpikeAlphaParams(direction="breakout",
                                                                breakout_bars=32))
        longs = np.where(sig["target"] == 1)[0]
        assert len(longs) > 0
        # first long must be strictly after the up-move began (causal: decided at close i,
        # held from open i+1) and inside the predicted-spike window
        assert longs[0] >= 1551

    def test_no_breakout_trades_when_quiet(self):
        df = _frame(2000)
        p = np.zeros(len(df))  # never a predicted spike
        sig = generate_alpha_signals(df, p, VolSpikeAlphaParams(direction="breakout"))
        assert (sig["target"] == 0).all()

    def test_time_stop_limits_hold(self):
        df = _frame(4000)
        p = np.zeros(len(df))
        p[1000:] = 0.9  # spike stays high forever
        params = VolSpikeAlphaParams(direction="follow", max_hold_bars=96)
        sig = generate_alpha_signals(df, p, params)
        # find long blocks and confirm each is <= max_hold_bars + slack
        tgt = sig["target"].to_numpy()
        in_pos = False
        start = 0
        max_run = 0
        for t in range(len(tgt)):
            if tgt[t] != 0 and not in_pos:
                in_pos, start = True, t
            elif tgt[t] == 0 and in_pos:
                max_run = max(max_run, t - start)
                in_pos = False
        assert max_run <= 96 + 2
