"""Unit tests for blueprint Phase 2-3 additions: vol-spike filter, maker fills, Kelly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.ml.metalabel import fractional_kelly_size, kelly_fraction
from research_lib.ml.volspike import (
    VolSpikeParams,
    build_vol_spike_dataset,
    causal_vol_features,
    evaluate_vol_spike_filter,
)
from research_lib.risk.costs import CostModel
from research_lib.risk.maker import MakerFillModel, simulate_maker_entry

from research_lib.tests.test_phase1_5 import _synth_df


class TestKelly:
    def test_zero_edge_no_bet(self):
        assert kelly_fraction(0.5, 1.0) == 0.0
        assert kelly_fraction(0.4, 1.0) == 0.0
        assert fractional_kelly_size(0.5, 1.0) == 0.0

    def test_positive_edge(self):
        # p=0.6, b=1 -> f* = 0.6 - 0.4 = 0.2
        assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)
        # quarter Kelly
        assert fractional_kelly_size(0.6, 1.0, kelly_frac=0.25) == pytest.approx(0.05)

    def test_monotonic_in_p(self):
        a = fractional_kelly_size(0.55, 1.0)
        b = fractional_kelly_size(0.7, 1.0)
        assert b > a


class TestMakerFills:
    def test_maker_price_at_touch(self):
        m = MakerFillModel(cost=CostModel(spread_bps=10.0))  # 5bp half-spread
        # buy rests at bid = mid - half spread
        assert m.maker_price(100.0, "long") == pytest.approx(100.0 * (1 - 0.0005))
        # sell rests at ask = mid + half spread
        assert m.maker_price(100.0, "short") == pytest.approx(100.0 * (1 + 0.0005))

    def test_maker_fill_is_probabilistic(self):
        m = MakerFillModel(p_fill_per_bar=1.0)
        rng = np.random.default_rng(0)
        price, filled = m.expected_fill_price(100.0, "long", rng)
        assert filled and price < 100.0  # filled at the bid (better than taker)
        m2 = MakerFillModel(p_fill_per_bar=0.0)
        _, filled2 = m2.expected_fill_price(100.0, "long", rng)
        assert not filled2

    def test_maker_fee_below_taker(self):
        m = MakerFillModel()
        assert m.maker_fee_rate() < m.cost.taker_fee

    def test_simulate_entry_respects_path(self):
        m = MakerFillModel(p_fill_per_bar=1.0, max_wait_bars=5)
        rng = np.random.default_rng(1)
        path = np.full(20, 100.0)
        bar, price = simulate_maker_entry(m, path, start_bar=0, side="long", rng=rng)
        assert bar == 1  # fills next bar with p=1
        assert price < 100.0

    def test_unfilled_falls_back_to_taker(self):
        m = MakerFillModel(p_fill_per_bar=0.0, max_wait_bars=3)
        rng = np.random.default_rng(2)
        path = np.full(20, 100.0)
        bar, price = simulate_maker_entry(m, path, start_bar=0, side="long", rng=rng)
        # no fill for 3 bars -> taker fallback (worse than maker touch)
        assert price > 100.0 * (1 - 0.0005)  # adverse taker price


class TestVolSpike:
    @staticmethod
    def _garch_df(n=6000, seed=7):
        """Stationary GARCH(1,1): vol clustering, burn-in so train/test vol match."""
        rng = np.random.RandomState(seed)
        burn = 3000
        total = burn + n
        rets = np.zeros(total)
        omega, alpha, beta = 1e-6, 0.1, 0.85
        sigma2 = np.full(total, omega / (1 - alpha - beta))  # unconditional variance
        for t in range(1, total):
            sigma2[t] = omega + alpha * rets[t - 1] ** 2 + beta * sigma2[t - 1]
            rets[t] = rng.normal(0, np.sqrt(sigma2[t]))
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

    def test_label_is_future(self):
        df = self._garch_df()
        ds, y = build_vol_spike_dataset(df, VolSpikeParams(horizon_bars=48))
        # no label on the last horizon bars (no future data available)
        assert np.isnan(ds["future_vol"].iloc[-48:]).all()
        assert set(np.unique(y[np.isfinite(y)])) <= {0, 1}

    def test_causal_features_no_future(self):
        df = self._garch_df(2000)
        f = causal_vol_features(df, VolSpikeParams())
        f_full = f.copy()
        cut = 1000
        f_pre = causal_vol_features(df.iloc[:cut].reset_index(drop=True), VolSpikeParams())
        num_cols = [c for c in f.columns if c != "date"]
        for col in num_cols:
            a = f_full[col].iloc[:cut].to_numpy()
            b = f_pre[col].iloc[:cut].to_numpy()
            both = ~(np.isnan(a.astype(float)) | np.isnan(b.astype(float)))
            if both.sum() > 10:
                assert np.allclose(a[both].astype(float), b[both].astype(float), atol=1e-12), col

    def test_evaluate_runs(self):
        df = self._garch_df()
        res = evaluate_vol_spike_filter(df, np.zeros(len(df)), VolSpikeParams(horizon_bars=48))
        assert "auc" in res, res
        assert 0 <= res["auc"] <= 1 or np.isnan(res["auc"])
        assert res["n_train"] > 0 and res["n_test"] > 0
        # GARCH clustering is a real, learnable signal: AUC should exceed 0.5 materially
        assert res["auc"] > 0.55, res
