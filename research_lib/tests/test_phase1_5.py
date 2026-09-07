"""Unit tests for Phases 1-5: data layer, bars, volatility, fracdiff, features/audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.data.infobars import BarSpec, build_bars
from research_lib.data.loader import load_pair_tf
from research_lib.data.quality import DataQualityValidator
from research_lib.features.audit import dynamic_causality_check, static_scan_python
from research_lib.features.engine import FeatureEngine
from research_lib.features.fracdiff import (
    FracdiffTransform,
    fit_fracdiff_transform,
    fracdiff_series,
    fracdiff_weights,
)
from research_lib.features.volatility import VolatilityEstimator, VolatilityEngine


def _synth_df(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rets = rng.normal(0.001, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    dates = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": rng.lognormal(10, 0.5, n),
        }
    )


# ------------------------------------------------------------------ Phase 1
class TestDataQuality:
    def test_clean_frame_passes(self):
        df = _synth_df(500)
        r = DataQualityValidator("t", "1h").validate(df)
        assert r.passed

    def test_duplicate_timestamp_hard(self):
        df = _synth_df(200)
        df = pd.concat([df, df.iloc[[50]]], ignore_index=True).sort_values("date")
        r = DataQualityValidator("t", "1h").validate(df)
        assert any(i.check == "duplicate_timestamp" and i.severity == "hard" for i in r.issues)

    def test_impossible_ohlc_hard(self):
        df = _synth_df(200)
        df.loc[10, "high"] = df.loc[10, "low"] - 1
        r = DataQualityValidator("t", "1h").validate(df)
        assert any(i.check == "impossible_ohlc" and i.severity == "hard" for i in r.issues)

    def test_missing_timestamp_detected(self):
        df = _synth_df(300)
        df = df.drop(df.index[100]).reset_index(drop=True)
        r = DataQualityValidator("t", "1h", max_pct_missing=0.0).validate(df)
        assert any(i.check == "missing_timestamps" for i in r.issues)

    def test_zero_price_hard(self):
        df = _synth_df(200)
        df.loc[10, "close"] = 0.0
        r = DataQualityValidator("t", "1h").validate(df)
        assert any(i.check == "nonpositive_price" and i.severity == "hard" for i in r.issues)


# ------------------------------------------------------------------ Phase 2
class TestInfoBars:
    def test_time_bars_passthrough(self):
        df = _synth_df(100)
        out = build_bars(df, BarSpec("time", 3600))
        assert len(out) == len(df)

    def test_volume_bars_causal_and_aggregate(self):
        df = _synth_df(500)
        thresh = df["volume"].iloc[:100].sum() / 10  # ~10 volume bars in first 100
        bars = build_bars(df, BarSpec("volume", thresh), symbol="X")
        assert len(bars) > 0
        # bar timestamps are within source range and non-decreasing
        assert bars["date"].is_monotonic_increasing
        assert bars["date"].iloc[-1] <= df["date"].iloc[-1]
        # OHLC sanity
        assert (bars["high"] >= bars["low"]).all()
        # cumulative volume close to total (last partial bar excluded)
        assert bars["volume"].sum() <= df["volume"].sum() * 1.0001
        # strictly causal: each bar's close is the close of a source candle at/before bar date
        src = df.set_index("date")["close"]
        for _, row in bars.head(5).iterrows():
            assert abs(row["close"] - src.loc[row["date"]]) < 1e-9

    def test_bad_threshold(self):
        with pytest.raises(ValueError):
            build_bars(_synth_df(100), BarSpec("volume", 0))


# ------------------------------------------------------------------ Phase 3
class TestVolatility:
    def test_estimators_finite_and_positive(self):
        df = _synth_df(1000)
        for m in ("close_to_close", "ewma", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"):
            s = VolatilityEstimator(method=m, window=20, session="cont").compute(df)
            valid = s.dropna()
            assert len(valid) > 500
            assert (valid > 0).all()
            assert np.isfinite(valid).all()

    def test_engine_outputs(self):
        df = _synth_df(1000)
        out = VolatilityEngine(method="yang_zhang", window=20).compute(df)
        for col in ("volatility", "volatility_percentile", "volatility_zscore", "volatility_regime"):
            assert col in out.columns
        assert out["volatility_regime"].dropna().isin(["LOW", "NORMAL", "HIGH", "EXTREME"]).all()

    def test_causality(self):
        df = _synth_df(500)
        full = VolatilityEstimator(method="yang_zhang", window=20).compute(df)
        cut = VolatilityEstimator(method="yang_zhang", window=20).compute(df.iloc[:300].reset_index(drop=True))
        # overlapping region must be identical
        a = full.iloc[:250].to_numpy()
        b = cut.iloc[:250].to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[both], b[both], equal_nan=True)


# ------------------------------------------------------------------ Phase 4
class TestFracdiff:
    def test_d0_identity(self):
        s = pd.Series(np.random.RandomState(1).normal(size=100))
        assert fracdiff_series(s, 0.0).equals(s)

    def test_d1_first_difference(self):
        s = pd.Series(np.cumsum(np.random.RandomState(1).normal(size=200)))
        fd = fracdiff_series(s, 1.0, window=200, min_periods=2)
        diff = s.diff()
        # d=1 with enough memory is the exact first difference
        assert np.allclose(fd.iloc[5:].to_numpy(), diff.iloc[5:].to_numpy(), atol=1e-9)

    def test_weights_recursion(self):
        w = fracdiff_weights(0.5, 10)
        # w_0 = 1; w_1 = -d; weights decay toward zero (finite memory)
        assert w[0] == 1.0
        assert w[1] == pytest.approx(-0.5)
        # analytic recurrence w_k = -w_{k-1}*(d-k+1)/k for k>=1
        for k in range(1, 10):
            expect = -fracdiff_weights(0.5, k + 1)[k - 1] * (0.5 - k + 1) / k
            assert w[k] == pytest.approx(expect)

    def test_fit_freeze_apply(self):
        rng = np.random.RandomState(3)
        # non-stationary train (random walk)
        s = pd.Series(np.cumsum(rng.normal(0, 1, 1500)))
        t = fit_fracdiff_transform(s, window=64)
        assert 0.0 <= t.d <= 1.0
        assert t.diagnostic is not None
        # frozen transform applied to a *later* disjoint series uses only the frozen d
        s2 = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        out = t.transform(s2)
        assert len(out) == len(s2)
        assert out.notna().sum() > len(s2) - 64

    def test_frozen_d_is_applied_not_refit(self):
        rng = np.random.RandomState(7)
        train = pd.Series(np.cumsum(rng.normal(0, 1, 600)))
        t = fit_fracdiff_transform(train, window=128)
        later = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
        # transforming the later series must equal fracdiff with the frozen d (identical)
        direct = fracdiff_series(later, t.d, window=128)
        via = t.transform(later)
        assert np.allclose(direct.to_numpy(), via.to_numpy(), equal_nan=True)


# ------------------------------------------------------------------ Phase 5
class TestFeatureEngine:
    def test_build_shapes_and_causality(self):
        df = _synth_df(1500)
        fe = FeatureEngine(hurst_window=300)
        res = dynamic_causality_check(fe.build, df.iloc[:1000].reset_index(drop=True), cutoff=600)
        assert res["passed"], res["issues"]

    def test_registry_entries(self):
        from research_lib.features.engine import FEATURE_REGISTRY

        assert "atr_20" in FEATURE_REGISTRY
        assert "rsi_14" in FEATURE_REGISTRY
        assert all(v["causal"] for v in FEATURE_REGISTRY.values())

    def test_real_data_load(self):
        df = load_pair_tf("BTC/USDT", "1h")
        assert len(df) > 10000
        assert df["date"].is_monotonic_increasing


class TestStaticAudit:
    def test_detects_real_leaks(self):
        src = (
            "def f(df):\n"
            "    x = df['close'].shift(-1)\n"
            "    y = df['close'].mean()\n"
            "    z = df['close'].rolling(20).mean()\n"
        )
        flags = static_scan_python(src)
        patterns = {f["pattern"] for f in flags}
        assert "shift_negative" in patterns
        assert "whole_frame_reduction" in patterns

    def test_clean_source_has_no_flags(self):
        src = (
            "def f(df):\n"
            "    x = df['close'].shift(1)\n"
            "    y = df['close'].rolling(20).mean()\n"
            "    z = df['close'].ewm(span=20).mean()\n"
        )
        assert static_scan_python(src) == []
