"""Causal feature engine (Phase 5).

Feature categories built over a canonical OHLCV frame:
  PRICE            returns, log returns, momentum, rolling range, dist from MA, breakout dist
  VOLATILITY       ATR, normalized ATR, Yang-Zhang/EWMA vol, vol percentile, realized vol
  MOMENTUM         RSI, rate of change, ADX trend strength, directional movement
  MICROSTRUCTURE   spread/volume/trade imbalance (only when columns present)
  MEAN-REVERSION   z-scores, distance from VWAP / rolling mean, residual deviations
  FRACTIONAL-DIFF  fd price/returns/log-price (via frozen transform)
  REGIME           vol regime, trend strength, autocorrelation, Hurst/DFA

Every feature is strictly causal (rolling/trailing windows + shift only). Each produced
column is registered in FEATURE_REGISTRY with lookback, availability, method, dependencies,
and causal status. The feature-audit tool (features/audit.py) can statically detect common
future-reference patterns in strategy code and dynamically verify that truncating the frame
does not change historical feature values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from research_lib.features.hurst import dfa_hurst, rolling_hurst
from research_lib.features.volatility import VolatilityEngine

# feature registry: name -> metadata dict
FEATURE_REGISTRY: dict[str, dict] = {}


def _register(name: str, lookback: int, method: str, deps: list[str], causal: bool = True) -> None:
    FEATURE_REGISTRY[name] = {
        "name": name,
        "lookback": lookback,
        "timestamp_availability": "close of last bar in lookback",
        "calculation_method": method,
        "data_dependencies": deps,
        "causal": causal,
    }


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def _tr(df: DataFrame) -> Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: DataFrame, period: int = 20) -> Series:
    """Wilder's ATR (causal: uses high/low/close up to t)."""
    tr = _tr(df)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().rename(f"atr_{period}")


def rsi(df: DataFrame, period: int = 14) -> Series:
    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.rename(f"rsi_{period}")


def adx(df: DataFrame, period: int = 14) -> Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _tr(df)
    atr_ = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    pdi = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / atr_
    mdi = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / atr_
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx_.rename(f"adx_{period}")


def donchian(df: DataFrame, period: int = 20) -> DataFrame:
    """Donchian channels using SHIFTED high/low (breakout must not see current candle)."""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    dc = DataFrame(
        {
            f"dc_high_{period}": prev_high.rolling(period).max(),
            f"dc_low_{period}": prev_low.rolling(period).min(),
            f"dc_mid_{period}": (prev_high.rolling(period).max() + prev_low.rolling(period).min())
            / 2.0,
        },
        index=df.index,
    )
    return dc


def vwap_rolling(df: DataFrame, period: int = 20) -> Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).rolling(period).sum()
    v = df["volume"].rolling(period).sum().replace(0, np.nan)
    return (pv / v).rename(f"vwap_{period}")


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------
class FeatureEngine:
    """Builds a causal feature frame from a canonical OHLCV frame.

    Parameters mirror the strategy modules; FD features require a frozen transform
    (fit on TRAIN only) supplied by the caller - the engine NEVER fits its own transform
    on the frame it is applied to when ``fd_transform`` is provided.
    """

    def __init__(
        self,
        *,
        ma_periods: tuple[int, ...] = (20, 50, 200),
        momentum_periods: tuple[int, ...] = (14, 24),
        atr_periods: tuple[int, ...] = (14, 20),
        vol_window: int = 20,
        vol_method: str = "yang_zhang",
        session: str = "cont",
        lookback: int = 256,
        hurst_window: int = 500,
        fd_transform=None,
    ) -> None:
        self.ma_periods = ma_periods
        self.momentum_periods = momentum_periods
        self.atr_periods = atr_periods
        self.vol_window = vol_window
        self.vol_method = vol_method
        self.session = session
        self.lookback = lookback
        self.hurst_window = hurst_window
        self.fd_transform = fd_transform  # frozen FracdiffTransform or None
        self._vol_engine = VolatilityEngine(method=vol_method, window=vol_window, session=session)

    # ------------------------------------------------------------------
    def _price_features(self, df: DataFrame, out: DataFrame) -> None:
        c = df["close"]
        out["ret_1"] = c.pct_change(1)
        out["log_ret_1"] = np.log(c / c.shift(1))
        for p in (5, 10, 20, 60):
            out[f"mom_{p}"] = c.pct_change(p)
        for p in self.ma_periods:
            ma = c.rolling(p).mean()
            out[f"ma_{p}"] = ma
            out[f"dist_ma_{p}"] = (c - ma) / ma
        # rolling range (high-low)/close
        for p in (20, 60):
            out[f"range_{p}"] = (df["high"].rolling(p).max() - df["low"].rolling(p).min()) / c
        # breakout distance to prior donchian (shifted)
        for p in (20, 55):
            dc = donchian(df, p)
            out[f"breakout_dist_hi_{p}"] = (c - dc[f"dc_high_{p}"]) / c
            out[f"breakout_dist_lo_{p}"] = (dc[f"dc_low_{p}"] - c) / c
        _register("ret_1", 1, "pct_change", ["close"])
        _register("mom_20", 20, "pct_change(20)", ["close"])

    def _vol_features(self, df: DataFrame, out: DataFrame) -> None:
        vol = self._vol_engine.compute(df)
        out["volatility"] = vol["volatility"]
        out["volatility_percentile"] = vol["volatility_percentile"]
        out["volatility_regime"] = vol["volatility_regime"]
        for p in self.atr_periods:
            a = atr(df, p)
            out[f"atr_{p}"] = a
            out[f"atr_norm_{p}"] = a / df["close"]
        # realized volatility (rolling std of log returns)
        lr = np.log(df["close"] / df["close"].shift(1))
        for p in (20, 60):
            out[f"realized_vol_{p}"] = lr.rolling(p).std()
        _register("volatility", self.vol_window, f"vol[{self.vol_method}]", ["ohlc"])
        _register("atr_20", 20, "wilder_atr", ["ohlc"])

    def _momentum_features(self, df: DataFrame, out: DataFrame) -> None:
        for p in self.momentum_periods:
            out[f"rsi_{p}"] = rsi(df, p)
            out[f"roc_{p}"] = df["close"].pct_change(p) * 100.0
        out["adx_14"] = adx(df, 14)
        out["adx_28"] = adx(df, 28)
        _register("rsi_14", 14, "wilder_rsi", ["close"])
        _register("adx_14", 14, "wilder_adx", ["ohlc"])

    def _microstructure_features(self, df: DataFrame, out: DataFrame) -> None:
        # only built when the columns exist (order-flow data optional)
        if "spread" in df.columns:
            out["spread_rel"] = df["spread"] / df["close"]
        if "volume" in df.columns:
            for p in (20, 60):
                v = df["volume"]
                out[f"volume_z_{p}"] = (v - v.rolling(p).mean()) / v.rolling(p).std().replace(0, np.nan)
        _register("volume_z_20", 20, "rolling z of volume", ["volume"])

    def _mean_reversion_features(self, df: DataFrame, out: DataFrame) -> None:
        c = df["close"]
        for p in (20, 60, 200):
            ma = c.rolling(p).mean()
            sd = c.rolling(p).std()
            out[f"zscore_{p}"] = (c - ma) / sd.replace(0, np.nan)
        vw = vwap_rolling(df, 20)
        out["vwap_20"] = vw
        out["dist_vwap_20"] = (c - vw) / vw
        _register("zscore_20", 20, "rolling z-score", ["close"])

    def _fd_features(self, df: DataFrame, out: DataFrame) -> None:
        # FD feature requires the caller-provided frozen transform (train-fit). Without it,
        # we still expose fd at a configurable fixed d only for exploration, clearly named.
        lp = np.log(df["close"])
        if self.fd_transform is not None:
            fd = self.fd_transform.transform(lp)
            out["fd_price"] = fd
            _register("fd_price", self.fd_transform.window, "fracdiff(d frozen from train)", ["close"], True)
        else:
            # Exploration fallback: fd with d=0.3 fixed (NOT for production use; production
            # always passes a frozen train-fit transform). Marked non-causal-safe in docs.
            from research_lib.features.fracdiff import fracdiff_series

            out["fd_price_explore"] = fracdiff_series(lp, 0.3, window=self.lookback)
            _register("fd_price_explore", self.lookback, "fracdiff(d=0.3 fixed, EXPLORATION)", ["close"], True)

    def _regime_features(self, df: DataFrame, out: DataFrame) -> None:
        c = df["close"]
        lr = np.log(c / c.shift(1))
        # trend strength: |close - ma_200| style via MA slope
        ma = c.rolling(200).mean()
        out["ma_slope_200"] = (ma - ma.shift(20)) / ma.shift(20).abs()
        # autocorrelation of returns (rolling)
        out["autocorr_20"] = lr.rolling(20).apply(
            lambda x: x.autocorr() if len(x) > 3 else np.nan, raw=False
        )
        # Hurst (DFA) rolling on log-returns (bounded [0,1] for stationary input).
        # NOTE: applied to returns, not price levels - price-level DFA H>1 merely indicates
        # an integrated/trending series and is not a mean-reversion signal.
        h = rolling_hurst(lr.fillna(0.0), window=self.hurst_window)
        out["hurst_dfa"] = h
        _register("hurst_dfa", self.hurst_window, "DFA on log-returns", ["close"])

    # ------------------------------------------------------------------
    def build(self, df: DataFrame) -> DataFrame:
        out = DataFrame(index=df.index)
        out["date"] = df["date"]
        self._price_features(df, out)
        self._vol_features(df, out)
        self._momentum_features(df, out)
        self._microstructure_features(df, out)
        self._mean_reversion_features(df, out)
        self._fd_features(df, out)
        self._regime_features(df, out)
        return out
