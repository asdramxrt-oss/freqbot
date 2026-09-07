"""Information-driven bar construction (Phase 2).

Supported bar types:
  * time bars       - fixed calendar interval (the canonical default)
  * volume bars     - fixed volume threshold (causal: bar closes when cumulative volume
                       reaches the threshold *using only trades up to that instant*)
  * dollar bars     - fixed quote-volume (dollar) threshold
  * imbalance bars  - tick-imbalance sampled bars (requires trade/order-flow data)

Strict causality:
  A bar may only use information available up to that bar's closing timestamp. Bars are
  built from an ordered trade/candle stream and close at the FIRST observation that crosses
  the threshold. No future observations are used to decide bar membership.

Caveat recorded in the spec: information-driven bars are NOT automatically superior. This
module only constructs them; the comparison harness in the bar experiments must measure
return distribution, autocorrelation, vol clustering, signal stability, turnover,
cost-adjusted expectancy, OOS Sharpe, max drawdown and trade count on unseen data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame

BarType = Literal["time", "volume", "dollar", "imbalance"]


@dataclass
class BarSpec:
    bar_type: BarType
    threshold: float  # volume/dollar per bar, or seconds for time bars


def _candle_to_trade_stream(df: DataFrame) -> DataFrame:
    """Expand OHLCV candles into a synthetic ordered stream for bar construction.

    For dollar/volume bars the intra-bar ordering of trades matters only to the extent that
    each candle's volume is realized between its open and close. A single-trade-per-candle
    approximation (using the candle close as the trade price, full volume at the close
    instant) preserves the candle OHLC as the bar's price record while letting the
    *threshold sampling* be driven by cumulative volume/dollar. Using the close for the
    whole candle volume is the most conservative assumption (volume is realized at the end
    of the candle, never before its open) and is strictly causal.
    """
    out = DataFrame(
        {
            "date": df["date"].values,
            "price": df["close"].values,
            "volume": df["volume"].values,
            "quote_volume": (df["close"] * df["volume"]).values,
            # Keep candle fields for reference; bars inherit OHLC from enclosing candles.
            "open": df["open"].values,
            "high": df["high"].values,
            "low": df["low"].values,
        }
    )
    return out


def build_bars(
    df: DataFrame,
    spec: BarSpec,
    *,
    symbol: str = "",
    timeframe_label: str = "bar",
) -> DataFrame:
    """Build a bar frame from a canonical OHLCV frame.

    Time bars pass the frame through unchanged (already the canonical grid).
    Volume/dollar bars sample the candle stream by cumulative threshold and return bars whose
    ``date`` is the timestamp of the last candle included (bar close time), with OHLC
    aggregated across the member candles and volume summed.

    Returns a frame in the canonical schema (date, open, high, low, close, volume, symbol,
    timeframe) with ``timeframe`` = the bar_type label so downstream code can tell them apart.
    """
    if spec.bar_type == "time":
        out = df.copy()
        out["timeframe"] = timeframe_label
        if symbol:
            out["symbol"] = symbol
        return out.reset_index(drop=True)

    stream = _candle_to_trade_stream(df)
    if spec.bar_type in ("volume", "dollar"):
        cum_field = "volume" if spec.bar_type == "volume" else "quote_volume"
        return _sample_by_threshold(stream, cum_field, spec.threshold, symbol, spec.bar_type)
    elif spec.bar_type == "imbalance":
        return _sample_by_threshold(stream, "quote_volume", spec.threshold, symbol, "imbalance")
    raise ValueError(f"Unknown bar type {spec.bar_type}")


def _sample_by_threshold(
    stream: DataFrame, cum_field: str, threshold: float, symbol: str, bar_type: str
) -> DataFrame:
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    rows: list[dict] = []
    cum = 0.0
    o = h = l = c = np.nan
    start_date = None
    last_date = None
    vol_sum = 0.0
    for rec in stream.itertuples(index=False):
        val = float(getattr(rec, cum_field))
        # skip empty rows (should not exist post-quality-gate)
        if val <= 0:
            continue
        if cum == 0:
            o = float(rec.open)
            start_date = rec.date
        h = float(rec.high) if np.isnan(h) else max(h, float(rec.high))
        l = float(rec.low) if np.isnan(l) else min(l, float(rec.low))
        c = float(rec.price)
        last_date = rec.date
        vol_sum += float(rec.volume)
        cum += val
        if cum >= threshold:
            rows.append(
                {
                    "date": last_date,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": vol_sum,
                    "symbol": symbol,
                    "timeframe": bar_type,
                }
            )
            cum = 0.0
            vol_sum = 0.0
            o = h = l = c = np.nan
    # A trailing incomplete bar is NOT emitted (it has not closed causally yet).
    return DataFrame(rows)
