"""Canonical market-data loaders.

All loaders return frames conforming to the canonical schema (see constants.CANDLE_COLUMNS)
with UTC-normalized ``date`` index/sort, float dtypes, and no fabricated rows.

Raw data lives in freqtrade-native feather files (research/data/<exchange>/<pair>-<tf>.feather)
and is NEVER overwritten by this package. Transformed/derived data is written separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from research_lib.constants import CANDLE_COLUMNS, HOLDOUT_DATA_DIR, RESEARCH_DATA_DIR


def _normalize(df: DataFrame, symbol: str, exchange: str, timeframe: str) -> DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    # Keep canonical columns in order; ignore extras the raw file may carry.
    cols = [c for c in CANDLE_COLUMNS if c in df.columns]
    df = df[cols].sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol
    df["exchange"] = exchange
    df["timeframe"] = timeframe
    for col in CANDLE_COLUMNS:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_feather(
    path: Path,
    symbol: str,
    exchange: str = "binance",
    timeframe: str = "1h",
) -> DataFrame:
    """Load a freqtrade OHLCV feather into the canonical schema (no fill, no drop)."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_feather(path)
    return _normalize(df, symbol, exchange, timeframe)


def load_pair_tf(
    pair: str,
    timeframe: str,
    exchange: str = "binance",
    data_dir: Path | None = None,
) -> DataFrame:
    """Load a pair from the freqtrade data-directory layout.

    Layout: <data_dir>/<exchange>/<pair-symbol>_<tf>.feather where pair-symbol is
    like ``BTC_USDT`` (slash replaced). This matches freqtrade's own history layout.
    """
    data_dir = data_dir or RESEARCH_DATA_DIR
    symbol = pair.replace("/", "_")
    # freqtrade writes single-exchange data directly in the datadir; multi-exchange layouts
    # use an exchange subdirectory. Try both.
    candidates = [
        data_dir / f"{symbol}-{timeframe}.feather",
        data_dir / exchange / f"{symbol}-{timeframe}.feather",
    ]
    for path in candidates:
        if path.exists():
            return load_feather(path, pair, exchange=exchange, timeframe=timeframe)
    raise FileNotFoundError(f"Data file not found for {pair} {timeframe} under {data_dir}")


def load_research_panel(
    pairs: list[str],
    timeframes: list[str],
    exchange: str = "binance",
    data_dir: Path | None = None,
) -> dict[tuple[str, str], DataFrame]:
    """Load a dict {(pair, timeframe): canonical frame} for the research window."""
    out: dict[tuple[str, str], DataFrame] = {}
    for pair in pairs:
        for tf in timeframes:
            out[(pair, tf)] = load_pair_tf(pair, tf, exchange, data_dir)
    return out


def load_holdout_panel(
    pairs: list[str], timeframes: list[str], exchange: str = "binance"
) -> dict[tuple[str, str], DataFrame]:
    """Load the final untouched holdout (separate directory, never used in research)."""
    return load_research_panel(pairs, timeframes, exchange, data_dir=HOLDOUT_DATA_DIR)
