"""Live market-data adapter: poll USDT-M perp 1h klines and extend local panels.

The router book trades 4h bars, so REST polling is the right tool (no websocket
needed): every poll fetches the most recent 1h klines per symbol, drops the
IN-PROGRESS bar (only closed bars are ever returned), and appends anything newer
than the local history. Panels are persisted back to the feather files the
research stack already reads, so the live path and the research path always see
one continuous series.

ccxt is an optional dependency (live only); importing this module without it
raises a clear error at feed construction time, not at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from research_lib.bot.router_book_runner import TREND_SYMS


@dataclass
class FetchResult:
    symbol: str
    new_bars: int
    last_closed: pd.Timestamp | None


class PerpDataFeed:
    """Poll closed 1h perp klines for the router-book universe."""

    def __init__(self, symbols: tuple[str, ...] = TREND_SYMS, exchange_id: str = "binanceusdm"):
        try:
            import ccxt  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - environment guard
            raise RuntimeError(
                "ccxt is required for the live feed (.venv: pip install ccxt)"
            ) from e
        self.symbols = symbols
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    @staticmethod
    def closed_only(df: DataFrame, now: pd.Timestamp | None = None) -> DataFrame:
        """Drop the in-progress bar: a 1h bar is closed only when now >= date + 1h."""
        if df.empty:
            return df
        now = now or pd.Timestamp.now(tz="UTC")
        close_time = df["date"] + pd.Timedelta(hours=1)
        return df[close_time <= now].reset_index(drop=True)

    def fetch_recent(self, symbol: str, since_ms: int | None = None, limit: int = 200) -> DataFrame:
        """Fetch recent CLOSED 1h klines for one symbol as a standard frame."""
        pair = f"{symbol}/USDT:USDT"
        rows = self.exchange.fetch_ohlcv(pair, timeframe="1h", since=since_ms, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["date", "open", "high", "low", "close", "volume"]].astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )
        return self.closed_only(df)

    def update_panel(self, symbol: str, feather_path: Path) -> FetchResult:
        """Append new closed bars to a symbol's feather panel. Returns what changed."""
        local = pd.read_feather(feather_path)
        local["date"] = pd.to_datetime(local["date"], utc=True, format="mixed")
        last_ts = int(local["date"].max().timestamp() * 1000) + 1
        recent = self.fetch_recent(symbol, since_ms=last_ts)
        new = recent[recent["date"] > local["date"].max()]
        if new.empty:
            return FetchResult(symbol, 0, local["date"].max())
        merged = (
            pd.concat([local, new])
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
        merged.to_feather(feather_path)
        return FetchResult(symbol, len(new), merged["date"].max())
