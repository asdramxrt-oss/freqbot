"""Download USDT-M perpetual futures OHLCV + funding-rate history for the carry book.

Fetches perp 1h candles and 8h funding prints from Binance USDT-M via ccxt (public
endpoints, no keys) and stores them under research/data:

  research/data/futures/{SYM}_USDT-1h-perp.feather   (perp OHLCV, canonical columns)
  research/data/funding/{SYM}_USDT_USDT_funding.csv  (date,funding_rate - existing format)

Why futures, not spot: the carry trade's short leg IS the perp. Marking it on the real
perp series replaces the spot-proxy + constant-basis approximation flagged as the
critical missing input in research/FIX_OR_ENHANCE_ASSESSMENT.md (step 1).

Holdout note: carry is parameter-free (nothing is fitted), and the funding files already
spanned the full range; folds in carry_acceptance prove time-consistency. The directional
research frames remain trimmed at 2025-06-01 and are untouched by this script.

Usage:  .venv/Scripts/python research/download_carry_data.py [--start 2022-01-01]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FUTURES_DIR = ROOT / "research" / "data" / "futures"
FUNDING_DIR = ROOT / "research" / "data" / "funding"
DATA_DIR = ROOT / "research" / "data"
SYMBOLS = ("BTC", "ETH", "BNB", "SOL", "XRP")
# spot 15m symbols the directional book needs but research/data lacks
SPOT_15M = ("XRP",)


def fetch_ohlcv(ex, symbol: str, start_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    since = start_ms
    limit = 1000  # binanceusdm caps klines at 1000 even when 1500 is requested
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 3_600_000
        if len(batch) < limit:
            break
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date")


def fetch_funding(ex, symbol: str, start_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    since = start_ms
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1]["timestamp"]) + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["date", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
    return df.drop_duplicates("date").sort_values("date")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--spot-15m", action="store_true", help="also fetch spot 15m for SPOT_15M symbols")
    args = ap.parse_args()
    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)

    import ccxt  # noqa: PLC0415

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    FUTURES_DIR.mkdir(parents=True, exist_ok=True)
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)

    for sym in SYMBOLS:
        pair = f"{sym}/USDT:USDT"
        t0 = time.time()
        ohlcv = fetch_ohlcv(ex, pair, start_ms)
        out = FUTURES_DIR / f"{sym}_USDT-1h-perp.feather"
        ohlcv.to_feather(out)
        print(f"{pair} perp 1h: {len(ohlcv):,} bars  {ohlcv['date'].min()} -> {ohlcv['date'].max()}  -> {out.name}")

        f_out = FUNDING_DIR / f"{sym}_USDT_USDT_funding.csv"
        if f_out.exists():
            print(f"{pair} funding: exists ({f_out.name}), skipped")
        else:
            funding = fetch_funding(ex, pair, start_ms)
            funding.to_csv(f_out, index=False)
            print(f"{pair} funding: {len(funding):,} prints -> {f_out.name}")
        print(f"  ({time.time() - t0:.0f}s)")

    if args.spot_15m and SPOT_15M:
        spot_ex = ccxt.binance({"enableRateLimit": True})
        for sym in SPOT_15M:
            pair = f"{sym}/USDT"
            out = DATA_DIR / f"{sym}_USDT-15m.feather"
            if out.exists():
                print(f"{pair} spot 15m: exists ({out.name}), skipped")
                continue
            t0 = time.time()
            rows: list[list] = []
            since = start_ms
            while True:
                batch = spot_ex.fetch_ohlcv(pair, timeframe="15m", since=since, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                since = batch[-1][0] + 900_000
                if len(batch) < 1000:
                    break
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date")
            df.to_feather(out)
            print(f"{pair} spot 15m: {len(df):,} bars  {df['date'].min()} -> {df['date'].max()}  -> {out.name}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
