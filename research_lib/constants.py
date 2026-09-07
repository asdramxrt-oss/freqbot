"""Shared constants and canonical data contract for the research stack.

Canonical market-data schema (columns on every normalized OHLCV frame):
    date          datetime64[ns, UTC]   candle open time (the bar's timestamp)
    symbol        str                   e.g. "BTC/USDT"
    exchange      str
    timeframe     str                   e.g. "1h"
    open/high/low/close  float
    volume        float
    quote_volume  float                 (optional)
    trade_count   float                 (optional)
    session       str                   explicit session label, e.g. "CONT" for 24/7 crypto

Downstream modules MUST consume this schema. Freqtrade/ccxt frames are normalized into it at
the data boundary; nothing downstream reads exchange-native formats directly.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "research"
DATA_DIR = PROJECT_ROOT / "research" / "data"
RESEARCH_DATA_DIR = DATA_DIR
HOLDOUT_DATA_DIR = PROJECT_ROOT / "research" / "data_holdout"
REPORTS_DIR = RESEARCH_DIR / "reports"

# ---------------------------------------------------------------------------
# Data window discipline (unchanged once research begins)
# ---------------------------------------------------------------------------
RESEARCH_START = "2022-01-01"
RESEARCH_END = "2025-06-01"          # research/walk-forward/CPCV window
HOLDOUT_START = "2025-06-01"         # final untouched holdout begins here
HOLDOUT_END = "2026-08-01"

# A final "never touched during research" margin inside the holdout is enforced by the
# validation layer (see validation/holdout.py). These bounds are the hard data limits.

# ---------------------------------------------------------------------------
# Canonical column contract
# ---------------------------------------------------------------------------
CANDLE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_COLUMNS = [
    "quote_volume",
    "trade_count",
    "bid",
    "ask",
    "spread",
    "session",
    "exchange",
    "timeframe",
    "symbol",
]

# Every candle frame produced by the normalizer carries these identifier columns when the
# loader knows them (symbol/exchange/timeframe are filled by the loaders).
META_COLUMNS = ["symbol", "exchange", "timeframe"]

# ---------------------------------------------------------------------------
# Timeframe grid (seconds)
# ---------------------------------------------------------------------------
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# ---------------------------------------------------------------------------
# Turtle specification defaults (documented; tuned only inside train windows)
# ---------------------------------------------------------------------------
TURTLE_DEFAULTS = {
    "system1_entry": 20,   # N-bar breakout
    "system2_entry": 55,
    "atr_period": 20,      # N = ATR(20)
    "stop_n": 2.0,         # initial stop 2N
    "pyramid_max_units": 4,
    "pyramid_step_n": 0.5,  # add a unit every 0.5N in favorable direction
    "risk_per_unit": 0.01,  # 1N move = 1% equity per unit
}

# OU defaults
OU_DEFAULTS = {
    "z_entry": 2.0,
    "z_exit": 0.0,
    "half_life_timeout_mult": 2.5,  # time-stop in half-lives
    "min_sample": 120,
}

# Regime / HMM
HMM_DEFAULTS = {
    "n_states": 3,
    "train_days": 180,
    "refit_days": 7,
    "seed": 42,
}

# Validation defaults
VALIDATION_DEFAULTS = {
    "embargo_bars": 24,        # 1 day on 1h bars
    "purge_overlap": True,
    "min_train_bars": 1000,
    "n_splits": 5,
    "cpcv_n": 6,
    "cpcv_k": 2,
}

# Cost model defaults (maker/taker on Binance spot, worst-case assumptions)
COST_DEFAULTS = {
    "maker_fee": 0.0010,   # 10 bps
    "taker_fee": 0.0010,
    "spread_bps": 1.0,     # 1 bp half-spread applied on entry+exit
    "slippage_bps": 0.5,   # additional adverse slippage per fill
    "funding_period_h": 8,  # not applied on spot (kept for futures models)
    "latency_bars": 0,      # execution delayed by this many bars (0 = next-bar open)
}

# Canonical bar types for information-driven bar experiments
BAR_TYPES = ["time", "volume", "dollar", "imbalance"]
