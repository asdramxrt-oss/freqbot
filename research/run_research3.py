"""Stage 9-10: Walk-forward Turtle + look-ahead regression gate.

Walk-forward discipline: the Turtle parameters (20/20/2N) are FIXED research defaults; this
stage evaluates the strategy on successive non-overlapping out-of-sample windows to see
whether the edge is stable through time or only exists in the bull segment.

Look-ahead gate: the Turtle simulator's entry decisions on the truncated prefix of the data
must be identical to those on the full data (signals cannot change when future data is
added) - the automated leakage test from the spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.constants import HOLDOUT_START
from research_lib.data.loader import load_pair_tf
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator
from research_lib.validation.metrics import summarize_backtest

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts"
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TF = "4h"
START_EQUITY = 100_000.0


def _save(df, stage: str, name: str) -> Path:
    out = ARTIFACTS / stage
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    if isinstance(df, pd.DataFrame):
        df.to_csv(path.with_suffix(".csv"), index=False)
    else:
        path.with_suffix(".json").write_text(json.dumps(df, indent=2, default=str))
    return path


def _run_one(df: pd.DataFrame, sub_eq: float):
    sim = TurtleSimulator(TurtleSimParams(equity=sub_eq, allow_short=False, max_leverage=1.0))
    trades, eq, log = sim.run(df)
    return trades, eq, log


def stage_walkforward_turtle() -> pd.DataFrame:
    print("\n=== STAGE 9: Walk-forward Turtle - edge stability through time (pooled) ===")
    rows = []
    # quarterly non-overlapping windows over the research period
    df_btc = load_pair_tf("BTC/USDT", TF)
    df_btc = df_btc[df_btc["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)
    start = pd.Timestamp("2022-01-01", tz="UTC")
    q = pd.tseries.offsets.DateOffset(months=6)
    window_start = start
    for i in range(7):
        window_end = min(window_start + q, df_btc["date"].max())
        sub = df_btc[(df_btc["date"] >= window_start) & (df_btc["date"] < window_end)]
        if len(sub) < 500:
            break
        trades, _, _ = _run_one(sub, START_EQUITY / 4)
        m = summarize_backtest(trades, None, timeframe=TF, start_equity=START_EQUITY / 4)
        rows.append({
            "window": f"{window_start.date()}..{window_end.date()}",
            "trades": m["n_trades"],
            "total_pnl": round(m["total_pnl"], 0),
            "sharpe": round(m["sharpe"], 2),
            "pf": round(m["profit_factor"], 2),
            "maxdd": round(m["max_dd"], 4),
        })
        print(f"  {window_start.date()}..{window_end.date()}: trades={m['n_trades']:3d} "
              f"pnl=${m['total_pnl']:>9,.0f} sharpe={m['sharpe']:+.2f} pf={m['profit_factor']:.2f} "
              f"maxdd={m['max_dd']:.1%}")
        window_start = window_end
    res = pd.DataFrame(rows)
    _save(res, "walkforward", "turtle_semiannual")
    return res


def stage_lookahead_gate() -> dict:
    print("\n=== STAGE 10: Look-ahead regression gate (signals must be prefix-invariant) ===")
    df = load_pair_tf("BTC/USDT", TF)
    df = df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)
    full_trades, _, full_log = _run_one(df, START_EQUITY / 4)
    full_entries = {(x["date"], x["side"]) for x in full_log if x["action"] == "entry"}
    # truncated at 60% of the data - a causal strategy must produce identical entries on
    # the shared prefix
    cut = int(len(df) * 0.6)
    prefix = df.iloc[:cut].reset_index(drop=True)
    _, _, prefix_log = _run_one(prefix, START_EQUITY / 4)
    prefix_entries = {(x["date"], x["side"]) for x in prefix_log if x["action"] == "entry"}
    # full-data entries that fall within the prefix window
    full_within = {e for e in full_entries if e[0] <= prefix["date"].iloc[-1]}
    missing = full_within - prefix_entries
    extra = prefix_entries - full_within
    print(f"  full entries within prefix: {len(full_within)}, prefix entries: {len(prefix_entries)}")
    print(f"  missing (future-dependent): {len(missing)}, extra (future-suppressed): {len(extra)}")
    passed = len(missing) == 0 and len(extra) == 0
    print(f"  look-ahead gate: {'PASS' if passed else 'FAIL'}")
    result = {"passed": passed, "n_missing": len(missing), "n_extra": len(extra)}
    _save(result, "lookahead_gate", "result")
    return result


def main() -> None:
    r9 = stage_walkforward_turtle()
    r10 = stage_lookahead_gate()


if __name__ == "__main__":
    main()
