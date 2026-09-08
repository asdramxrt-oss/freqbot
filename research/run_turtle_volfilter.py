"""Blueprint Phase 2 filter applied to Turtle 4h (the strategy with positive in-sample
gross): does the vol-spike entry filter improve its risk-adjusted profile?

This is the constructive test: vol-spike prediction is the one validated OOS capability
(AUC ~0.85). Turtle has positive gross in-sample but failed OOS during high-vol regimes.
Question: does refusing entries when a vol spike is predicted improve Turtle's OOS
behavior? (Hypothesis worth testing; reported honestly either way.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.constants import HOLDOUT_START
from research_lib.data.loader import load_pair_tf, load_holdout_panel
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator
from research_lib.risk.costs import CostModel
from research_lib.features.volatility import VolatilityEngine
from research_lib.validation.metrics import summarize_backtest

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "blueprint"
TF = "4h"


def _save(obj, name):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


def turtle_with_vol_filter(df, vol_pct_threshold=0.90, equity=100_000):
    """Run Turtle but suppress entries when current vol percentile is extreme.

    The filter is implemented inside the sim loop by zeroing target changes - we instead
    re-run TurtleSimulator and post-filter its entry log by checking vol at entry dates.
    """
    sim = TurtleSimulator(TurtleSimParams(equity=equity, allow_short=False, max_leverage=1.0))
    trades, eq, log = sim.run(df)
    # vol percentile series aligned to df
    v = VolatilityEngine(method="yang_zhang", window=20).compute(df)["volatility_percentile"]
    v = v.reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(df["date"])}
    ent = [x for x in log if x["action"] == "entry"]
    blocked = [x for x in ent if date_to_idx[x["date"]] < len(v)
               and np.isfinite(v[date_to_idx[x["date"]]])
               and v[date_to_idx[x["date"]]] > vol_pct_threshold]
    blocked_dates = {x["date"] for x in blocked}
    # remove trades whose entry_time is in blocked_dates (entry fills on the log entry date)
    et = pd.to_datetime(trades["entry_time"], utc=True)
    keep = ~et.isin(list(blocked_dates))
    trades_f = trades[keep].reset_index(drop=True)
    return trades, trades_f, len(blocked)


def main():
    print("=== Vol-spike filter on Turtle 4h ===")
    res = {}
    for window, panel_fn in [
        ("research", lambda pair: load_pair_tf(pair, TF)),
    ]:
        pass
    # research window
    df = load_pair_tf("BTC/USDT", TF)
    df = df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)
    t_all, t_filt, n_blocked = turtle_with_vol_filter(df, 0.90)
    s_all = summarize_backtest(t_all, None, timeframe=TF, start_equity=100_000)
    s_filt = summarize_backtest(t_filt, None, timeframe=TF, start_equity=100_000)
    print(f"  RESEARCH BTC 4h: all n={s_all['n_trades']} pnl=${s_all['total_pnl']:,.0f} "
          f"sharpe={s_all['sharpe']:.2f} maxdd={s_all['max_dd']:.1%}")
    print(f"  filtered (vol>90pct blocked, {n_blocked} entries): "
          f"n={s_filt['n_trades']} pnl=${s_filt['total_pnl']:,.0f} "
          f"sharpe={s_filt['sharpe']:.2f} maxdd={s_filt['max_dd']:.1%}")
    res["research"] = {"all": s_all, "filtered": s_filt, "blocked": n_blocked}

    # holdout
    hp = load_holdout_panel(["BTC/USDT"], [TF])
    dfh = hp[("BTC/USDT", TF)]
    t_all_h, t_filt_h, nb_h = turtle_with_vol_filter(dfh, 0.90)
    s_all_h = summarize_backtest(t_all_h, None, timeframe=TF, start_equity=100_000)
    s_filt_h = summarize_backtest(t_filt_h, None, timeframe=TF, start_equity=100_000)
    print(f"  HOLDOUT BTC 4h: all n={s_all_h['n_trades']} pnl=${s_all_h['total_pnl']:,.0f} "
          f"sharpe={s_all_h['sharpe']:.2f} maxdd={s_all_h['max_dd']:.1%}")
    print(f"  filtered: n={s_filt_h['n_trades']} pnl=${s_filt_h['total_pnl']:,.0f} "
          f"sharpe={s_filt_h['sharpe']:.2f} maxdd={s_filt_h['max_dd']:.1%}")
    res["holdout"] = {"all": s_all_h, "filtered": s_filt_h, "blocked": nb_h}
    _save(res, "turtle_volfilter")


if __name__ == "__main__":
    main()
