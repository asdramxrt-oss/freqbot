"""Pairs-trading research (blueprint Phase 1-2): spread mean-reversion on 15m data.

Research question (NOT assumed): does local-mean reversion of crypto log-ratios produce a
net, cost-adjusted, out-of-sample edge on 15m data?

Ratios tested: ETH/BTC, SOL/BTC, BNB/BTC, SOL/ETH.
Model: causal trailing-window local z-score (default 7d = 672 bars of 15m), z>=2 entry,
z-cross-0 or time-stop exit. Optional OU-tradability gate.
Execution: dollar-neutral two legs, per-leg taker costs. Research window 2022-01..2025-06
(trimmed at the holdout boundary). Final holdout is evaluated separately.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.constants import HOLDOUT_START
from research_lib.data.loader import load_pair_tf
from research_lib.strategies.pairs import PairsParams, generate_signals, log_ratio_series
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator
from research_lib.risk.costs import CostModel
from research_lib.validation.metrics import summarize_backtest

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "pairs"
RATIOS = [("ETH/USDT", "BTC/USDT"), ("SOL/USDT", "BTC/USDT"),
          ("BNB/USDT", "BTC/USDT"), ("SOL/USDT", "ETH/USDT")]
TF = "15m"
EQUITY = 100_000.0


def _save(df, name: str) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    p = ARTIFACTS / name
    if isinstance(df, pd.DataFrame):
        df.to_csv(p.with_suffix(".csv"), index=False)
    else:
        p.with_suffix(".json").write_text(json.dumps(df, indent=2, default=str))
    return p


def load_pair_trimmed(pair: str) -> pd.DataFrame:
    df = load_pair_tf(pair, TF)
    return df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)


def run_ratio(leg_a: str, leg_b: str, params: PairsParams, *, ou_gate=False,
              capital_fraction=0.25):
    a = load_pair_trimmed(leg_a)
    b = load_pair_trimmed(leg_b)
    m = log_ratio_series(a, b)
    # causal OU tradability filter (walk-forward style, computed on trailing windows)
    ou_arr = None
    if ou_gate:
        ou_arr = np.zeros(len(m), dtype=bool)
        step = 672
        win = params.est_window_bars
        from research_lib.strategies.ou import fit_ou, OUParams

        for t in range(win, len(m), step):
            seg = m["spread"].iloc[t - win : t]
            mdl = fit_ou(seg.reset_index(drop=True), OUParams(min_sample=params.ou_min_sample,
                                                              rolling_window=min(600, win // 2)))
            ou_arr[t : t + step] = mdl.valid
        # forward-fill any NaN region at the start stays False (no trade) - safe default
    res = generate_signals(m, params, ou_valid_arr=ou_arr)
    sim = PairsSimulator(PairsSimParams(equity=EQUITY, capital_fraction=capital_fraction,
                                        cost=params.cost))
    trades, eq = sim.run(m, res.target)
    return m, res, trades, eq


def main() -> None:
    t0 = time.time()
    print("=== PAIRS RESEARCH (15m, research window) ===")
    rows = []
    all_trades = []
    for leg_a, leg_b in RATIOS:
        name = f"{leg_a.split('/')[0]}/{leg_b.split('/')[0]}"
        p = PairsParams(est_window_bars=672, z_entry=2.0, z_exit=0.0, ou_gate=False,
                        cost=CostModel())
        m, res, trades, eq = run_ratio(leg_a, leg_b, p)
        # restrict to the region AFTER the estimation warmup for a fair metric
        s = summarize_backtest(trades, None, timeframe=TF, start_equity=EQUITY)
        rows.append({"ratio": name, "n_trades": s["n_trades"], "win": round(s["win_rate"], 3),
                     "pnl": round(s["total_pnl"], 0), "sharpe": round(s["sharpe"], 2),
                     "maxdd": round(s["max_dd"], 4), "pf": round(s["profit_factor"], 2),
                     "fees": round(s["fees_total"], 0), "avg_bars": round(s.get("avg_bars_held", 0), 1)})
        t = trades.copy()
        t["ratio"] = name
        all_trades.append(t)
        print(f"  {name:9s}: trades={s['n_trades']:5d} win={s['win_rate']:.2f} "
              f"pnl=${s['total_pnl']:>10,.0f} sharpe={s['sharpe']:+.2f} "
              f"maxdd={s['max_dd']:.1%} pf={s['profit_factor']:.2f} fees=${s['fees_total']:,.0f}")
    pooled = pd.concat(all_trades, ignore_index=True)
    sp = summarize_backtest(pooled, None, timeframe=TF, start_equity=EQUITY)
    print(f"  POOLED  : trades={sp['n_trades']:5d} win={sp['win_rate']:.2f} "
          f"pnl=${sp['total_pnl']:>10,.0f} sharpe={sp['sharpe']:+.2f} "
          f"maxdd={sp['max_dd']:.1%} pf={sp['profit_factor']:.2f}")
    res_df = pd.DataFrame(rows)
    _save(res_df, "per_ratio")
    _save(pooled, "pooled_trades")
    _save(sp, "pooled_metrics")
    print(f"  time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
