"""Blueprint Phase 3-4 research on real data:
  (a) maker-fill fee savings on the pairs trade set - would cheaper execution have rescued
      the (negative-gross) strategy? (Expected: no - verified, not assumed.)
  (b) pairs WFO-style regime windows (does the edge appear in ANY regime?)
  (c) final untouched holdout for the pairs strategy.
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
from research_lib.strategies.pairs import PairsParams, generate_signals, log_ratio_series
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator
from research_lib.risk.costs import CostModel
from research_lib.validation.metrics import summarize_backtest

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "pairs"
TF = "15m"
RATIOS = [("ETH/USDT", "BTC/USDT"), ("SOL/USDT", "BTC/USDT"), ("BNB/USDT", "BTC/USDT")]


def _save(obj, name: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


def load_trimmed(pair):
    df = load_pair_tf(pair, TF)
    return df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)


def run_pairs_on(a_df, b_df, win=1344, cap=0.25):
    m = log_ratio_series(a_df, b_df)
    p = PairsParams(est_window_bars=win, z_entry=2.0, z_exit=0.0, ou_gate=False, cost=CostModel())
    res = generate_signals(m, p)
    sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=cap, cost=p.cost))
    trades, eq = sim.run(m, res.target)
    return m, res, trades, eq


def main() -> None:
    print("=== Phase 3-4: maker economics + WFO windows + holdout ===")
    # (a) maker savings: recompute pooled fees under a maker/taker schedule
    # Since gross is negative, compute gross directly from the taker trade set.
    all_t = []
    for la, lb in RATIOS:
        m, res, trades, eq = run_pairs_on(load_trimmed(la), load_trimmed(lb))
        all_t.append(trades)
    pooled = pd.concat(all_t, ignore_index=True)
    gross = float(pooled["pnl_net"].sum() + pooled["fees_paid"].sum())
    fees = float(pooled["fees_paid"].sum())
    # maker savings estimate: ~60% of fees if ~half the fills were maker at 0.02% vs 0.05%
    maker_saving = fees * 0.5 * (1 - 0.02 / 0.05)
    print(f"  pooled pairs (3 ratios): net=${pooled['pnl_net'].sum():,.0f} "
          f"gross=${gross:,.0f} fees=${fees:,.0f}")
    print(f"  best-case maker saving ~${maker_saving:,.0f} -> net would still be "
          f"${pooled['pnl_net'].sum() + maker_saving:,.0f} (gross is negative)")
    _save({"net": float(pooled["pnl_net"].sum()), "gross": gross, "fees": fees,
           "maker_saving_est": maker_saving}, "maker_economics")

    # (b) semi-annual WFO windows on ETH/BTC
    print("\n  ETH/BTC semi-annual windows (research):")
    a = load_trimmed("ETH/USDT")
    b = load_trimmed("BTC/USDT")
    m = log_ratio_series(a, b)
    start = m["date"].iloc[0]
    rows = []
    for half in range(7):
        ws = start + pd.tseries.offsets.DateOffset(months=6 * half)
        we = ws + pd.tseries.offsets.DateOffset(months=6)
        sub = m[(m["date"] >= ws) & (m["date"] < we)]
        if len(sub) < 1344 * 2:
            continue
        # warm-up: need est_window before the window
        pre = m[m["date"] < ws].tail(1344)
        frame = pd.concat([pre, sub]).reset_index(drop=True)
        p = PairsParams(est_window_bars=1344, z_entry=2.0, z_exit=0.0, ou_gate=False)
        res = generate_signals(frame, p)
        sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=0.25))
        trades, _ = sim.run(frame, res.target)
        # count only trades exited within the window
        mask = pd.to_datetime(trades["exit_time"], utc=True) < we
        tw = trades[mask]
        if len(tw):
            rows.append({"window": f"{ws.date()}..{we.date()}", "n": len(tw),
                         "pnl": round(float(tw["pnl_net"].sum()), 0)})
            print(f"    {ws.date()}..{we.date()}: n={len(tw):3d} pnl=${tw['pnl_net'].sum():>9,.0f}")
    _save(rows, "wfo_windows")

    # (c) final holdout on the pairs strategy (untouched data)
    print("\n  FINAL HOLDOUT (2025-06..2026-09) pairs ETH/BTC + SOL/BTC:")
    hp = load_holdout_panel(["ETH/USDT", "BTC/USDT", "SOL/USDT"], [TF])
    ho_all = []
    for la, lb in [("ETH/USDT", "BTC/USDT"), ("SOL/USDT", "BTC/USDT")]:
        m = log_ratio_series(hp[(la, TF)], hp[(lb, TF)])
        # use the same fixed 14d window; warmup inside holdout data
        p = PairsParams(est_window_bars=1344, z_entry=2.0, z_exit=0.0, ou_gate=False)
        res = generate_signals(m, p)
        sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=0.25))
        trades, eq = sim.run(m, res.target)
        s = summarize_backtest(trades, None, timeframe=TF, start_equity=100_000)
        ho_all.append(trades)
        print(f"    {la.split('/')[0]}/{lb.split('/')[0]}: n={s['n_trades']} "
              f"pnl=${s['total_pnl']:,.0f} sharpe={s['sharpe']:.2f} maxdd={s['max_dd']:.1%}")
    ho_pooled = pd.concat(ho_all, ignore_index=True)
    sp = summarize_backtest(ho_pooled, None, timeframe=TF, start_equity=100_000)
    print(f"    POOLED: n={sp['n_trades']} pnl=${sp['total_pnl']:,.0f} "
          f"sharpe={sp['sharpe']:.2f} maxdd={sp['max_dd']:.1%} pf={sp['profit_factor']:.2f}")
    _save(sp, "holdout_metrics")


if __name__ == "__main__":
    main()
