"""Final untouched holdout evaluation (runs ONCE after all research is frozen).

The 2025-06-01 -> 2026-09-07 data in research/data_holdout has NEVER been touched by any
research stage. This script applies the FINAL selected configuration (the only one that
passed every research gate: Turtle 20-bar long-only, 1x spot, realistic costs) and reports
its out-of-sample performance.

Rules honored:
  * run exactly once for the final decision
  * no parameter changes after seeing results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.constants import HOLDOUT_START
from research_lib.data.loader import load_holdout_panel
from research_lib.data.quality import validate_panel, panel_passed
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator
from research_lib.risk.costs import CostModel
from research_lib.validation.metrics import summarize_backtest
from research_lib.validation.montecarlo import MCConfig, MonteCarlo

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "holdout"
PAIRS = ["BTC/USDT", "ETH/USDT"]
TF = "4h"
START_EQUITY = 100_000.0


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("=== FINAL UNTOUCHED HOLDOUT (2025-06 -> 2026-09) ===")
    print("Data dir: research/data_holdout (never used in research stages)")
    panel = load_holdout_panel(PAIRS, [TF])
    reports = validate_panel(panel, timeframe=TF)
    for k, r in reports.items():
        print(f"  quality {k}: {r.summary()}")

    all_trades = []
    for i, pair in enumerate(PAIRS):
        df = panel[(pair, TF)]
        sub_eq = START_EQUITY / len(PAIRS)
        sim = TurtleSimulator(TurtleSimParams(equity=sub_eq, allow_short=False, max_leverage=1.0))
        trades, eq, log = sim.run(df)
        m = summarize_backtest(trades, None, timeframe=TF, start_equity=sub_eq)
        t2 = trades.copy()
        t2["pair"] = pair
        all_trades.append(t2)
        print(f"  {pair}: trades={m['n_trades']} win={m['win_rate']:.2f} "
              f"pnl=${m['total_pnl']:,.0f} sharpe={m['sharpe']:.2f} maxdd={m['max_dd']:.2%} "
              f"pf={m['profit_factor']:.2f} fees=${m['fees_total']:,.0f}")
    pooled = pd.concat(all_trades, ignore_index=True)
    mp = summarize_backtest(pooled, None, timeframe=TF, start_equity=START_EQUITY)
    print(f"  POOLED HOLDOUT: trades={mp['n_trades']} win={mp['win_rate']:.2f} "
          f"pnl=${mp['total_pnl']:,.0f} sharpe={mp['sharpe']:.2f} maxdd={mp['max_dd']:.2%} "
          f"pf={mp['profit_factor']:.2f} total_return={mp['total_return']:.2%}")
    mc = MonteCarlo(MCConfig(n_sims=500, seed=42)).summarize(pooled, START_EQUITY)
    print(f"  MC: p(pnl<0)={mc['p_pnl_negative']:.3f} p(ruin)={mc['p_ruin']:.4f} "
          f"pnl_p5=${mc['pnl_p5']:,.0f} median=${mc['pnl_median']:,.0f} dd_p5={mc['dd_p5']:.2%}")
    pooled.to_csv(ARTIFACTS / "pooled_trades.csv", index=False)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(mp, indent=2, default=str))
    (ARTIFACTS / "montecarlo.json").write_text(json.dumps(mc, indent=2, default=str))


if __name__ == "__main__":
    main()
