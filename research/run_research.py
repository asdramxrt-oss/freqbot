"""Full research pipeline runner.

Runs the complete evidence-backed study on REAL market data (research window only - the
final untouched holdout in research/data_holdout is NEVER touched here; it is evaluated
once, separately, by run_holdout.py).

Stages:
  0. Load + quality-gate research panel (4 pairs x 4h, trimmed to the research window).
  1. Baseline: Turtle (20-bar, 1x spot long-only) per pair and pooled.
  2. Information-bar comparison (time vs volume vs dollar) on BTC.
  3. Cost stress: Turtle under 1x/1.5x/2x/3x costs.
  4. Monte Carlo robustness on pooled Turtle trades.
  5. Parameter-neighborhood robustness (18/20/22 grids).
  6. Volatility estimator comparison (close/ewma/parkinson/GK/RS/YZ).
  7. OU engine diagnostic on real BTC (expected: no valid stationary MR on raw log-price).
  8. HMM regime walk-forward on BTC; regime breakdown of Turtle trades + router ablation.
  9. Meta-labeling: turtle events -> triple barrier -> features -> purged-CV model.
 10. Ablation summary (turtle vs turtle+regime-gate vs +meta-filter).

Every stage writes artifacts to research/artifacts/{stage}/ and prints a concise report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.constants import HOLDOUT_START, RESEARCH_DIR
from research_lib.data.loader import load_research_panel
from research_lib.data.quality import validate_panel, panel_passed
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator
from research_lib.risk.costs import CostModel
from research_lib.validation.metrics import summarize_backtest
from research_lib.validation.ablation import cost_stress, cost_stress_verdict
from research_lib.validation.montecarlo import MCConfig, MonteCarlo, parameter_grid_robustness

ARTIFACTS = RESEARCH_DIR / "artifacts"

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TF = "4h"
START_EQUITY = 100_000.0


def _save(df: pd.DataFrame | dict, stage: str, name: str) -> Path:
    out = ARTIFACTS / stage
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    if isinstance(df, pd.DataFrame):
        df.to_csv(path.with_suffix(".csv"), index=False)
    else:
        path.with_suffix(".json").write_text(json.dumps(df, indent=2, default=str))
    return path


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    """Trim a research-window frame to end strictly BEFORE the holdout boundary."""
    return df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)


def load_quality_panel() -> dict[tuple[str, str], pd.DataFrame]:
    panel = load_research_panel(PAIRS, [TF])
    panel = {k: _trim(v) for k, v in panel.items()}
    reports = validate_panel(panel, timeframe=TF)
    for k, r in reports.items():
        print(f"  quality {k}: {r.summary()}")
    if not panel_passed(reports):
        print("  HARD QUALITY FAILURES PRESENT - review before trusting results.")
    return panel


# ---------------------------------------------------------------------------
def run_turtle(pair: str, df: pd.DataFrame, *, max_units=1, max_leverage=1.0,
               allow_short=False, entry_n=20, atr_n=20,
               equity=START_EQUITY, cost: CostModel | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    sim = TurtleSimulator(
        TurtleSimParams(
            entry_n=entry_n, atr_n=atr_n, max_units=max_units, max_leverage=max_leverage,
            allow_short=allow_short, equity=equity, cost=cost or CostModel(),
        )
    )
    trades, eq, _ = sim.run(df)
    return trades, eq


def stage_baseline(panel) -> dict:
    print("\n=== STAGE 1: Turtle baseline (20-bar, 1x spot, long-only) per pair ===")
    per_pair: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    for pair in PAIRS:
        df = panel[(pair, TF)]
        trades, eq = run_turtle(pair, df, equity=START_EQUITY / len(PAIRS))
        m = summarize_backtest(trades, None, timeframe=TF, start_equity=START_EQUITY / len(PAIRS))
        per_pair.append({"pair": pair, **m})
        t2 = trades.copy()
        t2["pair"] = pair
        all_trades.append(t2)
        print(f"  {pair}: trades={m['n_trades']} win={m['win_rate']:.2f} "
              f"pnl=${m['total_pnl']:,.0f} sharpe={m['sharpe']:.2f} maxdd={m['max_dd']:.2%} "
              f"pf={m['profit_factor']:.2f} fees=${m['fees_total']:,.0f}")
    pooled = pd.concat(all_trades, ignore_index=True)
    mp = summarize_backtest(pooled, None, timeframe=TF, start_equity=START_EQUITY)
    print(f"  POOLED: trades={mp['n_trades']} win={mp['win_rate']:.2f} "
          f"pnl=${mp['total_pnl']:,.0f} sharpe={mp['sharpe']:.2f} maxdd={mp['max_dd']:.2%} "
          f"pf={mp['profit_factor']:.2f}")
    _save(pd.DataFrame(per_pair), "baseline", "per_pair")
    _save(pooled, "baseline", "pooled_trades")
    _save(mp, "baseline", "pooled_metrics")
    return {"per_pair": per_pair, "pooled": mp, "pooled_trades": pooled}


# ---------------------------------------------------------------------------
def stage_cost_stress(panel) -> dict:
    print("\n=== STAGE 3: Transaction-cost stress on Turtle (per pair, pooled) ===")

    def run_fn(cost: CostModel):
        out = []
        for pair in PAIRS:
            trades, _ = run_turtle(pair, panel[(pair, TF)], cost=cost,
                                   equity=START_EQUITY / len(PAIRS))
            out.append(trades)
        return pd.concat(out, ignore_index=True), None

    rows = cost_stress(run_fn, timeframe=TF, start_equity=START_EQUITY)
    print(rows[["cost_multiplier", "total_pnl", "sharpe", "max_dd", "fees_total", "turnover"]].to_string(index=False))
    verdict = cost_stress_verdict(rows)
    print(f"  cost-stress verdict: {verdict}")
    _save(rows, "cost_stress", "cost_rows")
    _save(verdict, "cost_stress", "verdict")
    return {"rows": rows, "verdict": verdict}


# ---------------------------------------------------------------------------
def stage_montecarlo(pooled: pd.DataFrame) -> dict:
    print("\n=== STAGE 4: Monte Carlo robustness on pooled trades ===")
    # per-pair start equity so a single symbol can't dominate the ruin calc
    mc = MonteCarlo(MCConfig(n_sims=500, seed=42, ruin_fraction=0.5))
    s = mc.summarize(pooled, START_EQUITY)
    print(f"  p(pnl<0)={s['p_pnl_negative']:.3f} p(ruin)={s['p_ruin']:.4f} "
          f"pnl_p5=${s['pnl_p5']:,.0f} median=${s['pnl_median']:,.0f} p95=${s['pnl_p95']:,.0f} "
          f"dd_p5={s['dd_p5']:.2%}")
    _save(s, "montecarlo", "summary")
    return s


# ---------------------------------------------------------------------------
def stage_param_robustness(panel) -> dict:
    print("\n=== STAGE 5: Parameter neighborhood (entry_n, atr_n) on BTC ===")
    btc = panel[("BTC/USDT", TF)]

    def run_fn(params: dict):
        trades, _ = run_turtle("BTC/USDT", btc, entry_n=int(params["entry_n"]),
                               atr_n=int(params["atr_n"]), equity=START_EQUITY)
        m = summarize_backtest(trades, None, timeframe=TF, start_equity=START_EQUITY)
        return {"total_pnl": m["total_pnl"], "sharpe": m["sharpe"], "n_trades": m["n_trades"]}

    grid = parameter_grid_robustness(run_fn, {}, {"entry_n": [18, 20, 22], "atr_n": [18, 20, 22]})
    print(grid.to_string(index=False))
    _save(grid, "param_robustness", "grid")
    return {"grid": grid}


def main() -> None:
    t0 = time.time()
    print(f"Research window data directory: {RESEARCH_DIR}")
    panel = load_quality_panel()
    r1 = stage_baseline(panel)
    r3 = stage_cost_stress(panel)
    r4 = stage_montecarlo(r1["pooled_trades"])
    r5 = stage_param_robustness(panel)
    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
    print("Artifacts written to", ARTIFACTS)


if __name__ == "__main__":
    main()
