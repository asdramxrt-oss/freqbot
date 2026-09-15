"""The PDF Signal Router book, pre-registered 2026-09-14: Turtle + mean-reversion
working ALONGSIDE each other, routed by the causal HMM regime engine.

Design (verbatim from the design PDF / unified_strategy_plan):
    regime=TREND (confirmed trend) & no spike  -> Turtle book trades
    regime=LOW_VOL_MR/NEUTRAL (range) & no spike -> mean-reversion book trades

  TREND sleeve : the validated slow trend (4h Donchian 60/30 + SMA1200 ls, 2xATR
                 stop) on all 5 perps, active only while that coin's router
                 turtle_gate is open.
  MR sleeve    : OU z-score pairs (log A/B ratio, causal z, OU half-life gate ON)
                 on ETH/BTC, SOL/BTC, BNB/BTC at 4h, maker costs, active only
                 while leg A's router pairs_gate is open. BTC therefore trades
                 trend-only; the MR sleeve is short-bailiwick by design.

Equity split equally across 9 sleeves (5 trend + 4 MR). Standard 5-gate bundle,
every gate reported. The MR sleeve's only untested configuration is exactly this
one (HMM-regime-routed) - the structural study ruled out always-on MR, not
regime-conditional MR. Reported pass or fail.

Usage:  .venv/Scripts/python research/run_router_book_acceptance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.bot.directional_runner import (  # noqa: E402
    DirectionalRunnerConfig,
    build_trend_signals,
)
from research_lib.risk.backtest_engine import EngineConfig, EventBacktester  # noqa: E402
from research_lib.risk.costs import CostModel  # noqa: E402
from research_lib.strategies.directional import (  # noqa: E402
    DirectionalResult,
    TRADING_DAYS_PER_YEAR,
    resample_panel,
)
from research_lib.strategies.ou import OUParams, fit_ou  # noqa: E402
from research_lib.strategies.pairs import (  # noqa: E402
    PairsParams,
    generate_signals,
    log_ratio_series,
)
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator  # noqa: E402
from research_lib.strategies.unified_router import RouterConfig, filtered_hmm_router  # noqa: E402
from research_lib.validation.directional_acceptance import (  # noqa: E402
    acceptance_gates,
    evaluate_directional_folds,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data" / "futures"
REPORT = ROOT / "research" / "reports" / "ROUTER_BOOK_ACCEPTANCE_REPORT.md"

TREND_SYMS = ("BTC", "ETH", "BNB", "SOL", "XRP")
# all 5 coins trade in BOTH sleeves: 4 ratios cover every non-BTC leg vs BTC
MR_RATIOS = (("ETH", "BTC"), ("SOL", "BTC"), ("BNB", "BTC"), ("XRP", "BTC"))
EQUITY = 100_000.0
N_SLEEVES = len(TREND_SYMS) + len(MR_RATIOS)

COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)
MR_COST = CostModel(maker_fee=0.0002, taker_fee=0.0002, spread_bps=0.0, slippage_bps=0.0)

ROUTER_CFG = RouterConfig(
    refit_bars=168,            # ~4 weeks of 4h bars
    min_train_bars=1000,
    vol_kill_p=0.80,
    vol_refit_bars=336,
    train_window=1998,         # ~333 days of 4h bars
)

MR_PARAMS = PairsParams(
    est_window_bars=42,        # 7 days of 4h bars
    z_entry=2.0,
    z_exit=0.0,
    ou_gate=True,
    ou_min_sample=30,
    max_hold_bars=84,
)


def _stressed(cost: CostModel, mult: float) -> CostModel:
    return CostModel(
        maker_fee=cost.maker_fee * mult,
        taker_fee=cost.taker_fee * mult,
        spread_bps=cost.spread_bps * mult,
        slippage_bps=cost.slippage_bps * mult,
    )


def _ou_validity(m: pd.DataFrame, params: PairsParams) -> np.ndarray:
    arr = np.zeros(len(m), dtype=bool)
    step, win = 42, params.est_window_bars
    for t in range(win, len(m), step):
        seg = m["spread"].iloc[t - win : t]
        mdl = fit_ou(
            seg.reset_index(drop=True),
            OUParams(min_sample=params.ou_min_sample, rolling_window=min(600, win // 2)),
        )
        arr[t : t + step] = mdl.valid
    return arr


def run_book(cost: CostModel, mr_cost: CostModel, panels_4h, routers,
             trend_gate: str = "hmm", risk: float = 0.01, trend_share: float = 5 / 9):
    """Run all sleeves; return DirectionalResult of the combined portfolio.

    trend_gate: "hmm" (hard turtle_gate) or "volkill" (only the vol-spike kill
    vetoes - the configuration that maximizes trend pnl at higher dd).
    trend_share: fraction of equity across the 5 trend sleeves (rest to MR).
    """
    eq_sum, dates, all_trades = None, None, []
    per_sleeve: dict[str, float] = {}

    # --- TREND sleeves ---
    sub_trend = EQUITY * trend_share / len(TREND_SYMS)
    for sym in TREND_SYMS:
        df = panels_4h[sym]
        target, stop = build_trend_signals(df, DirectionalRunnerConfig(symbol=f"{sym}/USDT"))
        if trend_gate == "hmm":
            gated = np.where(routers[sym].turtle_gate, target, 0).astype(int)
        else:  # volkill: only the validated vol-spike kill vetoes entries
            pk = np.roll(routers[sym].p_spike, 1)
            pk[0] = np.nan
            spike_halt = np.where(np.isfinite(pk), pk > ROUTER_CFG.vol_kill_p, False)
            gated = np.where(spike_halt, 0, target).astype(int)
        bt = EventBacktester(
            EngineConfig(equity=sub_trend, cost=cost, risk_per_trade=risk,
                         max_leverage=2.0, allow_short=True)
        )
        trades, eq = bt.run(df, gated, stop=stop)
        if len(trades):
            trades["symbol"] = f"trend:{sym}"
            all_trades.append(trades)
        per_sleeve[f"trend:{sym}"] = float(eq[-1] - sub_trend)
        eq_sum = eq if eq_sum is None else eq_sum + eq
        dates = df["date"].to_numpy()

    # --- MR sleeves (gated by pairs_gate of leg A) ---
    sub_mr = EQUITY * (1.0 - trend_share) / len(MR_RATIOS)
    for a, b in MR_RATIOS:
        m = log_ratio_series(panels_4h[a], panels_4h[b])
        ou_arr = _ou_validity(m, MR_PARAMS)
        res = generate_signals(m, MR_PARAMS, ou_valid_arr=ou_arr)
        gate = routers[a].pairs_gate[: len(m)]
        sim = PairsSimulator(
            PairsSimParams(equity=sub_mr, capital_fraction=0.25 * (risk / 0.01), cost=mr_cost)
        )
        trades, eq = sim.run(m, res.target, entry_gate=gate)
        eq = pd.Series(eq).ffill().bfill().fillna(sub_mr).to_numpy(dtype=float)
        if len(trades):
            trades["symbol"] = f"mr:{a}/{b}"
            all_trades.append(trades)
        per_sleeve[f"mr:{a}/{b}"] = float(eq[-1] - sub_mr)
        # align lengths defensively (all sleeves share the 4h grid)
        if len(eq) != len(eq_sum):
            eq = np.resize(eq, len(eq_sum))
        eq_sum = eq_sum + eq

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(trades):
        trades = trades.sort_values("exit_time").reset_index(drop=True)
    eq = eq_sum
    peak = np.maximum.accumulate(eq)
    span_days = max(
        (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).total_seconds() / 86400.0, 1.0
    )
    growth = max(float(eq[-1] / EQUITY), 1e-9)
    return DirectionalResult(
        name="router_book", trades=trades, equity=eq, dates=dates,
        per_coin_net=per_sleeve, net_pnl=float(eq[-1] - EQUITY),
        annualised_return=float(growth ** (TRADING_DAYS_PER_YEAR / span_days) - 1.0),
        max_drawdown=float(((peak - eq) / peak).max()), n_trades=len(trades),
    )


def main() -> int:
    panels_1h = {
        s: pd.read_feather(DATA / f"{s}_USDT-1h-perp.feather") for s in TREND_SYMS
    }
    panels_4h = {s: resample_panel(df, "4h") for s, df in panels_1h.items()}

# Pre-registered profit-maximization grid (2026-09-14): portfolio-construction
# levers ONLY - gate strictness, sleeve weights, risk budget. No alpha
# parameters touched. Full grid reported, pass or fail.
GRID = (
    ("A_baseline",        dict(trend_gate="hmm",     risk=0.01, trend_share=5 / 9)),
    ("B_trend_heavy",     dict(trend_gate="hmm",     risk=0.01, trend_share=0.75)),
    ("C_volkill_trend",   dict(trend_gate="volkill", risk=0.01, trend_share=0.75)),
    ("D_volkill_risk2x",  dict(trend_gate="volkill", risk=0.02, trend_share=0.75)),
)


def main() -> int:
    panels_1h = {
        s: pd.read_feather(DATA / f"{s}_USDT-1h-perp.feather") for s in TREND_SYMS
    }
    panels_4h = {s: resample_panel(df, "4h") for s, df in panels_1h.items()}

    print("fitting HMM routers (5 coins, 4h)...")
    routers = {s: filtered_hmm_router(panels_4h[s], ROUTER_CFG, seed=42) for s in TREND_SYMS}

    lines = [
        "# PDF Signal Router Book — Profit-Maximization Grid",
        "",
        "TREND sleeve: slow trend (5 perps). MR sleeve: OU pairs under pairs_gate",
        "(ETH/BTC, SOL/BTC, BNB/BTC, XRP/BTC, maker). Causal HMM router per coin.",
        "Grid levers: trend gate (hmm hard vs volkill-only), trend equity share,",
        "risk per trade. Standard 5-gate bundle per config; full grid reported.",
        "",
        "## Summary",
        "",
        "| config | verdict | net pnl | ann. return | max dd | trades | worst fold |",
        "|---|---|---|---|---|---|---|",
    ]
    sections: list[str] = []
    best = None
    for name, kw in GRID:
        print(f"running {name} {kw}...")
        result = run_book(COST, MR_COST, panels_4h, routers, **kw)
        folds = evaluate_directional_folds(result, n_folds=6)
        stress = {
            m: run_book(_stressed(COST, m), _stressed(MR_COST, m), panels_4h, routers, **kw)
            for m in (1.5, 2.0)
        }
        gates = acceptance_gates(result, folds, stress)
        verdict = "PASS" if all(g.passed for g in gates) else "FAIL"
        worst = folds["net_pnl"].min() if len(folds) else float("nan")
        lines.append(
            f"| {name} | **{verdict}** | {result.net_pnl:,.0f} | {result.annualised_return:.1%} | "
            f"{result.max_drawdown:.1%} | {result.n_trades} | {worst:,.0f} |"
        )
        sections.append(f"## {name} — verdict: **{verdict}**")
        sections.append("")
        sections.append("| gate | result | evidence |")
        sections.append("|---|---|---|")
        for g in gates:
            sections.append(f"| {g.gate_id} {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.evidence} |")
        sections.append("")
        if len(folds):
            sections += ["| fold | window | net pnl |", "|---|---|---|"]
            for f in folds.itertuples():
                sections.append(f"| {f.fold} | {f.start} → {f.end} | {f.net_pnl:,.0f} |")
            sections.append("")
        sections += ["| sleeve | net pnl |", "|---|---|"]
        for k, v in sorted(result.per_coin_net.items()):
            sections.append(f"| {k} | {v:,.0f} |")
        sections.append("")
        print(f"  {verdict}: net={result.net_pnl:,.0f} dd={result.max_drawdown:.1%} worst_fold={worst:,.0f}")
        if best is None or (result.net_pnl > best[1].net_pnl and verdict == "PASS") or (
            best[2] != "PASS" and result.net_pnl > best[1].net_pnl
        ):
            if verdict == "PASS" or best is None:
                best = (name, result, verdict)

    if best:
        lines.append("")
        lines.append(f"**Recommended config: {best[0]} ({best[2]}, net {best[1].net_pnl:,.0f})**")
    REPORT.write_text("\n".join(lines + [""] + sections) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0 if best and best[2] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
