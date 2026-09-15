"""FINAL pre-registered OU-fix attempt (2026-09-14): OU mean-reversion on PERP SPREADS.

Context (do not re-litigate): OU on absolute price failed with high statistical power
(mirror symmetry, parameter-flat failure), and the blueprint's prescribed fix - OU on
spreads - ALSO failed when tested at 15m on spot with TAKER costs
(research/run_pairs.py: no configuration reached 5/6 positive-gross windows).

Three levers from that failure were never tested together, and each is theory-backed:
  1. PERP data (both legs shortable natively; no spot borrow assumptions)
  2. MAKER execution (post-only: mean-reversion is cost-dominated; the assessment
     showed maker cuts costs ~70% and "turns marginal strategies from negative to
     break-even")
  3. VOL-SPIKE KILL (the one PDF component proven to add value - 2026-09-14 v3:
     dd 12.9% -> 9.0%; mean-reversion dies exactly during vol expansions)

This script tests those three levers cumulatively on 1h perp ratios
(ETH/BTC, SOL/BTC, BNB/BTC) with the causal OU-tradability gate ON, through the
standard 5-gate acceptance bundle. Every variant is reported, pass or fail.
If nothing passes, the OU book is closed with the highest confidence yet.

Usage:  .venv/Scripts/python research/run_ou_spread_acceptance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.risk.costs import CostModel  # noqa: E402
from research_lib.strategies.directional import (  # noqa: E402
    DirectionalResult,
    TRADING_DAYS_PER_YEAR,
)
from research_lib.strategies.pairs import (  # noqa: E402
    PairsParams,
    generate_signals,
    log_ratio_series,
)
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator  # noqa: E402
from research_lib.validation.directional_acceptance import (  # noqa: E402
    acceptance_gates,
    evaluate_directional_folds,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data" / "futures"
REPORT = ROOT / "research" / "reports" / "OU_SPREAD_ACCEPTANCE_REPORT.md"

RATIOS = [("ETH", "BTC"), ("SOL", "BTC"), ("BNB", "BTC")]
EQUITY = 100_000.0

# Binance USDT-M perp VIP0
TAKER = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)
# post-only maker: fee only; resting fills earn the quote, no spread/slippage cost
MAKER = CostModel(maker_fee=0.0002, taker_fee=0.0002, spread_bps=0.0, slippage_bps=0.0)

PARAMS = PairsParams(
    est_window_bars=168,      # 7 days of 1h bars (was 672 at 15m)
    z_entry=2.0,
    z_exit=0.0,
    ou_gate=True,
    ou_min_sample=120,
    max_hold_bars=336,        # 2x estimation window
)

VARIANTS = (
    ("ou_spread_1h_taker", TAKER, False),
    ("ou_spread_1h_maker", MAKER, False),
    ("ou_spread_1h_maker_volkill", MAKER, True),
)


def _ou_validity(m: pd.DataFrame, params: PairsParams) -> np.ndarray:
    """Causal OU-tradability flags, block-refit on trailing windows (as run_pairs.py)."""
    from research_lib.strategies.ou import OUParams, fit_ou

    arr = np.zeros(len(m), dtype=bool)
    step, win = 168, params.est_window_bars
    for t in range(win, len(m), step):
        seg = m["spread"].iloc[t - win : t]
        mdl = fit_ou(
            seg.reset_index(drop=True),
            OUParams(min_sample=params.ou_min_sample, rolling_window=min(600, win // 2)),
        )
        arr[t : t + step] = mdl.valid
    return arr


def _vol_kill_gate(m: pd.DataFrame) -> np.ndarray:
    """Causal ratio-vol kill: gate[t] False when the spread's realized vol (24-bar)
    exceeds its trailing 90-day 90th percentile, decided with data <= t-1 only."""
    rv = m["spread"].diff().rolling(24).std()
    thr = rv.rolling(2160, min_periods=720).quantile(0.9)
    ok = (rv <= thr).to_numpy()
    gate = np.roll(ok, 1)
    gate[0] = False
    return np.where(np.isfinite(np.roll(rv, 1)), gate, False)


def _stressed(cost: CostModel, mult: float) -> CostModel:
    return CostModel(
        maker_fee=cost.maker_fee * mult,
        taker_fee=cost.taker_fee * mult,
        spread_bps=cost.spread_bps * mult,
        slippage_bps=cost.slippage_bps * mult,
    )


def run_variant(name: str, frames: dict[str, pd.DataFrame], cost: CostModel, vol_kill: bool) -> DirectionalResult:
    sub_equity = EQUITY / len(frames)
    eq_sum, dates, all_trades = None, None, []
    per_ratio_net: dict[str, float] = {}
    for ratio, m in sorted(frames.items()):
        ou_arr = _ou_validity(m, PARAMS)
        res = generate_signals(m, PARAMS, ou_valid_arr=ou_arr)
        gate = _vol_kill_gate(m) if vol_kill else None
        sim = PairsSimulator(
            PairsSimParams(equity=sub_equity, capital_fraction=0.25, cost=cost)
        )
        trades, eq = sim.run(m, res.target, entry_gate=gate)
        # the simulator leaves eq_curve[0] NaN (loop starts at t=1) - pin it to the
        # initial equity so peak/drawdown/folds don't propagate NaN
        eq = pd.Series(eq).ffill().bfill().fillna(sub_equity).to_numpy(dtype=float)
        if len(trades):
            trades["symbol"] = ratio
            all_trades.append(trades)
        per_ratio_net[ratio] = float(eq[-1] - sub_equity)
        eq_sum = eq if eq_sum is None else eq_sum + eq
        dates = m["date"].to_numpy()

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(trades) and "exit_time" in trades.columns:
        trades = trades.sort_values("exit_time").reset_index(drop=True)
    eq = eq_sum if eq_sum is not None else np.full(2, EQUITY)
    peak = np.maximum.accumulate(eq)
    span_days = max(
        (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).total_seconds() / 86400.0, 1.0
    )
    growth = max(float(eq[-1] / EQUITY), 1e-9)
    return DirectionalResult(
        name=name,
        trades=trades,
        equity=eq,
        dates=dates,
        per_coin_net=per_ratio_net,
        net_pnl=float(eq[-1] - EQUITY),
        annualised_return=float(growth ** (TRADING_DAYS_PER_YEAR / span_days) - 1.0),
        max_drawdown=float(((peak - eq) / peak).max()),
        n_trades=len(trades),
        cost_role="maker" if cost.taker_fee == cost.maker_fee else "taker",
    )


def main() -> int:
    perp = {
        sym: pd.read_feather(DATA / f"{sym}_USDT-1h-perp.feather")
        for sym in ("BTC", "ETH", "BNB", "SOL")
    }
    frames = {
        f"{a}/{b}": log_ratio_series(perp[a], perp[b]) for a, b in RATIOS
    }

    lines = [
        "# OU-on-Perp-Spreads — FINAL Acceptance Report",
        "",
        "The last untested levers for the OU book, cumulative: perp data -> maker",
        "execution -> vol-spike kill. 1h perp ratios ETH/BTC, SOL/BTC, BNB/BTC;",
        "causal z(168) entries at |z|>=2, exit at z-cross-0 or 336-bar time stop;",
        "causal OU half-life tradability gate ON; dollar-neutral 25%/leg.",
        "Standard 5-gate bundle. Every variant reported, pass or fail.",
        "",
        "## Summary",
        "",
        "| variant | verdict | net pnl | ann. return | max dd | trades |",
        "|---|---|---|---|---|---|",
    ]
    sections: list[str] = []
    any_pass = False

    for name, cost, vol_kill in VARIANTS:
        print(f"running {name} (vol_kill={vol_kill})...")
        result = run_variant(name, frames, cost, vol_kill)
        folds = evaluate_directional_folds(result, n_folds=6)
        stress = {
            m: run_variant(name, frames, _stressed(cost, m), vol_kill)
            for m in (1.5, 2.0)
        }
        gates = acceptance_gates(result, folds, stress)
        verdict = "PASS" if all(g.passed for g in gates) else "FAIL"
        lines.append(
            f"| {name} | **{verdict}** | {result.net_pnl:,.0f} | "
            f"{result.annualised_return:.1%} | {result.max_drawdown:.1%} | {result.n_trades} |"
        )
        sections.append(f"## {name} — verdict: **{verdict}**")
        sections.append("")
        sections.append("| gate | result | evidence |")
        sections.append("|---|---|---|")
        for g in gates:
            sections.append(
                f"| {g.gate_id} {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.evidence} |"
            )
        sections.append("")
        if len(folds):
            sections += ["| fold | window | net pnl |", "|---|---|---|"]
            for f in folds.itertuples():
                sections.append(f"| {f.fold} | {f.start} → {f.end} | {f.net_pnl:,.0f} |")
            sections.append("")
        sections += ["| ratio | net pnl |", "|---|---|"]
        for sym, v in sorted(result.per_coin_net.items()):
            sections.append(f"| {sym} | {v:,.0f} |")
        sections.append("")
        print(f"  {verdict}: net={result.net_pnl:,.0f} dd={result.max_drawdown:.1%} trades={result.n_trades}")
        if verdict == "PASS":
            any_pass = True

    REPORT.write_text("\n".join(lines + [""] + sections), encoding="utf-8")
    print(f"wrote {REPORT}")
    print("PASS variants:", "at least one" if any_pass else "NONE")
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
