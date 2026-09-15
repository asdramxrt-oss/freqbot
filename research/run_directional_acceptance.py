"""Run the directional acceptance bundle over the pre-registered variant grid.

Implements the directional retest from the BOT_HANDOVER discipline: the old
directional books failed as short-horizon, long-only, single-coin, taker-cost
strategies. research_lib/strategies/directional.py re-tests the *directional
hypothesis* as slow trend + cross-sectional variants on the multi-coin perp
panel, with the project's standard execution model and mechanical gates.

Multiple-testing honesty: EVERY variant in the grid is run and reported, pass
or fail. Exits 0 only if at least one variant passes every gate.

Usage:  .venv/Scripts/python research/run_directional_acceptance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.risk.costs import CostModel  # noqa: E402
from research_lib.strategies.directional import build_variant_grid  # noqa: E402
from research_lib.validation.directional_acceptance import (  # noqa: E402
    run_directional_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data" / "futures"
REPORT = ROOT / "research" / "reports" / "DIRECTIONAL_ACCEPTANCE_REPORT.md"

SYMBOLS = ("BTC", "ETH", "BNB", "SOL", "XRP")
# Binance USDT-M perp fee schedule (VIP0) + realistic spread/slippage
COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def load_panels() -> dict[str, pd.DataFrame]:
    return {
        sym: pd.read_feather(DATA / f"{sym}_USDT-1h-perp.feather") for sym in SYMBOLS
    }


def main() -> int:
    panels = load_panels()
    grid = build_variant_grid(panels)

    lines = [
        "# Directional Acceptance Report",
        "",
        "Slow-trend and cross-sectional directional variants on the 5-coin USDT-M",
        "perp panel (research/data/futures/, 1h bars resampled to 4h/1d). Signal at",
        "close t, fill at open t+1, adverse taker cost; per-coin sub-accounts summed",
        "to one portfolio equity curve. Gates: data sufficiency, every walk-forward",
        "fold positive, 1.5x/2x cost stress, no single-coin concentration, drawdown",
        "budget. The full pre-registered grid is reported, pass or fail.",
        "",
        "## Summary",
        "",
        "| variant | verdict | net pnl | ann. return | max dd | trades |",
        "|---|---|---|---|---|---|",
    ]

    any_pass = False
    sections: list[str] = []
    for name, (vp_panels, vp_targets) in grid.items():
        acc = run_directional_acceptance(name, vp_panels, vp_targets, COST, n_folds=6)
        r = acc.result
        lines.append(
            f"| {name} | **{acc.verdict}** | {r.net_pnl:,.0f} | "
            f"{r.annualised_return:.1%} | {r.max_drawdown:.1%} | {r.n_trades} |"
        )
        sections.append(f"## {name} — verdict: **{acc.verdict}**")
        sections.append("")
        sections.append("| gate | result | evidence |")
        sections.append("|---|---|---|")
        for g in acc.gates:
            sections.append(
                f"| {g.gate_id} {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.evidence} |"
            )
        sections.append("")
        if len(acc.folds):
            sections.append("| fold | window | net pnl |")
            sections.append("|---|---|---|")
            for f in acc.folds.itertuples():
                sections.append(f"| {f.fold} | {f.start} → {f.end} | {f.net_pnl:,.0f} |")
            sections.append("")
        sections.append("| coin | net pnl |")
        sections.append("|---|---|")
        for sym, v in sorted(r.per_coin_net.items()):
            sections.append(f"| {sym} | {v:,.0f} |")
        sections.append("")
        if acc.verdict == "PASS":
            any_pass = True

    REPORT.write_text("\n".join(lines + [""] + sections), encoding="utf-8")
    print(f"wrote {REPORT}")
    print("PASS variants:", "at least one" if any_pass else "NONE")
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
