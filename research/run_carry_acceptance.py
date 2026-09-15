"""Run the funding-carry acceptance bundle for BTC and ETH and write a report.

Implements step 3 of research/FIX_OR_ENHANCE_ASSESSMENT.md ("run the walk-forward +
acceptance gates") against research/data only - the holdout directory is never touched.
Exits non-zero if any symbol fails a gate, so this can run in CI.

Usage:  .venv/Scripts/python research/run_carry_acceptance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.risk.costs import CostModel  # noqa: E402
from research_lib.strategies.carry import CarryConfig, load_funding  # noqa: E402
from research_lib.validation.carry_acceptance import run_carry_acceptance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data"
REPORT = ROOT / "research" / "reports" / "CARRY_ACCEPTANCE_REPORT.md"

SYMBOLS = ("BTC", "ETH", "BNB", "SOL", "XRP")
# Binance USDT-M perp fee schedule (VIP0) + realistic spread/slippage
COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def main() -> int:
    lines = [
        "# Funding-Carry Acceptance Report",
        "",
        "Delta-neutral long spot + short perp, funding settled at 8h boundaries,",
        "both legs costed, re-hedge before liquidation. Post-only (maker) execution.",
        "Short leg marked on the REAL perp series (research/data/futures/, real basis risk).",
        "Runs truncate where the shorter of spot/perp series ends (BTC/ETH/BNB/SOL spot",
        "feathers end 2025-06-03; XRP runs to 2026-07 funding end).",
        "",
    ]
    rc = 0
    for sym in SYMBOLS:
        funding = load_funding(DATA / "funding" / f"{sym}_USDT_USDT_funding.csv")
        spot = pd.read_feather(DATA / f"{sym}_USDT-1h.feather")
        perp_path = DATA / "futures" / f"{sym}_USDT-1h-perp.feather"
        perp = pd.read_feather(perp_path) if perp_path.exists() else None
        acc = run_carry_acceptance(f"{sym}/USDT", spot, funding, CarryConfig(post_only=True), COST, n_folds=6, perp=perp)
        lines.append(f"## {sym}/USDT — verdict: **{acc.verdict}**")
        lines.append("")
        lines.append("| gate | result | evidence |")
        lines.append("|---|---|---|")
        for g in acc.gates:
            lines.append(f"| {g.gate_id} {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.evidence} |")
        lines.append("")
        usable = acc.folds.dropna(subset=["net_pnl"])
        lines.append("| fold | window | prints | net pnl |")
        lines.append("|---|---|---|---|")
        for r in usable.itertuples():
            lines.append(f"| {r.fold} | {r.start[:10]} → {r.end[:10]} | {r.n_prints} | {r.net_pnl:,.0f} |")
        lines.append("")
        if acc.verdict != "PASS":
            rc = 1
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
