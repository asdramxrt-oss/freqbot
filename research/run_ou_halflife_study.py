"""OU structural study (2026-09-14): is mean-reversion even PRESENT in perp spreads?

The OU book failed on absolute price and on spreads (15m spot taker; 1h perp
taker/maker/vol-kill - OU_SPREAD_ACCEPTANCE_REPORT.md). Before closing it for
good, answer the structural question directly rather than via trading PnL:

  1. Across 4 estimation horizons (3d/7d/14d/30d of 1h bars), how OFTEN does a
     perp spread admit a valid OU fit with a half-life SHORT enough to trade
     (< horizon/4)?
  2. Is the half-life STABLE (does a valid fit in one block predict MR gross
     profit in the NEXT block)? If validity carries no forward information,
     the ou_gate itself is noise and no window choice can rescue it.

Ratios: ETH/BTC, SOL/BTC, BNB/BTC on refreshed 1h perp data. Fully causal.
Output: research/reports/OU_HALFLIFE_STUDY.md

Usage:  .venv/Scripts/python research/run_ou_halflife_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.strategies.ou import OUParams, fit_ou  # noqa: E402
from research_lib.strategies.pairs import log_ratio_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data" / "futures"
REPORT = ROOT / "research" / "reports" / "OU_HALFLIFE_STUDY.md"

RATIOS = [("ETH", "BTC"), ("SOL", "BTC"), ("BNB", "BTC")]
HORIZONS = {"3d": 72, "7d": 168, "14d": 336, "30d": 720}


def block_stats(spread: pd.Series, win: int) -> pd.DataFrame:
    """Per non-overlapping block: OU validity, half-life, and NEXT block's MR gross pnl.

    MR gross pnl of the canonical z=2-in / z=0-out trade on the block, simulated
    without costs (structure test, not a strategy)."""
    n = len(spread)
    rows = []
    s = spread.to_numpy(dtype=float)
    for a in range(0, n - 2 * win, win):
        seg = pd.Series(s[a : a + win])
        mdl = fit_ou(seg.reset_index(drop=True),
                     OUParams(min_sample=min(120, win // 2), rolling_window=min(600, win // 2)))
        nxt = s[a + win : a + 2 * win]
        mu, sd = seg.mean(), seg.std()
        pnl, pos = 0.0, 0
        if sd > 0 and np.isfinite(sd):
            entry = 0.0
            for x in nxt:
                z = (x - mu) / sd
                if pos == 0:
                    if z <= -2.0:
                        pos, entry = 1, x
                    elif z >= 2.0:
                        pos, entry = -1, x
                elif pos == 1 and z >= 0.0:
                    pnl += (x - entry) / entry
                    pos = 0
                elif pos == -1 and z <= 0.0:
                    pnl += (entry - x) / entry
                    pos = 0
        rows.append(
            {"block_start": a, "valid": bool(mdl.valid),
             "half_life": float(mdl.half_life) if mdl.valid else np.nan,
             "next_mr_gross": pnl}
        )
    return pd.DataFrame(rows)


def main() -> int:
    perp = {
        s: pd.read_feather(DATA / f"{s}_USDT-1h-perp.feather")
        for s in ("BTC", "ETH", "BNB", "SOL")
    }
    lines = [
        "# OU Structural Study — is mean-reversion present in perp spreads?",
        "",
        "Rolling OU fits per non-overlapping block; validity + half-life + the",
        "NEXT block's gross mean-reversion pnl (z=2 entry, z=0 exit, no costs).",
        "If OU validity does not predict next-block MR profit, the gate is noise",
        "and no window choice can rescue the OU book.",
        "",
    ]
    summary_rows = []
    for a, b in RATIOS:
        m = log_ratio_series(perp[a], perp[b])
        lines.append(f"## {a}/{b}")
        lines.append("")
        lines.append("| horizon | blocks | OU valid % | median half-life (bars) | half-life < h/4 % | MR gross after VALID | MR gross after INVALID | information? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for hname, win in HORIZONS.items():
            st = block_stats(m["spread"], win)
            if not len(st):
                continue
            valid = st[st["valid"]]
            hl_short = (valid["half_life"] < win / 4).mean() if len(valid) else 0.0
            g_v = valid["next_mr_gross"].mean() if len(valid) else float("nan")
            g_i = st.loc[~st["valid"], "next_mr_gross"].mean()
            info = "none"
            if len(valid) >= 10 and np.isfinite(g_v) and np.isfinite(g_i):
                info = f"{'YES' if g_v > g_i else 'no'} ({g_v - g_i:+.4f}/block)"
            summary_rows.append(
                (f"{a}/{b}", hname, len(st), st["valid"].mean(),
                 valid["half_life"].median() if len(valid) else float("nan"),
                 hl_short, g_v, g_i, info)
            )
            lines.append(
                f"| {hname} | {len(st)} | {st['valid'].mean():.0%} | "
                f"{valid['half_life'].median() if len(valid) else float('nan'):.0f} | "
                f"{hl_short:.0%} | {g_v:+.4f} | {g_i:+.4f} | {info} |"
            )
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    for r in summary_rows:
        print(f"{r[0]:9s} {r[1]:>3s} valid={r[3]:5.0%} hl_med={r[4]:6.0f} short_hl={r[5]:5.0%} "
              f"gross_valid={r[6]:+.4f} gross_invalid={r[7]:+.4f} {r[8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
