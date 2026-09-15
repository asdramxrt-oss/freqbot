"""Run the full-architecture (design-PDF) directional bot on FUTURES data + gates.

The PDF pipeline (features -> Turtle -> causal HMM router -> triple barrier ->
meta model -> bet sizer -> risk engine) is built per coin on the 5-coin USDT-M
perp panel ONLY - no spot series is read for training or backtesting. Signals
are built once per variant and cached under research/artifacts/ (the HMM router
is the expensive stage); the acceptance bundle (walk-forward folds, 1.5x/2x cost
stress, concentration, drawdown) then runs on the frozen signals. Every variant
is reported, pass or fail (multiple-testing honesty).

Variants: the PDF stack at its canonical 1h timebase, and at 4h - the timeframe
where the simple slow-trend grid found its one passing variant
(ts_trend_4h_d60_sma200_ls) - with both the canonical turtle entry (20) and the
slow entry (60). Router/HMM constants scale with the bar size.

Usage:  .venv/Scripts/python research/run_pdf_pipeline_acceptance.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.bot.config import BotConfig  # noqa: E402
from research_lib.risk.costs import CostModel  # noqa: E402
from research_lib.strategies.directional import resample_panel  # noqa: E402
from research_lib.strategies.pdf_pipeline import (  # noqa: E402
    PDFPanelSignals,
    build_pdf_panel_signals,
    run_panel_pdf,
)
from research_lib.strategies.unified_router import RouterConfig  # noqa: E402
from research_lib.validation.directional_acceptance import (  # noqa: E402
    DirectionalAcceptance,
    acceptance_gates,
    evaluate_directional_folds,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data" / "futures"
ARTIFACTS = ROOT / "research" / "artifacts"
REPORT = ROOT / "research" / "reports" / "PDF_PIPELINE_ACCEPTANCE_REPORT.md"

SYMBOLS = ("BTC", "ETH", "BNB", "SOL", "XRP")
# Binance USDT-M perp fee schedule (VIP0) + realistic spread/slippage
COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)

# (variant name, bar rule, turtle entry_n). 1h result is cached from the first run.
VARIANTS = (
    ("pdf_1h_e20", "1h", 20),
    ("pdf_4h_e20", "4h", 20),
    ("pdf_4h_e60", "4h", 60),
)

# v2 enhancements (pre-registered 2026-09-14 BEFORE seeing results):
#   panelmeta - hard router gate kept, meta model pooled across coins (coin one-hot)
#   regfeat   - regime as FEATURE not gate (vol-spike kill kept), pooled panel meta
V2_VARIANTS = (
    ("pdf_v2_1h_e20_panelmeta", "1h", 20, "panelmeta"),
    ("pdf_v2_1h_e20_regfeat", "1h", 20, "regfeat"),
)

# v3 (pre-registered 2026-09-14 after v2 results): the PDF stack wraps the primary
# that HAS edge - the validated slow trend (4h d60/30 + SMA1200 long/short, 2xATR
# stop). Question: can triple-barrier meta-labeling improve a profitable primary?
V3_VARIANTS = (
    ("pdf_v3_4h_slowtrend_panelmeta", "4h", "panelmeta"),
    ("pdf_v3_4h_slowtrend_regfeat", "4h", "regfeat"),
)

# v4 (pre-registered 2026-09-14 after the ms/ns timestamp-alignment BUG was found
# and fixed in the meta feature pipeline): v3 + the meta battery's winning B3
# config - horizon-matched barriers (pt=6/sl=2/vert=96, the strategy's real
# geometry) + expanded features (fracdiff_close, vol_yz_20, funding level/30d
# mean). B3 purged-CV: rf 0.589 / xgb 0.552 mean AUC (unstable: 2 folds < 0.50).
V4_VARIANTS = (
    ("pdf_v4_4h_slowtrend_regfeat", "4h", "regfeat"),
)


def router_cfg_for(rule: str) -> RouterConfig:
    """15m-tuned router constants scaled to the bar size (same wall-clock windows)."""
    per_day = {"1h": 24, "4h": 6}[rule]
    return RouterConfig(
        refit_bars=28 * per_day,
        min_train_bars=max(42 * per_day, 1000),
        vol_kill_p=0.80,
        vol_refit_bars=56 * per_day,
        train_window=333 * per_day,
    )


def _stressed(cost: CostModel, mult: float) -> CostModel:
    return CostModel(
        maker_fee=cost.maker_fee * mult,
        taker_fee=cost.taker_fee * mult,
        spread_bps=cost.spread_bps * mult,
        slippage_bps=cost.slippage_bps * mult,
    )


def build_signals_cached(
    name: str, panels: dict[str, pd.DataFrame], rule: str, entry_n: int
) -> PDFPanelSignals:
    cache = ARTIFACTS / f"pdf_signals_{name}.pkl"
    if cache.exists():
        print(f"  loaded cached signals for {name}")
        return pickle.loads(cache.read_bytes())
    cfg = BotConfig()
    cfg.data.timeframe = rule
    cfg.data.use_hurst = True              # PDF feature set includes Hurst/DFA
    cfg.strategy.turtle_allow_short = True
    cfg.strategy.turtle_entry_n = entry_n

    signals = build_pdf_panel_signals(panels, cfg, router_cfg=router_cfg_for(rule))
    cache.write_bytes(pickle.dumps(signals))
    return signals


def build_signals_cached_v2(
    name: str, panels: dict[str, pd.DataFrame], rule: str, entry_n: int, mode: str,
    primary: str = "turtle", b3: bool = False,
) -> PDFPanelSignals:
    from research_lib.strategies.pdf_pipeline import build_pdf_panel_signals_v2

    cache = ARTIFACTS / f"pdf_signals_{name}.pkl"
    if cache.exists():
        print(f"  loaded cached signals for {name}")
        return pickle.loads(cache.read_bytes())
    cfg = BotConfig()
    cfg.data.timeframe = rule
    cfg.data.use_hurst = True
    cfg.strategy.turtle_allow_short = True
    cfg.strategy.turtle_entry_n = entry_n
    kwargs = {}
    if b3:
        from research_lib.strategies.carry import load_funding

        funding_map = {}
        for sym in SYMBOLS:
            f = ROOT / "research" / "data" / "funding" / f"{sym}_USDT_USDT_funding.csv"
            if f.exists():
                fd = load_funding(f).copy()
                fd["date_ns"] = (
                    (pd.to_datetime(fd["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
                    // pd.Timedelta(1, "ns")
                )
                fd["f30"] = fd["funding_rate"].rolling(90, min_periods=30).mean()
                funding_map[sym] = fd
        kwargs = dict(meta_pt=6.0, meta_sl=2.0, meta_vert=96,
                      meta_expanded=True, funding_map=funding_map)
    signals = build_pdf_panel_signals_v2(
        panels, cfg, router_cfg=router_cfg_for(rule), mode=mode, primary=primary, **kwargs
    )
    cache.write_bytes(pickle.dumps(signals))
    return signals


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    panels_1h = {
        sym: pd.read_feather(DATA / f"{sym}_USDT-1h-perp.feather") for sym in SYMBOLS
    }

    lines = [
        "# PDF-Architecture Futures Directional Bot — Acceptance Report",
        "",
        "Full design-PDF pipeline per coin on the 5-coin USDT-M perp panel:",
        "feature engine (YZ vol, ATR, RSI, fracdiff, Hurst) -> Turtle breakout",
        "(long+short) -> causal filtered-HMM regime router + vol-spike kill ->",
        "triple barrier -> meta model (AUC-gated) -> bet sizer -> risk engine",
        "(stop-based sizing, 2x leverage cap, taker costs). NO spot data anywhere.",
        "Router constants scaled from their 15m tuning to each bar size; HMM",
        "estimation uses a rolling ~333-day window. Gates identical to the",
        "directional/carry bundles. Every variant is reported, pass or fail.",
        "",
        "## Summary",
        "",
        "| variant | verdict | net pnl | ann. return | max dd | trades |",
        "|---|---|---|---|---|---|",
    ]
    sections: list[str] = []
    any_pass = False

    all_variants = (
        [(n, r, e, None, "turtle", False) for n, r, e in VARIANTS]
        + [(n, r, e, m, "turtle", False) for n, r, e, m in V2_VARIANTS]
        + [(n, r, 60, m, "slowtrend", False) for n, r, m in V3_VARIANTS]
        + [(n, r, 60, m, "slowtrend", True) for n, r, m in V4_VARIANTS]
    )
    for name, rule, entry_n, mode, primary, b3 in all_variants:
        panels = (
            panels_1h if rule == "1h"
            else {s: resample_panel(df, rule) for s, df in panels_1h.items()}
        )
        print(f"building signals: {name} ({rule}, turtle entry {entry_n}"
              + (f", mode={mode}, primary={primary}" if mode else "")
              + (", B3 meta" if b3 else "") + ")")
        if mode is None:
            signals = build_signals_cached(name, panels, rule, entry_n)
        else:
            signals = build_signals_cached_v2(name, panels, rule, entry_n, mode, primary, b3)
        for note in signals.notes:
            print(" ", note)

        result = run_panel_pdf(name, panels, signals, COST)
        folds = evaluate_directional_folds(result, n_folds=6)
        stress = {
            m: run_panel_pdf(name, panels, signals, _stressed(COST, m))
            for m in (1.5, 2.0)
        }
        gates = acceptance_gates(result, folds, stress)
        verdict = "PASS" if all(g.passed for g in gates) else "FAIL"
        acc = DirectionalAcceptance(
            name=name, folds=folds, gates=gates, verdict=verdict, result=result
        )

        lines.append(
            f"| {name} | **{acc.verdict}** | {result.net_pnl:,.0f} | "
            f"{result.annualised_return:.1%} | {result.max_drawdown:.1%} | {result.n_trades} |"
        )
        sections.append(f"## {name} — verdict: **{acc.verdict}**")
        sections.append("")
        sections.append("pipeline stages:")
        sections += [f"- {n}" for n in signals.notes]
        sections.append("")
        sections.append("| gate | result | evidence |")
        sections.append("|---|---|---|")
        for g in acc.gates:
            sections.append(
                f"| {g.gate_id} {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.evidence} |"
            )
        sections.append("")
        if len(acc.folds):
            sections += ["| fold | window | net pnl |", "|---|---|---|"]
            for f in acc.folds.itertuples():
                sections.append(f"| {f.fold} | {f.start} → {f.end} | {f.net_pnl:,.0f} |")
            sections.append("")
        sections += ["| coin | net pnl |", "|---|---|"]
        for sym, v in sorted(result.per_coin_net.items()):
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
