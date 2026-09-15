"""Vol-spike-kill ablation + funding-payment realism for the slow-trend perp book.

Pre-registered 2026-09-14 (after v3 passed 4/5 gates):

  ABLATION  - `pdf_v3_4h_slowtrend_regfeat` ran the slow trend with the vol-spike
    kill and beat the bare variant, but at a DIFFERENT cost model than the
    directional acceptance report. Here both arms run at ONE cost model through
    one engine, so the kill's contribution is isolated cleanly.

  FUNDING   - the directional harness has always ignored perp funding. The slow
    trend holds positions for WEEKS; at 3 prints/day that is hundreds of funding
    cashflows per trade. Longs pay when rate > 0, shorts receive. This overlay
    applies the real funding series to every open position and re-runs the
    5-gate bundle on funded equity. It is a measurement fix, not a strategy
    change - and it can move fold nets in either direction.

Grid: {plain, volkill} x {no funding, funding}. Every arm reported.
Usage:  .venv/Scripts/python research/run_trend_funding_ablation.py
"""

from __future__ import annotations

import pickle
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
from research_lib.strategies.carry import load_funding  # noqa: E402
from research_lib.strategies.directional import resample_panel  # noqa: E402
from research_lib.validation.directional_acceptance import (  # noqa: E402
    acceptance_gates,
    evaluate_directional_folds,
)
from research_lib.strategies.directional import DirectionalResult, TRADING_DAYS_PER_YEAR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data"
REPORT = ROOT / "research" / "reports" / "TREND_FUNDING_ABLATION_REPORT.md"
SYMBOLS = ("BTC", "ETH", "BNB", "SOL", "XRP")
EQUITY = 100_000.0
COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def _stressed(cost: CostModel, mult: float) -> CostModel:
    return CostModel(
        maker_fee=cost.maker_fee * mult,
        taker_fee=cost.taker_fee * mult,
        spread_bps=cost.spread_bps * mult,
        slippage_bps=cost.slippage_bps * mult,
    )


def _per_coin_run(df4, target, stop, cost, sub_equity):
    bt = EventBacktester(
        EngineConfig(
            equity=sub_equity, cost=cost, risk_per_trade=0.01,
            max_leverage=2.0, allow_short=True,
        )
    )
    trades, eq = bt.run(df4, target, stop=stop)
    return trades, eq


def _funding_overlay(trades, funding, dates, closes, eq):
    """Add real 8h funding cashflows to one coin's equity curve.

    cf = -signed_units * mark * rate at each print (longs pay positive rates).
    Positions are qty-locked at entry, so signed units per bar come straight
    from the trade list. Cashflow lands on the bar containing the print.
    """
    cf = np.zeros(len(eq))
    if trades is None or not len(trades) or funding is None or not len(funding):
        return eq + cf
    signed = np.zeros(len(eq))
    d64 = dates.astype("datetime64[ns]")
    for tr in trades.to_dict("records"):
        a = int(np.searchsorted(d64, np.datetime64(pd.Timestamp(tr["entry_time"]).asm8)))
        b = int(np.searchsorted(d64, np.datetime64(pd.Timestamp(tr["exit_time"]).asm8)))
        b = min(max(b, a + 1), len(eq))
        signed[a:b] = int(tr["side"]) * float(tr["units"])
    f_dates = funding["date"].to_numpy(dtype="datetime64[ns]")
    f_rates = funding["funding_rate"].to_numpy(float)
    idx = np.searchsorted(d64, f_dates, side="right") - 1
    ok = idx >= 0
    for k, r in zip(idx[ok], f_rates[ok], strict=True):
        if signed[k] != 0.0:
            cf[k] += -signed[k] * float(closes[k]) * r
    return eq + np.cumsum(cf)


def run_arm(name, panels_4h, signals, funding_map, cost, use_funding):
    sub_equity = EQUITY / len(panels_4h)
    eq_sum, dates, all_trades = None, None, []
    per_coin_net: dict[str, float] = {}
    total_funding = 0.0
    for sym in sorted(panels_4h):
        df = panels_4h[sym]
        target, stop = signals[sym]
        trades, eq = _per_coin_run(df, target, stop, cost, sub_equity)
        if use_funding:
            eq_f = _funding_overlay(
                trades, funding_map.get(sym), df["date"].to_numpy(),
                df["close"].to_numpy(float), eq,
            )
            total_funding += float(eq_f[-1] - eq[-1])
            eq = eq_f
        if len(trades):
            trades["symbol"] = sym
            all_trades.append(trades)
        per_coin_net[sym] = float(eq[-1] - sub_equity)
        eq_sum = eq if eq_sum is None else eq_sum + eq
        dates = df["date"].to_numpy()

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(trades):
        trades = trades.sort_values("exit_time").reset_index(drop=True)
    eq = eq_sum
    peak = np.maximum.accumulate(eq)
    span_days = max(
        (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).total_seconds() / 86400.0, 1.0
    )
    growth = max(float(eq[-1] / EQUITY), 1e-9)
    res = DirectionalResult(
        name=name, trades=trades, equity=eq, dates=dates, per_coin_net=per_coin_net,
        net_pnl=float(eq[-1] - EQUITY),
        annualised_return=float(growth ** (TRADING_DAYS_PER_YEAR / span_days) - 1.0),
        max_drawdown=float(((peak - eq) / peak).max()), n_trades=len(trades),
    )
    res.funding_total = total_funding  # type: ignore[attr-defined]
    return res


def _funding_regime_gates(funding, dates):
    """Pre-registered funding-regime filter (2026-09-14, BEFORE seeing results).

    f30 = 30-day causal mean of the 8h funding rate; q10/q90 = trailing 365-day
    rolling 10th/90th percentile of f30 (min 180d history). Deeply negative
    funding = bear-positioning regime -> suppress LONGS; extremely positive =
    euphoria -> suppress SHORTS. Decided with data <= t-1 only. No price-derived
    parameters; funding is positioning information, orthogonal to the OHLCV panel.

    Returns (gate_long, gate_short) bool arrays aligned to the 4h bar dates.
    """
    n = len(dates)
    if funding is None or not len(funding):
        return np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    f = funding.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.tz_localize(None)
    f = f.set_index("date").sort_index()
    f30 = f["funding_rate"].rolling(90, min_periods=30).mean()          # 30d of 8h prints
    q10 = f30.rolling(1095, min_periods=540).quantile(0.10)             # 1y, min 180d
    q90 = f30.rolling(1095, min_periods=540).quantile(0.90)
    ok_long = ~(f30 < q10)
    ok_short = ~(f30 > q90)
    d64 = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    f_idx = np.searchsorted(f.index.to_numpy(dtype="datetime64[ns]"), d64, side="right") - 1
    gate_long = np.zeros(n, dtype=bool)
    gate_short = np.zeros(n, dtype=bool)
    gl = ok_long.to_numpy()
    gs = ok_short.to_numpy()
    for t in range(n):
        k = f_idx[t] - 1  # last print strictly before this bar's decision point
        if k >= 0:
            gate_long[t] = gl[k]
            gate_short[t] = gs[k]
    return gate_long, gate_short


def main() -> int:
    panels_1h = {
        s: pd.read_feather(DATA / "futures" / f"{s}_USDT-1h-perp.feather") for s in SYMBOLS
    }
    panels_4h = {s: resample_panel(df, "4h") for s, df in panels_1h.items()}
    funding_map = {}
    for s in SYMBOLS:
        f = DATA / "funding" / f"{s}_USDT_USDT_funding.csv"
        if f.exists():
            funding_map[s] = load_funding(f)

    # arm 1 "plain": slow trend, no kill (build_trend_signals = fill-convention target)
    plain = {}
    for s, df in panels_4h.items():
        plain[s] = build_trend_signals(df, DirectionalRunnerConfig(symbol=f"{s}/USDT"))
    # arm 2 "volkill": the cached v3 regfeat signals (vol-spike kill applied, meta OFF)
    cached = pickle.loads(
        (ROOT / "research" / "artifacts" / "pdf_signals_pdf_v3_4h_slowtrend_regfeat.pkl").read_bytes()
    )
    volkill = {
        s: (cached.per_coin[s].target, cached.per_coin[s].stop) for s in SYMBOLS
    }

    # arm 3 "fundfilter": volkill targets + the pre-registered funding-regime filter
    fundfilter = {}
    for s, df in panels_4h.items():
        target, stop = volkill[s]
        gate_long, gate_short = _funding_regime_gates(funding_map.get(s), df["date"].to_numpy())
        t = np.asarray(target, dtype=int).copy()
        t[(t == 1) & ~gate_long] = 0
        t[(t == -1) & ~gate_short] = 0
        fundfilter[s] = (t, stop)

    arms = (
        ("plain_nofunding", plain, False),
        ("plain_funding", plain, True),
        ("volkill_nofunding", volkill, False),
        ("volkill_funding", volkill, True),
        ("fundfilter_nofunding", fundfilter, False),
        ("fundfilter_funding", fundfilter, True),
    )

    lines = [
        "# Slow-Trend Perp Book — Vol-Spike-Kill Ablation + Funding Realism",
        "",
        "One engine, one cost model (Binance VIP0 taker), 5-coin 4h perp panel.",
        "`plain` = validated slow trend (d60/30 + SMA1200 ls, 2xATR stop).",
        "`volkill` = same + the PDF vol-spike kill (cached v3 regfeat signals).",
        "`funding` arms apply the real 8h funding series to every open position.",
        "",
        "## Summary",
        "",
        "| arm | verdict | net pnl | ann. return | max dd | trades | funding total |",
        "|---|---|---|---|---|---|---|",
    ]
    sections: list[str] = []
    for name, sigs, use_funding in arms:
        print(f"running {name}...")
        res = run_arm(name, panels_4h, sigs, funding_map, COST, use_funding)
        folds = evaluate_directional_folds(res, n_folds=6)
        stress = {
            m: run_arm(name, panels_4h, sigs, funding_map, _stressed(COST, m), use_funding)
            for m in (1.5, 2.0)
        }
        gates = acceptance_gates(res, folds, stress)
        verdict = "PASS" if all(g.passed for g in gates) else "FAIL"
        lines.append(
            f"| {name} | **{verdict}** | {res.net_pnl:,.0f} | {res.annualised_return:.1%} | "
            f"{res.max_drawdown:.1%} | {res.n_trades} | {res.funding_total:,.0f} |"
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
        sections += ["| coin | net pnl |", "|---|---|"]
        for sym, v in sorted(res.per_coin_net.items()):
            sections.append(f"| {sym} | {v:,.0f} |")
        sections.append("")
        print(f"  {verdict}: net={res.net_pnl:,.0f} dd={res.max_drawdown:.1%} "
              f"funding={res.funding_total:,.0f} worst_fold={folds.net_pnl.min():,.0f}")

    REPORT.write_text("\n".join(lines + [""] + sections), encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
