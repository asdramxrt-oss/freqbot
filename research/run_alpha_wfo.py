"""Walk-forward engine for the vol-spike alpha (DeepSeek brief Step 3-4).

Discipline:
  * The research window (2022-01..2025-06) is split into semi-annual test windows.
  * For EACH window the vol-spike XGB model is trained ONLY on data strictly before the
    window (expanding), with the spike threshold also computed on that training history.
  * The model emits P(spike) over the window; the strategy runs on those causal
    probabilities; per-window gross and net PnL are recorded.
  * Nothing from the window enters training; nothing from the final holdout is touched.

Gross = signal PnL at next-open fills with NO costs. Net = with taker fee + spread +
slippage (+ funding on the futures-style account). The brief's Step-4 bar is positive
gross in >=5/7 windows; the final gate requires net economics too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.constants import HOLDOUT_START
from research_lib.data.loader import load_pair_tf
from research_lib.ml.volspike import VolSpikeParams, build_vol_spike_dataset, causal_vol_features
from research_lib.strategies.volspike_alpha import VolSpikeAlphaParams, generate_alpha_signals
from research_lib.risk.costs import CostModel
from research_lib.risk.backtest_engine import EventBacktester, EngineConfig

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "alpha"
TF = "15m"
WINDOW_MONTHS = 6


def _save(obj, name):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


def _gross_pnl(df: pd.DataFrame, target: np.ndarray, opens: np.ndarray, closes: np.ndarray) -> float:
    """Signal PnL at next-open fills, no costs, one position at a time (gross)."""
    pnl = 0.0
    pos = 0
    entry_px = 0.0
    prev_desired = 0
    for t in range(1, len(df)):
        desired = int(target[t])
        if pos == 0 and desired != 0 and prev_desired == 0:
            pos = desired
            entry_px = opens[t]
        elif pos != 0 and desired == 0:
            pnl += (opens[t] - entry_px) * pos
            pos = 0
        elif pos != 0 and desired != 0 and desired != pos and prev_desired == 0:
            pnl += (opens[t] - entry_px) * pos
            pos = desired
            entry_px = opens[t]
        prev_desired = desired
    if pos != 0:
        pnl += (closes[-1] - entry_px) * pos
    return float(pnl)


def run_window_wfo(pair: str, direction: str, wf_blocks: list[tuple[pd.Timestamp, pd.Timestamp]],
                   seed: int = 0) -> dict:
    from xgboost import XGBClassifier

    df_all = load_pair_tf(pair, TF)
    df_all = df_all[df_all["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)
    params = VolSpikeParams(horizon_bars=96, percentile=0.90)
    feats_all = causal_vol_features(df_all, params)
    # realized vol on TRAIN history only is used for the spike threshold per window
    from research_lib.features.volatility import VolatilityEngine

    vol_all = VolatilityEngine(method="yang_zhang", window=20).compute(df_all)["volatility"]
    cols = [c for c in feats_all.columns if c != "date"]
    alpha_params = VolSpikeAlphaParams(direction=direction)

    window_rows = []
    for (ws, we) in wf_blocks:
        hist = df_all[df_all["date"] < ws]
        win_mask = (df_all["date"] >= ws) & (df_all["date"] < we)
        if len(hist) < 4000 or win_mask.sum() < 500:
            continue
        # ---- train on history only ----
        thr = float(vol_all[df_all["date"] < ws].dropna().quantile(params.percentile))
        ds_hist, _ = build_vol_spike_dataset(hist.reset_index(drop=True), params, threshold=thr)
        fh = causal_vol_features(hist.reset_index(drop=True), params)
        keep = fh.dropna(subset=cols).index
        yh = ds_hist["y"].loc[keep]
        if len(keep) < 1500 or len(np.unique(yh)) < 2:
            continue
        m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                          eval_metric="logloss", random_state=seed, n_jobs=1)
        m.fit(fh.loc[keep, cols], yh)
        # ---- emit over the window ----
        win = df_all[win_mask].reset_index(drop=True)
        fw = causal_vol_features(win, params)
        fw_clean = fw.dropna(subset=cols)
        p = np.full(len(win), np.nan)
        if len(fw_clean):
            p[fw_clean.index] = m.predict_proba(fw_clean[cols])[:, 1]
        sig = generate_alpha_signals(win, p, alpha_params)
        opens = win["open"].to_numpy(dtype=float)
        closes = win["close"].to_numpy(dtype=float)
        gross = _gross_pnl(win, sig["target"].to_numpy(), opens, closes)
        # net via EventBacktester (taker + spread + slippage; 1x notional cap; shorts OK)
        bt = EventBacktester(EngineConfig(
            equity=100_000.0,
            cost=CostModel(),
            risk_per_trade=0.01,
            max_leverage=1.0,
            allow_short=True,
        ))
        trades, _ = bt.run(win, sig["target"].to_numpy(), stop=sig["stop"].to_numpy())
        net = float(trades["pnl_net"].sum()) if len(trades) else 0.0
        fees = float(trades["fees_paid"].sum()) if len(trades) else 0.0
        n_trades = int(len(trades)) if len(trades) else 0
        window_rows.append({
            "window": f"{ws.date()}..{min(we, df_all['date'].max()).date()}",
            "gross": round(gross, 0), "net": round(net, 0), "fees": round(fees, 0),
            "trades": n_trades, "p_pred_mean": round(float(np.nanmean(p)), 3) if np.isfinite(p).any() else None,
        })
    return {"pair": pair, "direction": direction, "windows": window_rows}


def build_semiannual_blocks(start: str = "2022-01-01") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    blocks = []
    s = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(HOLDOUT_START, tz="UTC")
    while s < end:
        e = s + pd.tseries.offsets.DateOffset(months=WINDOW_MONTHS)
        blocks.append((s, min(e, end)))
        s = e
    return blocks


def main():
    blocks = build_semiannual_blocks()
    print("WFO blocks:", [f"{a.date()}..{b.date()}" for a, b in blocks])
    results = {}
    for pair in ("BTC/USDT", "ETH/USDT"):
        for direction in ("follow", "fade", "dipbuy"):
            r = run_window_wfo(pair, direction, blocks)
            results[f"{pair}_{direction}"] = r
            pos = sum(1 for w in r["windows"] if w["gross"] > 0)
            gross = sum(w["gross"] for w in r["windows"])
            net = sum(w["net"] for w in r["windows"])
            print(f"\n{pair} {direction}: gross-positive {pos}/{len(r['windows'])} "
                  f"windows | total gross=${gross:,.0f} net=${net:,.0f}")
            for w in r["windows"]:
                print(f"  {w['window']}: gross=${w['gross']:>10,.0f} net=${w['net']:>10,.0f} "
                      f"trades={w['trades']:4d}")
    _save(results, "wfo_alpha")


if __name__ == "__main__":
    main()
