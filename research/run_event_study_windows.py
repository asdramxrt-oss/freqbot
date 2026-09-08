"""Regime decomposition of the vol-spike event study.

The pooled study shows large positive forward drift after predicted vol spikes in BOTH
momentum directions - a bull-market-beta signature. Decompose by semi-annual window to see
whether the conditional effect survives BEAR regimes (2022) or is bull-only.
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
from research_lib.features.volatility import VolatilityEngine

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "alpha"


def _save(obj, name):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


def windowed_study(pair: str) -> dict:
    df = load_pair_tf(pair, "15m")
    df = df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)
    params = VolSpikeParams(horizon_bars=96, percentile=0.90)
    n = len(df)
    cut = int(n * 0.6)
    feats = causal_vol_features(df, params)
    vol = VolatilityEngine(method="yang_zhang", window=20).compute(df)["volatility"]
    thr = float(vol.iloc[:cut].dropna().quantile(params.percentile))
    ds, _ = build_vol_spike_dataset(df, params, threshold=thr)
    cols = [c for c in feats.columns if c != "date"]
    tr = feats.iloc[:cut].dropna(subset=cols).index
    from xgboost import XGBClassifier

    m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, eval_metric="logloss",
                      random_state=0, n_jobs=1)
    m.fit(feats.loc[tr, cols], ds["y"].loc[tr])
    te_idx = feats.iloc[cut:].dropna(subset=cols).index
    p = m.predict_proba(feats.loc[te_idx, cols])[:, 1]
    out = df.iloc[te_idx][["date", "close"]].copy()
    out["p"] = p
    out["mom8"] = np.log(out["close"] / out["close"].shift(8))
    out["fwd_96"] = np.log(out["close"].shift(-96) / out["close"])

    rows = []
    start = out["date"].iloc[0]
    while start < out["date"].iloc[-1]:
        we = start + pd.tseries.offsets.DateOffset(months=6)
        w = out[(out["date"] >= start) & (out["date"] < we)]
        if len(w) > 500:
            # window realized return (regime label)
            reg = float(np.log(w["close"].iloc[-1] / w["close"].iloc[0])) * 1e4
            hi = w["p"] > 0.6
            row = {"window": f"{start.date()}..{min(we, out['date'].iloc[-1]).date()}",
                   "n": int(len(w)), "regime_ret_bps": round(reg, 1),
                   "spike_n": int(hi.sum())}
            for lbl, mask in [("all", np.ones(len(w), bool)), ("spike", hi),
                              ("spike_mom+", hi & (w["mom8"] > 0)),
                              ("spike_mom-", hi & (w["mom8"] < 0))]:
                s = w.loc[mask, "fwd_96"].dropna()
                row[f"{lbl}_fwd96_bps"] = round(float(s.mean() * 1e4), 1) if len(s) else None
            rows.append(row)
        start = we
    return rows


def main():
    res = {}
    for pair in ("BTC/USDT", "ETH/USDT"):
        res[pair] = windowed_study(pair)
        print(f"--- {pair} by semi-annual window (regime_ret = window return bps) ---")
        hdr = f"{'window':26s} {'regime':>8s} {'n':>6s} {'spk':>5s} {'all':>8s} {'spike':>8s} {'spk_mom+':>9s} {'spk_mom-':>9s}"
        print(hdr)
        for r in res[pair]:
            print(f"{r['window']:26s} {r['regime_ret_bps']:8.0f} {r['n']:6d} {r['spike_n']:5d} "
                  f"{r['all_fwd96_bps']:8.1f} {r['spike_fwd96_bps']:8.1f} "
                  f"{r['spike_mom+_fwd96_bps']:9.1f} {r['spike_mom-_fwd96_bps']:9.1f}")
    _save(res, "event_study_windowed")


if __name__ == "__main__":
    main()
