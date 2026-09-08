"""Alpha-discovery event study (DeepSeek brief Step 1-2, evidence first).

Question: is there ANY conditional directional signal after a PREDICTED volatility spike?

We condition forward returns on:
  A) the model's vol-spike prediction (causal, walk-forward probabilities)
  B) short-horizon momentum sign (pre-spike drift)
and measure E[forward return | state] net of nothing (gross) across horizons.

If no state has materially positive gross expectancy, then directional vol-ignition
strategies (H1/H2/H3) have no basis and we report that instead of curve-fitting.
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

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "alpha"
TF = "15m"


def _save(obj, name):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


def event_study(pair: str) -> dict:
    df = load_pair_tf(pair, TF)
    df = df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)
    params = VolSpikeParams(horizon_bars=96, percentile=0.90)
    n = len(df)
    cut = int(n * 0.6)
    # causal fit on first 60%, emit on last 40% (the validated AUC 0.82-0.90 setting)
    feats = causal_vol_features(df, params)
    eng = __import__("research_lib.features.volatility", fromlist=["VolatilityEngine"]).VolatilityEngine(
        method="yang_zhang", window=20)
    vol = eng.compute(df)["volatility"]
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

    # align series
    out = df.iloc[te_idx][["date", "close"]].copy()
    out["p"] = p
    out["y"] = ds["y"].loc[te_idx].to_numpy()
    # momentum: last 8-bar return (2h), causal
    out["mom"] = np.log(out["close"] / out["close"].shift(8))
    # forward returns at several horizons (gross)
    for h in (8, 24, 48, 96):
        out[f"fwd_{h}"] = np.log(out["close"].shift(-h) / out["close"])

    # event study: conditional means
    res = {}
    hi = out["p"] > 0.6          # predicted spike
    lo = out["p"] < 0.3          # predicted quiet
    res["n_test"] = int(len(out))
    res["p_spike_rate"] = float(hi.mean())
    res["y_spike_rate"] = float(out["y"].mean())
    res["auc"] = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
        out["y"], out["p"])) if len(np.unique(out["y"])) > 1 else None

    # E[forward ret | predicted state]  (gross, bps)
    for state_name, mask in [("all", np.ones(len(out), bool)), ("pred_spike", hi),
                             ("pred_quiet", lo), ("spike_mom+", hi & (out["mom"] > 0)),
                             ("spike_mom-", hi & (out["mom"] < 0)),
                             ("quiet_mom+", lo & (out["mom"] > 0)),
                             ("quiet_mom-", lo & (out["mom"] < 0))]:
        row = {}
        for h in (8, 24, 48, 96):
            s = out.loc[mask, f"fwd_{h}"].dropna()
            row[f"fwd_{h}_bps"] = float(s.mean() * 1e4) if len(s) else None
            row[f"n"] = int(len(s))
        res[state_name] = row
    print(f"--- {pair} event study ---")
    print(f"  auc={res['auc']:.3f} pred_spike_rate={res['p_spike_rate']:.2%}")
    for k in ("all", "pred_spike", "pred_quiet", "spike_mom+", "spike_mom-", "quiet_mom+", "quiet_mom-"):
        r = res[k]
        print(f"  {k:10s}: " + " ".join(f"{h}:{r[f'fwd_{h}_bps']:+.1f}bps" for h in (8, 24, 48, 96) if r.get(f'fwd_{h}_bps') is not None))
    return res


def main():
    results = {}
    for pair in ("BTC/USDT", "ETH/USDT"):
        results[pair] = event_study(pair)
    _save(results, "event_study")


if __name__ == "__main__":
    main()
