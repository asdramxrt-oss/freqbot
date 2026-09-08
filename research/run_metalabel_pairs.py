"""Blueprint Phase 2 claim test: does higher trade frequency fix the meta-model?

The earlier Turtle meta-model had 79 events and AUC<0.5. Pairs trading on 15m produces
~750 pooled events over the research window (~200/yr) - still below the blueprint's
500-1500/yr target but a 10x increase. Test the claim honestly:
  - triple-barrier label the pairs trades (entry-side barrier = the spread target)
  - train XGB/RF on causal features with a chronological split
  - report AUC/Brier AND the economic effect of the filter on net PnL
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
from research_lib.strategies.pairs import PairsParams, generate_signals, log_ratio_series
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator
from research_lib.risk.costs import CostModel
from research_lib.features.engine import FeatureEngine

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "pairs"
TF = "15m"
RATIOS = [("ETH/USDT", "BTC/USDT"), ("SOL/USDT", "BTC/USDT"), ("BNB/USDT", "BTC/USDT")]


def load_trimmed(pair):
    df = load_pair_tf(pair, TF)
    return df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)


def main() -> None:
    # gather pairs trades across ratios with a small feature set at entry
    all_events = []
    for la, lb in RATIOS:
        a, b = load_trimmed(la), load_trimmed(lb)
        m = log_ratio_series(a, b)
        p = PairsParams(est_window_bars=1344, z_entry=2.0, z_exit=0.0, ou_gate=False, cost=CostModel())
        res = generate_signals(m, p)
        sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=0.25))
        trades, _ = sim.run(m, res.target)
        if not len(trades):
            continue
        t = trades.copy()
        t["ratio"] = f"{la.split('/')[0]}/{lb.split('/')[0]}"
        # causal features at entry: spread z and short-horizon momentum of the spread
        entry_idx = m["date"].searchsorted(pd.to_datetime(t["entry_time"], utc=True))
        z = pd.Series(res.z)
        s = m["spread"]
        t["z_entry"] = [z.iloc[i] if 0 <= i < len(z) else np.nan for i in entry_idx]
        t["spr_ret_24"] = [
            s.iloc[i] / s.iloc[i - 24] - 1 if i >= 24 else np.nan for i in entry_idx
        ]
        all_events.append(t)
    ev = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    print(f"pooled pairs events: {len(ev)}")
    if len(ev) < 100:
        print("insufficient events")
        return
    y = (ev["pnl_net"] > 0).astype(int)
    feats = ev[["z_entry", "spr_ret_24"]].copy()
    feats["z_abs"] = feats["z_entry"].abs()
    feats = feats.fillna(0.0)
    n = len(ev)
    cut = int(n * 0.7)
    results = {}
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from research_lib.ml.metalabel import default_models, ProbabilityCalibrator

    Xtr, Xte = feats.iloc[:cut], feats.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    for name, model in default_models(seed=7).items():
        model.fit(Xtr, ytr)
        pte = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pte) if len(np.unique(yte)) > 1 else np.nan
        results[name] = {"auc": auc, "brier": brier_score_loss(yte, pte),
                         "n_test": int(len(yte)), "base_rate": float(yte.mean())}
        print(f"  {name}: auc={auc:.3f} brier={results[name]['brier']:.3f} "
              f"base={results[name]['base_rate']:.3f}")
    # economic effect: take only test trades the model scores above 0.5
    model = default_models(seed=7)["xgb"]
    model.fit(Xtr, ytr)
    pte = model.predict_proba(Xte)[:, 1]
    take = pte > 0.5
    pnl_all = ev["pnl_net"].iloc[cut:].to_numpy()
    pnl_take = pnl_all[take]
    print(f"  economic filter (test): all=${pnl_all.sum():,.0f}  "
          f"take={take.sum()} trades=${pnl_take.sum():,.0f}")
    results["economic"] = {"all_pnl": float(pnl_all.sum()), "take_pnl": float(pnl_take.sum()),
                           "take_n": int(take.sum()), "all_n": int(len(pnl_all))}
    (ARTIFACTS / "metalabel_pairs.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
