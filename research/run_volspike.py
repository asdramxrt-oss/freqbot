"""Blueprint Phase 2 research on REAL 15m data:
  (a) does the vol-spike model have genuine OOS signal (AUC) on BTC/ETH 15m?
  (b) does filtering pairs entries during predicted vol expansions reduce losses?
  (c) does a vol-spike filter turn the negative pairs edge positive? (expected: no - a
      filter cannot manufacture edge from a negative-gross signal; verified, not assumed)
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
from research_lib.ml.volspike import (
    VolSpikeParams,
    causal_vol_features,
    evaluate_vol_spike_filter,
)
from research_lib.strategies.pairs import PairsParams, generate_signals, log_ratio_series
from research_lib.strategies.pairs_sim import PairsSimParams, PairsSimulator
from research_lib.risk.costs import CostModel

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "pairs"
TF = "15m"


def _save(obj, name: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


def load_trimmed(pair):
    df = load_pair_tf(pair, TF)
    return df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)


def main() -> None:
    print("=== Phase 2 research: vol-spike prediction on real 15m ===")
    # (a) vol-spike model OOS signal on BTC and ETH
    vol_res = {}
    for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        df = load_trimmed(pair)
        r = evaluate_vol_spike_filter(df, np.zeros(len(df)),
                                      VolSpikeParams(horizon_bars=96, min_train_bars=2000))
        print(f"  {pair}: {r}")
        vol_res[pair] = r
    _save(vol_res, "volspike_model")

    # (b)+(c) filter effect on the pairs strategy (ETH/BTC 14d window)
    print("\n=== pairs + vol-spike entry filter (ETH/BTC, 14d window) ===")
    a, b = load_trimmed("ETH/USDT"), load_trimmed("BTC/USDT")
    m = log_ratio_series(a, b)
    p = PairsParams(est_window_bars=1344, z_entry=2.0, z_exit=0.0, ou_gate=False, cost=CostModel())
    res = generate_signals(m, p)
    sim = PairsSimulator(PairsSimParams(equity=100_000, capital_fraction=0.25))
    trades_base, eq_base = sim.run(m, res.target)
    print(f"  unfiltered: n={len(trades_base)} pnl=${trades_base['pnl_net'].sum():,.0f}")

    # build a simple causal vol filter: no entry if current YZ vol percentile > 0.85
    from research_lib.features.volatility import VolatilityEngine

    v = VolatilityEngine(method="yang_zhang", window=20).compute(a)
    vol_pct = v["volatility_percentile"].to_numpy()
    # align to m: m is an inner join of a,b by date; recompute on m directly using a's frame
    # align via dates
    ad = a.set_index("date")
    md = m["date"]
    vol_pct_aligned = pd.Series(vol_pct, index=a["date"]).reindex(md).to_numpy()
    # filter: suppress entries when vol percentile high
    res_f = res  # copy targets then zero-out entry bars with high vol
    target_f = res.target.copy()
    # entry bars are where target transitions 0->nonzero; find them
    diff = np.diff(np.concatenate([[0], target_f]))
    entry_idx = np.where(diff != 0)[0]
    blocked = 0
    for i in entry_idx:
        if np.isfinite(vol_pct_aligned[i]) and vol_pct_aligned[i] > 0.85:
            # remove this entry and hold flat until target would have exited
            j = i
            while j < len(target_f) and target_f[j] != 0:
                target_f[j] = 0
                j += 1
            blocked += 1
    trades_filt, eq_filt = sim.run(m, target_f)
    print(f"  vol-filtered : n={len(trades_filt)} pnl=${trades_filt['pnl_net'].sum():,.0f} "
          f"(blocked {blocked} entries)")
    _save({"unfiltered_pnl": float(trades_base["pnl_net"].sum()),
           "filtered_pnl": float(trades_filt["pnl_net"].sum()),
           "blocked_entries": blocked, "n_base": len(trades_base), "n_filt": len(trades_filt)},
          "volspike_filter_effect")


if __name__ == "__main__":
    main()
