"""WFO for the vol-spike BREAKOUT alpha (range breakout gated on predicted vol).

The one structurally distinct monetization of the magnitude forecast: join the expansion
direction only after price exits its pre-spike range, rather than guessing direction from
pre-spike momentum.
"""

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
from research_lib.strategies.volspike_alpha import VolSpikeAlphaParams, generate_alpha_signals
from research_lib.risk.backtest_engine import EventBacktester, EngineConfig
from research_lib.risk.costs import CostModel

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts" / "alpha"
TF = "15m"


def _save(obj, name):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


def gross_pnl(win, target):
    opens = win["open"].to_numpy(float)
    closes = win["close"].to_numpy(float)
    pnl, pos, epx, prev = 0.0, 0, 0.0, 0
    for t in range(1, len(win)):
        d = int(target[t])
        if pos == 0 and d != 0 and prev == 0:
            pos, epx = d, opens[t]
        elif pos != 0 and d == 0:
            pnl += (opens[t] - epx) * pos
            pos = 0
        elif pos != 0 and d != pos and prev == 0:
            pnl += (opens[t] - epx) * pos
            pos, epx = d, opens[t]
        prev = d
    if pos != 0:
        pnl += (closes[-1] - epx) * pos
    return float(pnl)


def evaluate_breakout(pair, p_entry, pct=0.90, breakout_bars=32, pct_label=""):
    df = load_pair_tf(pair, TF)
    df = df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)
    params = VolSpikeParams(horizon_bars=96, percentile=pct)
    feats = causal_vol_features(df, params)
    vol = VolatilityEngine(method="yang_zhang", window=20).compute(df)["volatility"]
    cols = [c for c in feats.columns if c != "date"]
    from xgboost import XGBClassifier

    a_params = VolSpikeAlphaParams(direction="breakout", p_entry=p_entry,
                                   p_exit=round(p_entry * 0.6, 3),
                                   breakout_bars=breakout_bars)
    rows = []
    s = pd.Timestamp("2022-01-01", tz="UTC")
    while s < pd.Timestamp(HOLDOUT_START, tz="UTC"):
        we = s + pd.tseries.offsets.DateOffset(months=6)
        hist = df[df["date"] < s]
        win = df[(df["date"] >= s) & (df["date"] < min(we, df["date"].max()))].reset_index(drop=True)
        if len(hist) >= 4000 and len(win) >= 500:
            thr = float(vol[df["date"] < s].dropna().quantile(pct))
            dh, _ = build_vol_spike_dataset(hist.reset_index(drop=True), params, threshold=thr)
            fh = causal_vol_features(hist.reset_index(drop=True), params)
            keep = fh.dropna(subset=cols).index
            yh = dh["y"].loc[keep]
            m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                              eval_metric="logloss", random_state=0, n_jobs=1)
            m.fit(fh.loc[keep, cols], yh)
            fw = causal_vol_features(win, params).dropna(subset=cols)
            p = np.full(len(win), np.nan)
            p[fw.index] = m.predict_proba(fw[cols])[:, 1]
            sig = generate_alpha_signals(win, p, a_params)
            g = gross_pnl(win, sig["target"].to_numpy())
            bt = EventBacktester(EngineConfig(equity=100_000.0, cost=CostModel(),
                                              risk_per_trade=0.01, max_leverage=1.0, allow_short=True))
            tr, _ = bt.run(win, sig["target"].to_numpy(), stop=sig["stop"].to_numpy())
            rows.append({"window": f"{s.date()}..{min(we, df['date'].max()).date()}",
                         "gross": round(g, 0), "net": round(float(tr['pnl_net'].sum()), 0) if len(tr) else 0.0,
                         "trades": int(len(tr))})
        s = we
    return rows


def main():
    out = {}
    for pair in ("BTC/USDT", "ETH/USDT"):
        for pct in (0.75, 0.90):
            for pe in (0.3, 0.5):
                for bb in (16, 32):
                    rows = evaluate_breakout(pair, pe, pct=pct, breakout_bars=bb)
                    pos = sum(1 for r in rows if r["gross"] > 0)
                    tg = sum(r["gross"] for r in rows)
                    tn = sum(r["net"] for r in rows)
                    nt = sum(r["trades"] for r in rows)
                    key = f"{pair[:3]}_p{pct:.2f}_pe{pe}_bb{bb}"
                    out[key] = {"pair": pair, "pct": pct, "pe": pe, "bb": bb, "windows": rows,
                                "gross_pos": f"{pos}/{len(rows)}", "total_gross": tg,
                                "total_net": tn, "trades": nt}
                    print(f"{pair:9s} p{pct:.2f} pe{pe:.1f} bb{bb:2d}: {pos}/{len(rows)} gross-pos "
                          f"gross=${tg:>9,.0f} net=${tn:>9,.0f} trades={nt}")
    _save(out, "wfo_breakout")


if __name__ == "__main__":
    main()
