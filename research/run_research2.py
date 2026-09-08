"""Research pipeline stages 6-10: regime, OU diagnostic, meta-labeling, ablation.

Extends run_research.py. Run after stages 1-5:
  .venv/Scripts/python research/run_research2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.constants import HOLDOUT_START
from research_lib.data.loader import load_pair_tf
from research_lib.strategies.regime_hmm import HMMParams, HMMRegimeEngine
from research_lib.strategies.ou import OUEngine, OUParams, fit_ou
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimulator
from research_lib.ml.triple_barrier import TripleBarrierLabeler, TripleBarrierParams
from research_lib.features.engine import FeatureEngine
from research_lib.risk.costs import CostModel
from research_lib.validation.metrics import summarize_backtest, trade_metrics

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts"
TF = "4h"
START_EQUITY = 100_000.0
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]


def _save(df, stage: str, name: str) -> Path:
    out = ARTIFACTS / stage
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    if isinstance(df, pd.DataFrame):
        df.to_csv(path.with_suffix(".csv"), index=False)
    else:
        path.with_suffix(".json").write_text(json.dumps(df, indent=2, default=str))
    return path


def load_btc() -> pd.DataFrame:
    df = load_pair_tf("BTC/USDT", TF)
    return df[df["date"] < pd.Timestamp(HOLDOUT_START, tz="UTC")].reset_index(drop=True)


def turtle_trades_for_regime(df: pd.DataFrame, equity: float = START_EQUITY) -> tuple[pd.DataFrame, list[dict]]:
    sim = TurtleSimulator(TurtleSimParams(equity=equity, allow_short=False, max_leverage=1.0))
    trades, _, log = sim.run(df)
    return trades, log


# ---------------------------------------------------------------------------
def stage_regime_breakdown() -> dict:
    print("\n=== STAGE 6: HMM regime walk-forward + Turtle trade regime attribution ===")
    df = load_btc()
    eng = HMMRegimeEngine(HMMParams(n_states=3, train_days=120, refit_days=60, n_iter=100))
    t0 = time.time()
    reg = eng.run(df)
    print(f"  HMM walk-forward on {len(df)} bars in {time.time()-t0:.1f}s")
    # label distribution + state characterization
    print("  label counts:", reg.labels.value_counts().to_dict())
    print("  characterizations:")
    seen = set()
    for c in reg.characterizations:
        key = c.state_id
        if key in seen:
            continue
        seen.add(key)
        print(f"    state {c.state_id}: vol={c.volatility:.4f} ac={c.autocorr:.3f} "
              f"mu={c.mean_return:.5f} dur={c.duration_bars:.0f} -> {c.regime_label}")
    # attribute turtle trades to the regime at ENTRY time (regime label known at entry bar)
    trades, _ = turtle_trades_for_regime(df)
    df["regime"] = reg.labels.to_numpy()
    entry_dates = pd.to_datetime(trades["entry_time"], utc=True)
    trade_regime = []
    for d in entry_dates:
        # regime at the entry time = regime of the last bar strictly before entry
        mask = df["date"] < d
        if mask.sum() == 0:
            trade_regime.append("UNCERTAIN")
        else:
            trade_regime.append(df.loc[mask, "regime"].iloc[-1])
    trades["entry_regime"] = trade_regime
    out_rows = []
    for regime, grp in trades.groupby("entry_regime"):
        if len(grp) < 3:
            continue
        m = summarize_backtest(grp, None, timeframe=TF, start_equity=START_EQUITY / 4)
        out_rows.append({"regime": regime, "trades": m["n_trades"], "win": round(m["win_rate"], 3),
                         "total_pnl": round(m["total_pnl"], 0), "pf": round(m["profit_factor"], 2)})
        print(f"  {regime:12s}: trades={m['n_trades']:3d} win={m['win_rate']:.2f} "
              f"pnl=${m['total_pnl']:>9,.0f} pf={m['profit_factor']:.2f}")
    res = pd.DataFrame(out_rows)
    _save(res, "regime", "turtle_regime_breakdown")
    _save(reg.labels.value_counts().to_dict(), "regime", "label_counts")
    return {"breakdown": res}


# ---------------------------------------------------------------------------
def stage_ou_diagnostic() -> dict:
    print("\n=== STAGE 7: OU mean-reversion diagnostic on real BTC (honest check) ===")
    df = load_btc()
    # candidate tradable series: log price (a spread-less single series). OU on log price is
    # the natural first candidate - but crypto log-prices are not stationary, so we EXPECT
    # the gates to reject it. We also test the z-score of log price (demeaned) explicitly.
    lp = np.log(df["close"])
    model = fit_ou(lp, OUParams(min_sample=1500, rolling_window=1000))
    print(f"  log-price OU: valid={model.valid} b={model.b:.4f} hl={model.half_life:.1f} "
          f"r2={model.r2:.4f} reasons={model.reject_reasons[:3]}")
    # explicitly run the engine - it must NOT trade if the model is invalid
    eng = OUEngine(OUParams(min_sample=1500))
    res = eng.run(df, lp, fit_end=4000)
    n_signals = int((res.signals["target"] != 0).sum())
    print(f"  OU engine: model_valid={res.model.valid} signals={n_signals}")
    out = {
        "log_price_valid": bool(model.valid),
        "b": model.b, "half_life": model.half_life,
        "reject_reasons": model.reject_reasons[:5],
        "n_signals": n_signals,
    }
    _save(out, "ou", "diagnostic")
    return out


# ---------------------------------------------------------------------------
def stage_meta_label() -> dict:
    print("\n=== STAGE 8: Triple-barrier meta-labeling of Turtle entries (BTC) ===")
    df = load_btc()
    # turtle events (fresh-breakout entries)
    sim = TurtleSimulator(TurtleSimParams(equity=START_EQUITY / 4))
    trades, _, log = sim.run(df)
    entries = pd.DataFrame(
        [
            {"date": x["date"], "side": x["side"]}
            for x in log
            if x["action"] == "entry"
        ]
    )
    # causal feature frame for event rows
    fe = FeatureEngine(hurst_window=400)
    feats = fe.build(df)
    feats = feats.dropna().reset_index(drop=True)
    df = df.iloc[feats.index].reset_index(drop=True)
    # align entries to available feature rows
    fdates = pd.to_datetime(feats["date"], utc=True)
    valid_entries = entries[
        entries["date"].isin(set(fdates))
    ].reset_index(drop=True)
    # label with triple barrier (vertical = 60 bars = 10 days of 4h)
    lb = TripleBarrierLabeler(TripleBarrierParams(pt_mult=3, sl_mult=3, vertical_bars=60))
    labels, _ = lb.label_events(df, valid_entries)
    print(f"  turtle entries={len(valid_entries)} labeled={len(labels)} "
          f"label dist: {labels['label'].value_counts().to_dict()}")
    if len(labels) < 30:
        print("  too few events for a meaningful meta-model - recording and skipping.")
        return {"n_events": len(labels)}
    # meta target and features
    from research_lib.ml.metalabel import ProbabilityCalibrator, build_event_dataset, default_models

    X, y, meta = build_event_dataset(feats, labels)
    # keep only numeric columns for the model (drop dates/ids/string regime labels)
    X = X.select_dtypes(include=[np.number])
    meta_cols = list(X.columns)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Purged K-fold evaluation (events are time-ordered with horizons)
    from sklearn.metrics import roc_auc_score, brier_score_loss

    n = len(y)
    cut = int(n * 0.7)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y[:cut], y[cut:]
    results = {}
    for name, model in default_models(seed=7).items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
        brier = brier_score_loss(yte, p)
        # isotonic calibration on a validation slice of TRAIN
        cal = ProbabilityCalibrator()
        cal.fit(ytr[: int(len(ytr) * 0.7)], model.predict_proba(Xtr)[: int(len(ytr) * 0.7), 1])
        p_cal = cal.calibrate(p)
        results[name] = {
            "auc": auc, "brier_raw": brier, "brier_cal": brier_score_loss(yte, p_cal),
            "base_rate": float(yte.mean()), "mean_p": float(p.mean()),
        }
        print(f"  {name}: auc={auc:.3f} brier_raw={brier:.3f} brier_cal={results[name]['brier_cal']:.3f}")
    _save(results, "metalabel", "cv_summary")
    return {"n_events": len(labels), "results": results}


# ---------------------------------------------------------------------------
def main() -> None:
    r6 = stage_regime_breakdown()
    r7 = stage_ou_diagnostic()
    r8 = stage_meta_label()
    print("\nStages 6-8 complete.")


if __name__ == "__main__":
    main()
