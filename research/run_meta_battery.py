"""FINAL meta-model battery (pre-registered 2026-09-14): try everything legitimate.

The meta model self-vetoed on 4 event pools (AUC 0.46-0.54). Two genuine defects
remained untested, plus a stronger evaluation:

  B1  current setup      - barriers pt=2/sl=2/vertical=48 (inherited from the 15m
                           config), base 5 features + coin one-hot.
  B2  horizon-matched    - the REAL defect: the barriers describe a quick
                           double-or-nothing while the slow trend trades a 2xATR
                           stop with NO take-profit over multi-week holds. Labels
                           therefore misclassify trend winners. Fix: pt=6/sl=2/
                           vertical=96 (the strategy's actual geometry).
  B3  horizon-matched + expanded features - blueprint 2.2: add fracdiff_close,
                           vol_yz_20, funding rate level and 30d funding mean
                           (positioning information, not price-derived).

Evaluation (the strictest in the project): pooled panel events, chronological
60/40 split (continuity) AND PurgedKFold(6, embargo=96) per-fold AUC for BOTH
xgb and rf. Meta is "fixed" only if, for B2 or B3, BOTH models average >= 0.55
across purged folds with NO fold below 0.50. Anything less: the meta book is
closed with maximal evidence. Events: slow-trend entries (cached v3 regfeat
targets, 4h, 5-coin perp panel). Every configuration reported.

Usage:  .venv/Scripts/python research/run_meta_battery.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lib.bot.config import BotConfig  # noqa: E402
from research_lib.bot.features import build_feature_frame  # noqa: E402
from research_lib.ml.triple_barrier import TripleBarrierLabeler, TripleBarrierParams  # noqa: E402
from research_lib.strategies.carry import load_funding  # noqa: E402
from research_lib.strategies.directional import resample_panel  # noqa: E402
from research_lib.strategies.pdf_pipeline import _entry_events, _v2_event_row  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data"
REPORT = ROOT / "research" / "reports" / "META_BATTERY_REPORT.md"
SYMBOLS = ("BTC", "ETH", "BNB", "SOL", "XRP")

CONFIGS = {
    "B1_current": dict(pt=2.0, sl=2.0, vert=48, expanded=False),
    "B2_horizon_matched": dict(pt=6.0, sl=2.0, vert=96, expanded=False),
    "B3_horizon_expanded": dict(pt=6.0, sl=2.0, vert=96, expanded=True),
}


def _funding_at(funding, ts_ns):
    """(last rate, 30d mean rate) strictly before ts_ns; NaN when no history."""
    if funding is None:
        return np.nan, np.nan
    d = funding["date_ns"].to_numpy()
    k = int(np.searchsorted(d, ts_ns, side="right")) - 1
    if k < 0:
        return np.nan, np.nan
    return float(funding["funding_rate"].iloc[k]), float(funding["f30"].iloc[k])


def build_pool(panels_4h, cached, funding_map, cfg):
    """Label all coins' slow-trend events under one barrier config; pool features."""
    bot_cfg = BotConfig()
    bot_cfg.data.timeframe = "4h"
    bot_cfg.data.use_hurst = True
    tb = TripleBarrierLabeler(
        TripleBarrierParams(
            pt_mult=cfg["pt"], sl_mult=cfg["sl"], vertical_bars=cfg["vert"],
            vol_scale_col="atr_20",
        )
    )
    rows, ys, times, t0s, t1s = [], [], [], [], []
    syms = sorted(panels_4h)
    for si, sym in enumerate(syms):
        df = panels_4h[sym]
        feats = build_feature_frame(df, bot_cfg)
        c = cached.per_coin[sym]
        events = _entry_events(c.target, c.stop, df, sym)
        ev_df = pd.DataFrame(
            {
                "date": [pd.Timestamp(e.signal_time) for e in events],
                "side": [int(e.side) for e in events],
                "entry_price": [float(e.entry_ref) for e in events],
            }
        )
        if len(ev_df) < 10:
            continue
        labels, _ = tb.label_events(feats, ev_df)
        if labels is None or not len(labels):
            continue
        from research_lib.ml.metalabel import meta_target_from_labels

        y = meta_target_from_labels(labels).to_numpy(dtype=int)
        # ns conversion must be unit-explicit (pyarrow feathers are millisecond)
        dates_ns = (
            (pd.to_datetime(feats["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
            // pd.Timedelta(1, "ns")
        ).to_numpy()
        fund = funding_map.get(sym)
        for rec, yi in zip(labels.to_dict("records"), y, strict=True):
            ts = pd.Timestamp(rec["event_start"])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            idx = min(max(int(np.searchsorted(dates_ns, np.int64(ts.value), side="left")), 0), len(feats) - 1)
            bar = feats.iloc[idx]
            side = int(rec.get("side") or 1)
            pt, sl = rec.get("pt"), rec.get("sl")
            entry = float(bar["close"])
            if pt is not None and sl is not None and np.isfinite(pt) and np.isfinite(sl):
                entry = (float(pt) + float(sl)) / 2.0
            row = _v2_event_row(bar, entry, side, si, len(syms), None, None)
            if row is None:
                continue
            if cfg["expanded"]:
                def g(col, default=np.nan):
                    v = bar.get(col, default)
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return default

                f_rate, f30 = _funding_at(fund, np.int64(ts.value))
                row = np.concatenate(
                    [row, [g("fracdiff_close"), g("vol_yz_20"),
                           np.nan_to_num(f_rate), np.nan_to_num(f30)]]
                )
            if not np.isfinite(row).all():
                continue
            rows.append(row)
            ys.append(int(yi))
            times.append(np.int64(ts.value))
            t0s.append(idx)
            t1s.append(min(idx + cfg["vert"], len(feats) - 1))

    order = np.argsort(np.array(times)) if times else []
    X = np.vstack([rows[i] for i in order]) if times else np.empty((0, 0))
    y = np.array([ys[i] for i in order], dtype=int)
    ev = pd.DataFrame({"t0": [t0s[i] for i in order], "t1": [t1s[i] for i in order]})
    return X, y, ev


def evaluate(X, y, events, seed=42):
    """Chronological 60/40 + PurgedKFold(6, embargo=96) AUCs for xgb and rf."""
    from sklearn.metrics import roc_auc_score

    from research_lib.ml.metalabel import default_models

    out = {"n": len(y)}
    if len(y) < 120 or len(np.unique(y)) < 2:
        out["note"] = f"insufficient events ({len(y)})"
        return out

    # chronological split (continuity with earlier runs)
    split = max(min(int(len(y) * 0.6), len(y) - 20), 20)
    chrono = {}
    for name, mdl in default_models(seed=seed).items():
        try:
            mdl.fit(X[:split], y[:split])
            p = mdl.predict_proba(X[split:])[:, 1]
            chrono[name] = float(roc_auc_score(y[split:], p))
        except Exception:  # noqa: BLE001
            chrono[name] = float("nan")
    out["chrono"] = chrono

    # purged folds (the strict test): 6 contiguous TIME blocks over the panel;
    # test = events starting in the block, train = events whose label horizon
    # ends before block_start - embargo or starts after block_end + embargo.
    # (Event-count folds purge everything when episodes cluster in time.)
    embargo = 96
    t0 = events["t0"].to_numpy(dtype=int)
    t1 = events["t1"].to_numpy(dtype=int)
    max_bar = max(int(t1.max()) + 1, 1)
    bounds = np.linspace(0, max_bar, 7).astype(int)
    splits = []
    for k in range(6):
        a, b = bounds[k], bounds[k + 1]
        te = np.where((t0 >= a) & (t0 < b))[0]
        tr = np.where((t1 < a - embargo) | (t0 > b + embargo))[0]
        if len(te) >= 10 and len(tr) >= 40:
            splits.append((tr, te))
    out["purged_cfg"] = f"6 time blocks, embargo={embargo}, usable folds={len(splits)}"
    folds = {}
    for name, mdl in default_models(seed=seed).items():
        aucs = []
        for tr, te in splits:
            if len(te) < 10 or len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            import copy

            m = copy.deepcopy(mdl)
            try:
                m.fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                aucs.append(float(roc_auc_score(y[te], p)))
            except Exception:  # noqa: BLE001
                continue
        folds[name] = aucs
    out["purged"] = folds
    return out


def main() -> int:
    panels_1h = {
        s: pd.read_feather(DATA / "futures" / f"{s}_USDT-1h-perp.feather") for s in SYMBOLS
    }
    panels_4h = {s: resample_panel(df, "4h") for s, df in panels_1h.items()}
    funding_map = {}
    for s in SYMBOLS:
        f = DATA / "funding" / f"{s}_USDT_USDT_funding.csv"
        if f.exists():
            fd = load_funding(f)
            fd = fd.copy()
            fd["date_ns"] = (
                (pd.to_datetime(fd["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
                // pd.Timedelta(1, "ns")
            )
            fd["f30"] = fd["funding_rate"].rolling(90, min_periods=30).mean()
            funding_map[s] = fd

    cached = pickle.loads(
        (ROOT / "research" / "artifacts" / "pdf_signals_pdf_v3_4h_slowtrend_regfeat.pkl").read_bytes()
    )

    lines = [
        "# FINAL Meta-Model Battery",
        "",
        "Slow-trend entry events (v3 regfeat targets, 4h, 5-coin perp panel),",
        "pooled panel meta. B1: current barriers/features. B2: horizon-matched",
        "barriers (pt=6/sl=2/vert=96 = the strategy's real geometry). B3: B2 +",
        "fracdiff/YZ-vol/funding features. Evaluation: chronological 60/40 +",
        "PurgedKFold(6, embargo=96). Fix criterion (pre-registered): BOTH models",
        "average >= 0.55 across purged folds with no fold < 0.50.",
        "",
        "## Summary",
        "",
        "| config | n events | chrono AUC (xgb/rf) | purged mean AUC (xgb/rf) | purged folds (xgb) | purged folds (rf) | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    any_fixed = False
    for name, cfg in CONFIGS.items():
        print(f"building pool: {name}...")
        X, y, events = build_pool(panels_4h, cached, funding_map, cfg)
        res = evaluate(X, y, events)
        if "note" in res:
            lines.append(f"| {name} | {res['n']} | - | - | - | - | {res['note']} |")
            continue
        cx, cr = res["chrono"].get("xgb", float("nan")), res["chrono"].get("rf", float("nan"))
        px, pr = res["purged"].get("xgb", []), res["purged"].get("rf", [])
        mx, mr = (float(np.mean(px)) if px else float("nan")), (float(np.mean(pr)) if pr else float("nan"))
        fixed = bool(
            px and pr and mx >= 0.55 and mr >= 0.55
            and min(px) >= 0.50 and min(pr) >= 0.50
        )
        any_fixed |= fixed
        fmt = lambda a: ", ".join(f"{v:.3f}" for v in a)  # noqa: E731
        lines.append(
            f"| {name} | {res['n']} | {cx:.3f} / {cr:.3f} | {mx:.3f} / {mr:.3f} | "
            f"{fmt(px)} | {fmt(pr)} | {'FIXED' if fixed else 'no edge'} |"
        )
        print(f"  {name}: purged xgb={mx:.3f} rf={mr:.3f} -> {'FIXED' if fixed else 'no edge'}")

    lines += [
        "",
        f"**Verdict: meta model {'FIXED - has OOS edge' if any_fixed else 'CLOSED - no configuration reaches the pre-registered bar. The self-veto was correct.'}**",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0 if any_fixed else 1


if __name__ == "__main__":
    raise SystemExit(main())
