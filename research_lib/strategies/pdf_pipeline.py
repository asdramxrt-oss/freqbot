"""Full-architecture directional pipeline on FUTURES (perp) panels.

Wires the design-PDF architecture end to end, per coin, on USDT-M perp data only:

    perp OHLCV -> Feature Engine (YZ vol, ATR, RSI, fracdiff, Hurst)
      -> Turtle Engine (breakout, long AND short - perps allow both)
      -> Regime Engine (causal filtered HMM) + vol-spike kill switch -> Signal Router
      -> Triple Barrier -> Meta Model (RF/XGB, AUC-gated) -> Bet Sizer
      -> EventBacktester (risk engine: stop-based sizing, leverage cap, costed fills)

No spot series is read anywhere in this module. Every stage delegates to the
validated research modules; the only new code is the per-coin wiring, the
meta-model entry-suppression state machine, and the panel aggregation with
stop passthrough. Acceptance gates live in validation/directional_acceptance.py.

Timebase note: router/HMM/vol-spike constants were tuned on 15m bars; they are
scaled x1/4 for the 1h perp panel (same wall-clock windows). HMM estimation uses
a rolling 8000-bar window (not expanding) so the 46k-bar panel stays tractable;
regime models are local by nature, and the characterization is train-window only
either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.bot.config import BotConfig
from research_lib.bot.features import PrimarySignal, build_feature_frame
from research_lib.bot.sizing import MetaModel
from research_lib.risk.backtest_engine import EngineConfig, EventBacktester
from research_lib.risk.costs import CostModel
from research_lib.strategies.directional import DirectionalResult, TRADING_DAYS_PER_YEAR
from research_lib.strategies.turtle import TurtleEngine, TurtleParams
from research_lib.strategies.unified_router import RouterConfig, filtered_hmm_router

# 15m-tuned router constants scaled to the 1h perp panel (same wall-clock time)
ROUTER_CFG_1H = RouterConfig(
    refit_bars=672,          # ~4 weeks of 1h bars (15m: 2688 -> /4)
    min_train_bars=2000,     # ~12 weeks (15m: 4000 -> /2, keeps HMM sample adequate)
    vol_kill_p=0.80,
    vol_refit_bars=1344,
    train_window=8000,       # rolling estimation window cap (see module docstring)
)


@dataclass
class PDFCoinResult:
    symbol: str
    target: np.ndarray
    stop: np.ndarray
    n_primary_entries: int
    n_routed_entries: int
    n_meta_suppressed: int
    meta_enabled: bool
    meta_note: str


@dataclass
class PDFPanelSignals:
    per_coin: dict[str, PDFCoinResult] = field(default_factory=dict)
    # the fitted pooled panel meta model (v2+), exported so the production
    # runner can score entries with the SAME model the acceptance measured
    meta_model: object | None = None
    meta_calibrator: object | None = None

    @property
    def notes(self) -> list[str]:
        out = []
        for c in self.per_coin.values():
            out.append(
                f"{c.symbol}: {c.n_primary_entries} primary -> {c.n_routed_entries} routed, "
                f"{c.n_meta_suppressed} meta-suppressed; meta "
                f"{'ON' if c.meta_enabled else 'OFF'} ({c.meta_note})"
            )
        return out


def _entry_events(target: np.ndarray, stop: np.ndarray, df: DataFrame, symbol: str) -> list[PrimarySignal]:
    """Flat -> +-1 transitions of a (routed) turtle target, as PrimarySignals."""
    events: list[PrimarySignal] = []
    opens = df["open"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()
    prev = 0
    for t in range(len(target)):
        side = int(target[t])
        if side != 0 and prev == 0 and np.isfinite(opens[t]) and opens[t] > 0:
            events.append(
                PrimarySignal(
                    strategy_id="turtle",
                    symbol=symbol,
                    bar_index=t,
                    signal_time=pd.Timestamp(dates[t - 1] if t > 0 else dates[t]),
                    side=side,
                    entry_ref=float(opens[t]),
                    stop_price=float(stop[t]) if np.isfinite(stop[t]) else None,
                )
            )
        prev = side
    return events


def build_pdf_panel_signals(
    panels: dict[str, DataFrame],
    bot_cfg: BotConfig,
    router_cfg: RouterConfig | None = None,
    seed: int = 42,
) -> PDFPanelSignals:
    """Run the whole PDF stack per coin and return backtest-ready targets + stops."""
    router_cfg = router_cfg or ROUTER_CFG_1H
    out = PDFPanelSignals()
    for sym in sorted(panels):
        df = panels[sym].reset_index(drop=True)
        feats = build_feature_frame(df, bot_cfg)

        # Turtle primary (long + short - futures allow both sides)
        params = TurtleParams(
            system=1,
            entry_n=bot_cfg.strategy.turtle_entry_n,
            atr_n=bot_cfg.strategy.turtle_atr_n,
            stop_mult=bot_cfg.strategy.turtle_stop_mult,
        )
        signals_df, _ = TurtleEngine(params).generate(df)
        turtle_target = signals_df["target"].to_numpy(dtype=int)
        turtle_stop = signals_df["stop"].to_numpy(dtype=float)

        # Regime router: causal filtered HMM + vol-spike kill switch (this coin's regime)
        router = filtered_hmm_router(df, router_cfg, seed=seed)
        gate = router.turtle_gate
        routed = np.where(gate, turtle_target, 0).astype(int)

        primary_events = _entry_events(turtle_target, turtle_stop, df, sym)
        routed_events = _entry_events(routed, turtle_stop, df, sym)

        # Triple barrier -> meta model (AUC-gated) -> bet sizer suppression
        meta = MetaModel(bot_cfg)
        report = meta.fit(feats, routed_events)
        final = routed.copy()
        n_suppressed = 0
        if report.enabled:
            ev_by_bar = {e.bar_index: e for e in routed_events}
            state = 0
            suppressed_episode = False
            for t in range(len(routed)):
                d = int(routed[t])
                if state == 0:
                    if suppressed_episode:
                        if d == 0:
                            suppressed_episode = False
                        final[t] = 0
                        continue
                    if d != 0:
                        sig = ev_by_bar.get(t)
                        p = meta.predict(feats, sig) if sig is not None else 0.5
                        if p < bot_cfg.meta.min_confidence:
                            suppressed_episode = True
                            n_suppressed += 1
                            final[t] = 0
                            continue
                        state = d
                    final[t] = state
                else:
                    state = d
                    final[t] = d

        out.per_coin[sym] = PDFCoinResult(
            symbol=sym,
            target=final,
            stop=turtle_stop,
            n_primary_entries=len(primary_events),
            n_routed_entries=len(routed_events),
            n_meta_suppressed=n_suppressed,
            meta_enabled=report.enabled,
            meta_note=report.note,
        )
    return out


# ---------------------------------------------------------------------------
# v2 enhancements (2026-09-14), pre-registered BEFORE the acceptance run:
#   mode "panelmeta": hard router gate kept; the triple-barrier meta model is
#       trained on events POOLED across all coins (coin one-hot added) instead of
#       one starved model per coin. Fixes the v1 failure mode where every coin
#       had < 120 routed events and the meta stage could never switch on.
#   mode "regfeat": the HMM regime is a FEATURE, not a gate (blueprint 2.1: "use
#       it as a feature for the XGBoost model"). Only the validated vol-spike
#       kill switch still vetoes entries; the pooled panel meta model gets
#       regime one-hot + P(spike) as inputs and decides.
# ---------------------------------------------------------------------------
_REGIMES = ("TREND", "LOW_VOL_MR", "NEUTRAL", "UNCERTAIN")


def _v2_event_row(
    bar: pd.Series,
    entry_price: float,
    side: int,
    sym_idx: int,
    n_syms: int,
    regime: str | None,
    p_spike: float | None,
) -> np.ndarray | None:
    """Feature row for one event: v1 base + coin one-hot (+ regime features)."""
    from research_lib.bot.sizing import _row_from_bar

    row = _row_from_bar(bar, entry_price)
    if row is None:
        return None
    row = row.copy()
    row[4] *= float(np.sign(side) or 1.0)
    coin_oh = np.zeros(n_syms)
    coin_oh[sym_idx] = 1.0
    parts = [row, coin_oh]
    if regime is not None:
        reg_oh = np.array([1.0 if regime == r else 0.0 for r in _REGIMES])
        ps = p_spike if p_spike is not None and np.isfinite(p_spike) else 0.5
        parts += [reg_oh, np.array([ps])]
    return np.concatenate(parts)


def build_pdf_panel_signals_v2(
    panels: dict[str, DataFrame],
    bot_cfg: BotConfig,
    router_cfg: RouterConfig | None = None,
    mode: str = "panelmeta",
    seed: int = 42,
    primary: str = "turtle",
    meta_pt: float | None = None,
    meta_sl: float | None = None,
    meta_vert: int | None = None,
    meta_expanded: bool = False,
    funding_map: dict | None = None,
) -> PDFPanelSignals:
    """v2 of the PDF stack: pooled panel meta model, regime-as-feature option.

    ``primary`` selects the entry engine the PDF stack wraps:
      "turtle"    - the fast Turtle breakout (v1/v2; proven edge-less, kept for
                    the honest comparison grid)
      "slowtrend" - the ONE validated directional variant (4h Donchian 60/30,
                    long above / short below the 1200-bar SMA, 2xATR stop) - v3.
                    The meta question that actually matters: can triple-barrier
                    meta-labeling improve a primary that already has edge?

    Same honesty contract as ``MetaModel.fit``: chronological 60/40 split (on the
    pooled, time-sorted events), EVERY candidate model must clear min_test_auc on
    the held-out tail or the meta stage is disabled and primaries pass at full size.
    """
    if mode not in ("panelmeta", "regfeat"):
        raise ValueError(f"unknown v2 mode: {mode}")
    if primary not in ("turtle", "slowtrend"):
        raise ValueError(f"unknown primary: {primary}")
    router_cfg = router_cfg or ROUTER_CFG_1H
    syms = sorted(panels)
    n_syms = len(syms)
    regfeat = mode == "regfeat"

    per_coin: dict[str, dict] = {}
    fit_rows: list[np.ndarray] = []
    fit_y: list[int] = []
    fit_times: list[pd.Timestamp] = []

    from research_lib.ml.metalabel import meta_target_from_labels
    from research_lib.ml.triple_barrier import TripleBarrierLabeler, TripleBarrierParams

    # B3 (meta battery, 2026-09-14): horizon-matched barriers + expanded features
    # are what gave the meta model its first real (if unstable) OOS signal.
    tb = TripleBarrierLabeler(
        TripleBarrierParams(
            pt_mult=meta_pt if meta_pt is not None else bot_cfg.meta.pt_mult,
            sl_mult=meta_sl if meta_sl is not None else bot_cfg.meta.sl_mult,
            vertical_bars=meta_vert if meta_vert is not None else bot_cfg.meta.vertical_bars,
            vol_scale_col=bot_cfg.meta.vol_scale_col,
        )
    )

    def _funding_extras(sym: str, ts_ns: int) -> tuple[float, float]:
        """(last funding rate, 30d mean) strictly before ts_ns; 0.0 when unknown."""
        if not funding_map or sym not in funding_map:
            return 0.0, 0.0
        f = funding_map[sym]
        d = f["date_ns"].to_numpy()
        k = int(np.searchsorted(d, ts_ns, side="right")) - 1
        if k < 0:
            return 0.0, 0.0
        return float(f["funding_rate"].iloc[k]), float(f["f30"].iloc[k])

    def _row(sym, si, bar, entry, side, regime, ps, ts_ns):
        row = _v2_event_row(bar, entry, side, si, n_syms, regime, ps)
        if row is None:
            return None
        if meta_expanded:
            def g(col, default=np.nan):
                v = bar.get(col, default)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            f_rate, f30 = _funding_extras(sym, ts_ns)
            row = np.concatenate([row, [g("fracdiff_close"), g("vol_yz_20"), f_rate, f30]])
        return row

    for si, sym in enumerate(syms):
        df = panels[sym].reset_index(drop=True)
        feats = build_feature_frame(df, bot_cfg)
        if primary == "slowtrend":
            from research_lib.bot.directional_runner import (
                DirectionalRunnerConfig,
                build_trend_signals,
            )

            turtle_target, turtle_stop = build_trend_signals(
                df, DirectionalRunnerConfig(symbol=sym)
            )
            turtle_target = turtle_target.astype(int)
        else:
            params = TurtleParams(
                system=1,
                entry_n=bot_cfg.strategy.turtle_entry_n,
                atr_n=bot_cfg.strategy.turtle_atr_n,
                stop_mult=bot_cfg.strategy.turtle_stop_mult,
            )
            signals_df, _ = TurtleEngine(params).generate(df)
            turtle_target = signals_df["target"].to_numpy(dtype=int)
            turtle_stop = signals_df["stop"].to_numpy(dtype=float)

        router = filtered_hmm_router(df, router_cfg, seed=seed)
        if regfeat:
            # vol-spike kill still vetoes (validated component); regime is NOT a gate
            pk = np.roll(router.p_spike, 1)
            pk[0] = np.nan
            spike_halt = np.where(np.isfinite(pk), pk > router_cfg.vol_kill_p, False)
            routed = np.where(spike_halt, 0, turtle_target).astype(int)
        else:
            routed = np.where(router.turtle_gate, turtle_target, 0).astype(int)

        events = _entry_events(routed, turtle_stop, df, sym)
        primary_events = _entry_events(turtle_target, turtle_stop, df, sym)
        per_coin[sym] = {
            "routed": routed,
            "stop": turtle_stop,
            "events": events,
            "n_primary": len(primary_events),
            "router": router,
            "feats": feats,
            "sym_idx": si,
        }

        # label this coin's events for the pooled fit
        ev_df = pd.DataFrame(
            {
                "date": [pd.Timestamp(e.signal_time) for e in events],
                "side": [int(e.side) for e in events],
                "entry_price": [float(e.entry_ref) for e in events],
            }
        )
        if len(ev_df) < 10:
            continue
        try:
            labels, _ = tb.label_events(feats, ev_df)
        except Exception:  # noqa: BLE001
            continue
        if labels is None or not len(labels):
            continue
        y = meta_target_from_labels(labels).to_numpy(dtype=int)
        # ns conversion must be unit-explicit: pyarrow feathers are millisecond,
        # astype("int64") would keep ms and clamp every lookup to the last bar
        dates_ns = (
            (pd.to_datetime(feats["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
            // pd.Timedelta(1, "ns")
        ).to_numpy()
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
            row = _row(
                sym, si, bar, entry, side,
                str(router.regime[idx]) if regfeat else None,
                float(router.p_spike[idx]) if regfeat else None,
                np.int64(ts.value),
            )
            if row is None or not np.isfinite(row).all():
                continue
            fit_rows.append(row)
            fit_y.append(int(yi))
            fit_times.append(ts)

    # ---- pooled panel meta model -------------------------------------------
    model, calibrator, meta_note = None, None, ""
    order = np.argsort(np.array(fit_times, dtype="datetime64[ns]")) if fit_times else []
    X = np.vstack([fit_rows[i] for i in order]) if fit_times else np.empty((0, 0))
    y = np.array([fit_y[i] for i in order], dtype=int)
    min_events = int(bot_cfg.meta.min_events)
    min_auc = float(getattr(bot_cfg.meta, "min_test_auc", 0.55))
    if len(y) >= min_events and len(np.unique(y)) == 2:
        split = int(len(y) * float(bot_cfg.meta.train_fraction))
        split = max(min(split, len(y) - 20), 20)
        X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]
        from sklearn.metrics import roc_auc_score

        from research_lib.ml.metalabel import default_models

        best_name, best_auc, best_model = "", -np.inf, None
        aucs: dict[str, float] = {}
        for name, mdl in default_models(seed=seed).items():
            try:
                mdl.fit(X_tr, y_tr)
                p = mdl.predict_proba(X_te)[:, 1]
                auc = float(roc_auc_score(y_te, p)) if len(np.unique(y_te)) == 2 else np.nan
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(auc):
                aucs[name] = auc
                if auc > best_auc:
                    best_name, best_auc, best_model = name, auc, mdl
        all_beat = bool(aucs) and all(a >= min_auc for a in aucs.values())
        if best_model is not None and np.isfinite(best_auc) and best_auc >= min_auc and all_beat:
            try:
                from sklearn.model_selection import cross_val_predict

                from research_lib.ml.metalabel import ProbabilityCalibrator

                cal = ProbabilityCalibrator(method="isotonic", cv=3)
                p_oof = cross_val_predict(best_model, X, y, cv=3, method="predict_proba")[:, 1]
                cal.fit(y, p_oof)
                best_model.fit(X, y)
                model, calibrator = best_model, cal
            except Exception:  # noqa: BLE001
                best_model.fit(X, y)
                model, calibrator = best_model, None
            meta_note = (
                f"panel meta enabled: {len(y)} pooled events, every model cleared "
                f"AUC >= {min_auc:.2f} (best {best_name} {best_auc:.3f}, per-model {aucs})"
            )
        else:
            meta_note = (
                f"panel meta OFF: no reliable OOS edge on {len(y)} pooled events "
                f"(best AUC {best_auc:.3f} < {min_auc:.2f}, per-model {aucs})"
            )
    else:
        meta_note = f"panel meta OFF: too few pooled events ({len(y)} < {min_events})"

    # ---- per-coin suppression with the panel model --------------------------
    out = PDFPanelSignals(meta_model=model, meta_calibrator=calibrator)
    for sym in syms:
        c = per_coin[sym]
        routed, events = c["routed"], c["events"]
        router, feats, si = c["router"], c["feats"], c["sym_idx"]
        final = routed.copy()
        n_suppressed = 0
        if model is not None:
            ev_by_bar = {e.bar_index: e for e in events}
            state = 0
            suppressed_episode = False
            for t in range(len(routed)):
                d = int(routed[t])
                if state == 0:
                    if suppressed_episode:
                        if d == 0:
                            suppressed_episode = False
                        final[t] = 0
                        continue
                    if d != 0:
                        sig = ev_by_bar.get(t)
                        p = 0.5
                        if sig is not None:
                            bar_t = feats.iloc[min(t, len(feats) - 1)]
                            ts_ns = int(
                                (pd.Timestamp(bar_t["date"]).tz_localize("UTC")
                                 if pd.Timestamp(bar_t["date"]).tzinfo is None
                                 else pd.Timestamp(bar_t["date"])).value
                            )
                            row = _row(
                                sym, si, bar_t,
                                float(sig.entry_ref), int(sig.side),
                                str(router.regime[t]) if regfeat else None,
                                float(router.p_spike[t]) if regfeat else None,
                                np.int64(ts_ns),
                            )
                            if row is not None and np.isfinite(row).all():
                                try:
                                    p = float(model.predict_proba(row.reshape(1, -1))[:, 1][0])
                                    if calibrator is not None:
                                        p = float(calibrator.calibrate(np.array([p]))[0])
                                except Exception:  # noqa: BLE001
                                    p = 0.5
                        if p < bot_cfg.meta.min_confidence:
                            suppressed_episode = True
                            n_suppressed += 1
                            final[t] = 0
                            continue
                        state = d
                    final[t] = state
                else:
                    state = d
                    final[t] = d

        out.per_coin[sym] = PDFCoinResult(
            symbol=sym,
            target=final,
            stop=c["stop"],
            n_primary_entries=c["n_primary"],
            n_routed_entries=len(events),
            n_meta_suppressed=n_suppressed,
            meta_enabled=model is not None,
            meta_note=meta_note,
        )
    return out


def run_panel_pdf(
    name: str,
    panels: dict[str, DataFrame],
    signals: PDFPanelSignals,
    cost: CostModel,
    equity: float = 100_000.0,
    risk_per_trade: float = 0.01,
    max_leverage: float = 2.0,
) -> DirectionalResult:
    """Per-coin EventBacktester with stop passthrough; portfolio = sum of sub-equities.

    Same aggregation contract as ``run_panel_directional`` (futures linear PnL, one
    position per coin, risk-sized by stop distance, capped by leverage).
    """
    syms = sorted(panels)
    sub_equity = equity / len(syms)
    all_trades: list[DataFrame] = []
    min_len = min(len(panels[s]) for s in syms)
    eq_sum = None
    dates = None
    per_coin_net: dict[str, float] = {}

    for sym in syms:
        df = panels[sym].reset_index(drop=True).iloc[:min_len]
        c = signals.per_coin[sym]
        bt = EventBacktester(
            EngineConfig(
                equity=sub_equity,
                cost=cost,
                risk_per_trade=risk_per_trade,
                max_leverage=max_leverage,
                allow_short=True,
            )
        )
        trades, eq = bt.run(df, c.target[:min_len], stop=c.stop[:min_len])
        trades["symbol"] = sym
        all_trades.append(trades)
        per_coin_net[sym] = float(eq[-1] - sub_equity) if len(eq) else 0.0
        eq_sum = eq if eq_sum is None else eq_sum + eq
        dates = df["date"].to_numpy()

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else DataFrame()
    if len(trades):
        trades = trades.sort_values("exit_time").reset_index(drop=True)

    eq = eq_sum if eq_sum is not None else np.full(2, equity)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / peak).max()) if len(eq) else 0.0
    span_days = max(
        (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).total_seconds() / 86400.0, 1.0
    ) if dates is not None and len(dates) > 1 else 1.0
    growth = max(float(eq[-1] / equity), 1e-9)
    return DirectionalResult(
        name=name,
        trades=trades,
        equity=eq,
        dates=dates if dates is not None else np.array([]),
        per_coin_net=per_coin_net,
        net_pnl=float(eq[-1] - equity),
        annualised_return=float(growth ** (TRADING_DAYS_PER_YEAR / span_days) - 1.0),
        max_drawdown=max_dd,
        n_trades=len(trades),
        cost_role="taker",
    )
