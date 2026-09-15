"""Router-book runner - the production-path twin of research/run_router_book_acceptance.py
config D (the profit-maximizing configuration within the 25% drawdown budget):

    TREND sleeves (75% of equity, 5 perps): slow trend (4h Donchian 60/30 +
      SMA1200 long/short, 2xATR stop) with ONLY the validated vol-spike kill as
      veto, 2% equity risk per trade.
    MR sleeves (25% of equity, 4 ratios vs BTC): OU z-score pairs with the
      causal OU half-life gate, active only while leg A's HMM pairs_gate is
      open, maker fills, dollar-neutral legs.

Every order in every sleeve takes the production chain live orders must take:

    OrderIntent -> CentralRiskEngine -> ExecutionEngine (idempotency + order
    state machine) -> PaperBroker/LiveBroker -> PositionReconciler

so research-vs-production divergence is measurable before deployment (the test
suite cross-checks this runner against the research harness on identical data).

Signals are computed causally up front from the SAME research functions
(build_trend_signals, generate_signals, filtered_hmm_router) - the runner only
replays them through the execution chain, bar by bar, exactly as it would in
live. Naked-leg protection on MR entries: if the second leg fails to fill, the
first is unwound immediately at taker (carry_runner contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.bot.broker import PaperBroker
from research_lib.bot.config import ExecutionConfig
from research_lib.execution.engine import ExecutionEngine
from research_lib.execution.order_state import OrderStateStore
from research_lib.execution.reconciliation import PositionReconciler
from research_lib.risk.costs import CostModel
from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits
from research_lib.strategies.directional import resample_panel
from research_lib.strategies.ou import OUParams, fit_ou
from research_lib.strategies.pairs import PairsParams, generate_signals, log_ratio_series
from research_lib.strategies.unified_router import RouterConfig, RouterOutput

TREND_SYMS = ("BTC", "ETH", "BNB", "SOL", "XRP")
MR_RATIOS = (("ETH", "BTC"), ("SOL", "BTC"), ("BNB", "BTC"), ("XRP", "BTC"))

# 15m-tuned router constants scaled to 4h bars (same wall-clock windows)
ROUTER_CFG_4H = RouterConfig(
    refit_bars=168,
    min_train_bars=1000,
    vol_kill_p=0.80,
    vol_refit_bars=336,
    train_window=1998,
)


@dataclass
class RouterBookConfig:
    """Config D - see research/reports/ROUTER_BOOK_ACCEPTANCE_REPORT.md."""

    equity: float = 100_000.0
    risk_per_trade: float = 0.02       # D: 2% (validated frontier maximum)
    trend_share: float = 0.75          # D: 75% trend / 25% MR
    leverage: float = 2.0
    donchian_entry: int = 60
    donchian_exit: int = 30
    sma_bars: int = 1200
    atr_n: int = 20
    stop_mult: float = 2.0
    vol_kill_p: float = 0.80
    mr_est_window: int = 42            # 7 days of 4h bars
    mr_z_entry: float = 2.0
    mr_z_exit: float = 0.0
    mr_max_hold: int = 84
    mr_ou_min_sample: int = 30
    mr_capital_fraction: float = 0.50  # 0.25 scaled by the 2% risk budget
    mr_post_only: bool = True
    recon_every_bars: int | None = 168
    # ---- PDF bet sizer (fractional Kelly on the pooled panel meta model) ----
    # OFF by default: the acceptance showed the meta ranks (AUC 0.615) but the
    # binary 0.52 suppression threshold is degenerate. Kelly sizing is the
    # PDF-native use: scale = fractional_kelly_size(p, b); 0 means no bet.
    meta_kelly: bool = False
    meta_kelly_frac: float = 0.25      # quarter-Kelly (variance control)
    meta_b_prior: float = 3.0          # prior win/loss ratio (realized b = 6.1
    meta_b_min_trades: int = 30        # but estimated causally after N trades)
    # live mode: do NOT flatten at end of data - the final positions are the
    # book's current target state, which the live loop diffs against the broker
    flatten_at_end: bool = True


class RouterBookRunner:
    """Run the 9-sleeve router book through the production execution chain."""

    def __init__(
        self,
        cfg: RouterBookConfig,
        routers: dict[str, RouterOutput],
        exec_cfg: ExecutionConfig | None = None,
        meta_signals=None,
        funding_map: dict | None = None,
    ):
        self.cfg = cfg
        self.routers = routers  # precomputed causal router outputs per trend coin
        # pooled panel meta model (fitted by build_pdf_panel_signals_v2, B3 config)
        self._meta_model = getattr(meta_signals, "meta_model", None)
        self._meta_cal = getattr(meta_signals, "meta_calibrator", None)
        self._funding_map = funding_map or {}
        ec = exec_cfg or ExecutionConfig()
        self.cost = CostModel(
            maker_fee=ec.maker_fee, taker_fee=ec.taker_fee,
            spread_bps=ec.spread_bps, slippage_bps=ec.slippage_bps,
        )
        self.broker = PaperBroker(ec)
        # portfolio-level limits; sleeve sizing governs (stop-based trend sizing,
        # fixed fraction MR). The drawdown kill switch stays armed at the budget.
        self.risk = CentralRiskEngine(
            limits=RiskLimits(
                per_trade_risk=cfg.risk_per_trade,
                per_symbol_notional=cfg.leverage,
                portfolio_notional=4.0,   # 9 sleeves, MR legs 2-sided by design
                max_leverage=4.0,
                max_position_notional=cfg.leverage,
                daily_loss_limit=1.0,
                weekly_loss_limit=1.0,
                max_drawdown_kill=0.25,
                max_strategy_fraction=cfg.leverage,
            )
        )
        self.risk.state.equity = cfg.equity
        self.risk.state.peak_equity = cfg.equity
        self.risk.state.day_start_equity = cfg.equity
        self.execution = ExecutionEngine(
            risk=self.risk, store=OrderStateStore(), broker=self.broker,
            model_version="router-book-D-v1",
        )
        self.reconciler = PositionReconciler(self.risk)
        self.log: list[dict] = []

    # ------------------------------------------------------------------
    def _intent(self, strategy: str, symbol: str, side: int, units: float,
                price: float, stop: float | None, when: str, post_only: bool) -> OrderIntent:
        return OrderIntent(
            strategy_id=strategy, symbol=symbol, signal_time=when, side=side,
            units=units, entry_ref_price=price, stop_price=stop,
            notional=units * price, post_only=post_only,
        )

    def _execute(self, intent: OrderIntent, ref_price: float) -> dict[str, Any] | None:
        # PaperBroker holds one mark; it is set immediately before each submit,
        # and the fill books against the intent's own symbol
        self.broker.set_market(ref_price)
        res = self.execution.execute(intent)
        if res.submitted and res.broker_ack and res.broker_ack.get("status") == "filled":
            return res.broker_ack
        self.log.append(
            {"action": "order_not_filled", "symbol": intent.symbol,
             "side": intent.side, "message": res.message, "time": intent.signal_time}
        )
        return None

    # ------------------------------------------------------------------
    def run(self, perp_1h: dict[str, DataFrame]) -> dict:
        """Replay the book over 1h perp history (resampled to 4h) via the chain."""
        cfg = self.cfg
        panels = {s: resample_panel(df.sort_values("date").reset_index(drop=True), "4h")
                  for s, df in perp_1h.items()}
        n = min(len(panels[s]) for s in TREND_SYMS)
        panels = {s: panels[s].iloc[:n].reset_index(drop=True) for s in panels}

        from research_lib.bot.directional_runner import DirectionalRunnerConfig, build_trend_signals

        trend: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in TREND_SYMS:
            trend[s] = build_trend_signals(
                panels[s], DirectionalRunnerConfig(
                    symbol=f"{s}/USDT", donchian_entry=cfg.donchian_entry,
                    donchian_exit=cfg.donchian_exit, sma_bars=cfg.sma_bars,
                    atr_n=cfg.atr_n, stop_mult=cfg.stop_mult,
                )
            )
        mr = self._mr_signals(panels)

        # ---- PDF bet sizer setup (fractional Kelly on the panel meta model) --
        meta_on = bool(cfg.meta_kelly and self._meta_model is not None)
        feats_map: dict[str, DataFrame] = {}
        if meta_on:
            from research_lib.bot.config import BotConfig as _BotCfg
            from research_lib.bot.features import build_feature_frame

            bc = _BotCfg()
            bc.data.timeframe = "4h"
            bc.data.use_hurst = True
            for s in TREND_SYMS:
                feats_map[s] = build_feature_frame(panels[s], bc)
        closed_pnls: list[float] = []

        def current_b() -> float:
            """Causal win/loss ratio from this run's own closed trend trades."""
            wins = [x for x in closed_pnls if x > 0]
            losses = [x for x in closed_pnls if x < 0]
            if len(closed_pnls) >= cfg.meta_b_min_trades and wins and losses:
                return float(np.mean(wins) / abs(np.mean(losses)))
            return cfg.meta_b_prior

        def meta_scale(sym: str, bar, entry_price: float, side: int, t: int) -> float | None:
            """fractional-Kelly scale in [0, 1] for this entry; None = unscorable."""
            from research_lib.ml.metalabel import fractional_kelly_size
            from research_lib.strategies.pdf_pipeline import _v2_event_row

            si = TREND_SYMS.index(sym)
            row = _v2_event_row(
                bar, entry_price, side, si, len(TREND_SYMS),
                str(self.routers[sym].regime[t]), float(self.routers[sym].p_spike[t]),
            )
            if row is None:
                return None

            def g(col, default=np.nan):
                v = bar.get(col, default)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            f_rate, f30 = 0.0, 0.0
            f = self._funding_map.get(sym)
            if f is not None:
                ts = pd.Timestamp(bar["date"])
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
                d = f["date_ns"].to_numpy()
                k = int(np.searchsorted(d, np.int64(ts.value), side="right")) - 1
                if k >= 0:
                    f_rate = float(f["funding_rate"].iloc[k])
                    f30 = float(f["f30"].iloc[k])
            row = np.concatenate([row, [g("fracdiff_close"), g("vol_yz_20"), f_rate, f30]])
            if not np.isfinite(row).all():
                return None
            try:
                p = float(self._meta_model.predict_proba(row.reshape(1, -1))[:, 1][0])
                if self._meta_cal is not None:
                    p = float(self._meta_cal.calibrate(np.array([p]))[0])
            except Exception:  # noqa: BLE001
                return None
            return fractional_kelly_size(
                p, current_b(), cfg.meta_kelly_frac, max_position_frac=1.0
            )

        # per-sleeve state
        sub_trend = cfg.equity * cfg.trend_share / len(TREND_SYMS)
        sub_mr = cfg.equity * (1.0 - cfg.trend_share) / len(MR_RATIOS)
        t_side = {s: 0 for s in TREND_SYMS}
        t_units = {s: 0.0 for s in TREND_SYMS}
        t_entry = {s: 0.0 for s in TREND_SYMS}
        t_stop = {s: np.nan for s in TREND_SYMS}
        t_openfee = {s: 0.0 for s in TREND_SYMS}
        t_eq = {s: sub_trend for s in TREND_SYMS}
        m_pos = {r: 0 for r in MR_RATIOS}      # +1 long A/short B, -1 reverse
        m_qa = {r: 0.0 for r in MR_RATIOS}
        m_qb = {r: 0.0 for r in MR_RATIOS}
        m_pa = {r: 0.0 for r in MR_RATIOS}
        m_pb = {r: 0.0 for r in MR_RATIOS}
        m_eq = {r: sub_mr for r in MR_RATIOS}
        cooldown = {s: False for s in TREND_SYMS}

        trades: list[dict] = []
        curve: list[dict] = []
        fees_total = 0.0
        dates = panels["BTC"]["date"].to_numpy()

        def close_trend(sym, price, t, reason):
            nonlocal fees_total
            side = t_side[sym]
            when = str(pd.Timestamp(dates[t]))
            ack = self._execute(
                self._intent(f"trend_{sym}", f"{sym}/USDT:PERP", -side, t_units[sym],
                             price, None, when, post_only=False), price)
            if ack is None:
                self.log.append({"action": "close_failed_abort", "sleeve": sym, "time": when})
                return False
            xp = float(ack["avg_price"])
            fee = float(ack.get("fee", 0.0))
            fees_total += fee
            pnl = side * (xp - t_entry[sym]) * t_units[sym] - (t_openfee[sym] + fee)
            self.risk.on_position_closed(f"{sym}/USDT:PERP", pnl)
            t_eq[sym] += pnl
            closed_pnls.append(pnl)
            trades.append({"sleeve": f"trend:{sym}", "exit_time": dates[t], "side": side,
                           "entry_price": t_entry[sym], "exit_price": xp,
                           "pnl_net": pnl, "exit_reason": reason})
            t_side[sym] = 0
            t_units[sym] = 0.0
            return True

        def mr_close(ratio, px_a, px_b, t, reason):
            nonlocal fees_total
            pos = m_pos[ratio]
            a, b = ratio
            when = str(pd.Timestamp(dates[t]))
            # close both legs; if either fails, keep the book honest and abort
            pnl_legs = 0.0
            for sym, qty, entry, side in (
                (a, m_qa[ratio], m_pa[ratio], pos), (b, m_qb[ratio], m_pb[ratio], -pos)
            ):
                px = px_a if sym == a else px_b
                ack = self._execute(
                    self._intent(f"mr_{a}{b}", f"{sym}/USDT:PERP", -side, qty,
                                 px, None, when, post_only=False), px)
                if ack is None:
                    self.log.append({"action": "mr_close_failed_abort", "ratio": ratio, "time": when})
                    return False
                xp = float(ack["avg_price"])
                fee = float(ack.get("fee", 0.0))
                fees_total += fee
                pnl_legs += side * (xp - entry) * qty - fee
                self.risk.on_position_closed(f"{sym}/USDT:PERP",
                                             side * (xp - entry) * qty - fee)
            m_eq[ratio] += pnl_legs
            trades.append({"sleeve": f"mr:{a}/{b}", "exit_time": dates[t], "side": pos,
                           "pnl_net": pnl_legs, "exit_reason": reason})
            m_pos[ratio] = 0
            m_qa[ratio] = m_qb[ratio] = 0.0
            return True

        def mr_open(ratio, pos, px_a, px_b, t):
            """Dollar-neutral two legs; unwind the first if the second fails."""
            nonlocal fees_total
            a, b = ratio
            when = str(pd.Timestamp(dates[t]))
            notional = m_eq[ratio] * cfg.mr_capital_fraction
            legs = ((a, pos, px_a), (b, -pos, px_b))
            acks = []
            for sym, side, px in legs:
                ack = self._execute(
                    self._intent(f"mr_{a}{b}", f"{sym}/USDT:PERP", side,
                                 notional / px, px, None, when, cfg.mr_post_only), px)
                acks.append(ack)
                if ack is None:
                    break
            if len(acks) == 2 and all(acks):
                m_pos[ratio] = pos
                m_qa[ratio] = float(acks[0]["filled"])
                m_qb[ratio] = float(acks[1]["filled"])
                m_pa[ratio] = float(acks[0]["avg_price"])
                m_pb[ratio] = float(acks[1]["avg_price"])
                fees_total += float(acks[0].get("fee", 0.0)) + float(acks[1].get("fee", 0.0))
                for sym, side, px in legs:
                    q = m_qa[ratio] if sym == a else m_qb[ratio]
                    e = m_pa[ratio] if sym == a else m_pb[ratio]
                    self.risk.on_position_opened(
                        self._intent(f"mr_{a}{b}", f"{sym}/USDT:PERP", side, q, e,
                                     None, when, False))
                return True
            if acks and acks[0] is not None:
                sym, side, px = legs[0]
                self._execute(
                    self._intent(f"mr_{a}{b}", f"{sym}/USDT:PERP", -side,
                                 float(acks[0]["filled"]), px, None, when, post_only=False), px)
                self.log.append({"action": "mr_entry_aborted_second_leg_failed", "time": when})
            return False

        for t in range(n):
            when = str(pd.Timestamp(dates[t]))
            # ---- TREND sleeves ----
            for s in TREND_SYMS:
                df = panels[s]
                target, stop = trend[s]
                desired = int(target[t])
                # config D's vol-spike kill: P(spike) decided at close t-1 vetoes
                # the bar (blocks entries AND flushes open positions, matching
                # the research harness's gated-target semantics)
                pk = float(self.routers[s].p_spike[t - 1]) if t > 0 else np.nan
                if np.isfinite(pk) and pk > cfg.vol_kill_p:
                    desired = 0
                o, h, l, c = (float(df["open"].iloc[t]), float(df["high"].iloc[t]),
                              float(df["low"].iloc[t]), float(df["close"].iloc[t]))
                if desired == 0:
                    cooldown[s] = False
                if desired != t_side[s] and not (cooldown[s] and t_side[s] == 0):
                    if t_side[s] != 0 and not close_trend(s, o, t, "signal"):
                        break
                    if desired != 0:
                        st = stop[t]
                        if np.isfinite(st) and abs(o - st) > 0:
                            # same sizing contract as the research EventBacktester:
                            # risk-budget units, notional capped at leverage x
                            # sleeve equity. (The book-level risk engine checks
                            # margin against TOTAL equity, so the 0.45
                            # single-symbol cap from the directional runner does
                            # not apply here.)
                            scale = 1.0
                            if meta_on:
                                sc = meta_scale(s, feats_map[s].iloc[t], o, desired, t)
                                if sc is not None:
                                    scale = sc
                                if scale <= 0.0:
                                    self.log.append(
                                        {"action": "meta_kelly_zero", "sleeve": s, "time": when}
                                    )
                                    continue  # Kelly says no bet: episode skipped
                            risk_units = (t_eq[s] * cfg.risk_per_trade * scale * (1.0 - 1e-9)
                                          / abs(o - st))
                            u = float(min(risk_units, t_eq[s] * cfg.leverage / o))
                            ack = self._execute(
                                self._intent(f"trend_{s}", f"{s}/USDT:PERP", desired, u,
                                             o, float(st), when, post_only=False), o)
                            if ack is not None:
                                t_side[s] = desired
                                t_units[s] = float(ack["filled"])
                                t_entry[s] = float(ack["avg_price"])
                                t_stop[s] = float(st)
                                t_openfee[s] = float(ack.get("fee", 0.0))
                                fees_total += t_openfee[s]
                                self.risk.on_position_opened(
                                    self._intent(f"trend_{s}", f"{s}/USDT:PERP", desired,
                                                 t_units[s], t_entry[s], float(st), when, False))
                if t_side[s] == 1 and l <= t_stop[s]:
                    px = o if o <= t_stop[s] else t_stop[s]
                    if not close_trend(s, px, t, "stop"):
                        break
                    cooldown[s] = True
                elif t_side[s] == -1 and h >= t_stop[s]:
                    px = o if o >= t_stop[s] else t_stop[s]
                    if not close_trend(s, px, t, "stop"):
                        break
                    cooldown[s] = True

            # ---- MR sleeves ----
            for r in MR_RATIOS:
                a, b = r
                df = panels[a]
                tgt = mr[r]["target"]
                gate = mr[r]["gate"]
                desired = int(tgt[t]) if t < len(tgt) else 0
                oa = float(panels[a]["open"].iloc[t])
                ob = float(panels[b]["open"].iloc[t])
                if m_pos[r] != 0 and (desired == 0 or not gate[t]):
                    if not mr_close(r, oa, ob, t, "signal" if desired == 0 else "regime_halt"):
                        break
                elif m_pos[r] == 0 and desired != 0 and gate[t]:
                    mr_open(r, desired, oa, ob, t)

            # ---- mark-to-market ----
            mtm = 0.0
            for s in TREND_SYMS:
                c = float(panels[s]["close"].iloc[t])
                mtm += t_eq[s] + (t_side[s] * (c - t_entry[s]) * t_units[s]
                                  if t_side[s] != 0 else 0.0)
            for r in MR_RATIOS:
                a, b = r
                ca = float(panels[a]["close"].iloc[t])
                cb = float(panels[b]["close"].iloc[t])
                u = 0.0
                if m_pos[r] != 0:
                    u = (m_pos[r] * (ca - m_pa[r]) * m_qa[r]
                         + (-m_pos[r]) * (cb - m_pb[r]) * m_qb[r])
                mtm += m_eq[r] + u
            curve.append({"date": dates[t], "equity": mtm})

            if cfg.recon_every_bars and (t + 1) % cfg.recon_every_bars == 0:
                internal = {}
                for s in TREND_SYMS:
                    if t_side[s] != 0:
                        internal[f"{s}/USDT:PERP"] = internal.get(f"{s}/USDT:PERP", 0.0) + t_side[s] * t_units[s]
                for r in MR_RATIOS:
                    if m_pos[r] != 0:
                        a, b = r
                        internal[f"{a}/USDT:PERP"] = internal.get(f"{a}/USDT:PERP", 0.0) + m_pos[r] * m_qa[r]
                        internal[f"{b}/USDT:PERP"] = internal.get(f"{b}/USDT:PERP", 0.0) - m_pos[r] * m_qb[r]
                rr = self.reconciler.reconcile(internal, self.broker.positions())
                if rr.consistent:
                    self.reconciler.resume()
                else:
                    self.log.append({"action": "reconciliation_failed", "diffs": rr.diffs, "time": when})

        # end of data: flatten every sleeve through the same chain (research),
        # or keep positions as the live target state (flatten_at_end=False)
        last = n - 1
        if cfg.flatten_at_end:
            for s in TREND_SYMS:
                if t_side[s] != 0:
                    close_trend(s, float(panels[s]["close"].iloc[last]), last, "end_of_data")
            for r in MR_RATIOS:
                if m_pos[r] != 0:
                    mr_close(r, float(panels[r[0]]["close"].iloc[last]),
                             float(panels[r[1]]["close"].iloc[last]), last, "end_of_data")

        book_flat = all(t_side[s] == 0 for s in TREND_SYMS) and all(
            m_pos[r] == 0 for r in MR_RATIOS)
        final_positions: dict[str, float] = {}
        if not cfg.flatten_at_end:
            for s in TREND_SYMS:
                if t_side[s] != 0:
                    key = f"{s}/USDT:PERP"
                    final_positions[key] = final_positions.get(key, 0.0) + t_side[s] * t_units[s]
            for r in MR_RATIOS:
                if m_pos[r] != 0:
                    a, b = r
                    ka, kb = f"{a}/USDT:PERP", f"{b}/USDT:PERP"
                    final_positions[ka] = final_positions.get(ka, 0.0) + m_pos[r] * m_qa[r]
                    final_positions[kb] = final_positions.get(kb, 0.0) - m_pos[r] * m_qb[r]
        eq = np.array([c["equity"] for c in curve], dtype=float) if curve else np.full(2, cfg.equity)
        peak = np.maximum.accumulate(eq)
        span_days = max(
            (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).total_seconds() / 86400.0, 1.0
        )
        growth = max(float(eq[-1] / cfg.equity), 1e-9)
        return {
            "equity_curve": DataFrame(curve),
            "trades": DataFrame(trades),
            "net_pnl": float(eq[-1] - cfg.equity),
            "final_equity": float(eq[-1]),
            "total_fees": fees_total,
            "n_trades": len(trades),
            "max_drawdown": float(((peak - eq) / peak).max()) if len(eq) else 0.0,
            "annualised_return": growth ** (365.0 / span_days) - 1.0,
            "kill_switch": self.risk.state.kill_switch,
            "reconciliation_ok": self.risk.state.reconciliation_ok,
            "book_flat": book_flat,
            "final_positions": final_positions,
            "sleeve_state": {
                **{f"trend:{s}": {"side": t_side[s], "units": t_units[s], "eq": t_eq[s]}
                   for s in TREND_SYMS},
                **{f"mr:{a}/{b}": {"pos": m_pos[(a, b)], "eq": m_eq[(a, b)]}
                   for a, b in MR_RATIOS},
            },
            "log": self.log,
        }

    # ------------------------------------------------------------------
    def _mr_signals(self, panels: dict[str, DataFrame]) -> dict:
        """OU pairs targets + router pairs_gate per ratio (causal, precomputed)."""
        cfg = self.cfg
        params = PairsParams(
            est_window_bars=cfg.mr_est_window, z_entry=cfg.mr_z_entry,
            z_exit=cfg.mr_z_exit, ou_gate=True, ou_min_sample=cfg.mr_ou_min_sample,
            max_hold_bars=cfg.mr_max_hold,
        )
        out = {}
        for a, b in MR_RATIOS:
            m = log_ratio_series(panels[a], panels[b])
            ou_arr = np.zeros(len(m), dtype=bool)
            step, win = 42, params.est_window_bars
            for t in range(win, len(m), step):
                seg = m["spread"].iloc[t - win : t]
                mdl = fit_ou(
                    seg.reset_index(drop=True),
                    OUParams(min_sample=params.ou_min_sample,
                             rolling_window=min(600, win // 2)),
                )
                ou_arr[t : t + step] = mdl.valid
            res = generate_signals(m, params, ou_valid_arr=ou_arr)
            gate = self.routers[a].pairs_gate[: len(m)]
            out[(a, b)] = {"target": res.target, "gate": gate}
        return out
