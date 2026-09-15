"""Slow-trend directional runner - the production-path twin of the ONE validated
directional variant (``ts_trend_4h_d60_sma200_ls``, DIRECTIONAL_ACCEPTANCE_REPORT.md).

The research harness (``research_lib/strategies/directional.py``) proved the edge on
the 5-coin perp panel and passed all 5 acceptance gates. This runner executes THE SAME
trade through the production chain every order must take in live:

    OrderIntent -> CentralRiskEngine -> ExecutionEngine (idempotency + order state
    machine) -> PaperBroker/LiveBroker -> PositionReconciler

so live-vs-backtest divergence is measurable before anything real is deployed (the
test-suite cross-checks this runner against ``EventBacktester`` on identical signals).

Signal (all futures data, causal): 1h perp -> 4h bars -> Donchian 60/30 breakout,
long above the 1200-bar SMA / short below (regime filter) -> decided at close t,
filled at open t+1. Position sized at 1% equity risk against a fixed 2xATR(20) stop,
capped by the leverage limit - the same contract the risk engine enforces live.
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
from research_lib.features.engine import atr
from research_lib.risk.costs import CostModel
from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits
from research_lib.strategies.directional import (
    apply_regime_filter,
    donchian_target,
    resample_panel,
)


@dataclass
class DirectionalRunnerConfig:
    symbol: str = "BTC/USDT"
    equity: float = 100_000.0
    leverage: float = 1.0        # the validated variant ran at max_leverage=1.0
    risk_per_trade: float = 0.01
    donchian_entry: int = 60
    donchian_exit: int = 30
    sma_bars: int = 1200           # 200 days of 4h bars
    atr_n: int = 20
    stop_mult: float = 2.0
    post_only: bool = False        # backtest parity is measured at taker cost
    recon_every_bars: int | None = 168  # ~4 weeks of 4h bars (None -> off)


# The risk engine's margin check compares the new order's notional against
# equity - total - 5% reserve, where `total` ALREADY includes the new order -
# so any single order is effectively capped near 0.475x equity (same ceiling the
# carry runner documented as 3N <= 0.95E). Size below it instead of being rejected.
MARGIN_SAFE_NOTIONAL = 0.45


def build_trend_signals(df_4h: DataFrame, cfg: DirectionalRunnerConfig) -> tuple[np.ndarray, np.ndarray]:
    """(target, stop) arrays for the validated variant, shifted to the fill convention.

    target[t] is the position held from open t (decision at close t-1). stop[t] is the
    protective stop for the position held during bar t (fixed at 2xATR from the entry
    fill reference, NaN when flat) - shared verbatim with the research backtester so
    divergence is attributable to execution, not signal drift.
    """
    raw = donchian_target(df_4h["close"], cfg.donchian_entry, cfg.donchian_exit, allow_short=True)
    filtered = apply_regime_filter(raw, df_4h["close"], cfg.sma_bars, "long_short")
    target = np.roll(filtered, 1)
    target[0] = 0.0

    a = atr(df_4h, cfg.atr_n).to_numpy(dtype=float)
    opens = df_4h["open"].to_numpy(dtype=float)
    stop = np.full(len(df_4h), np.nan)
    side = 0
    for t in range(len(df_4h)):
        d = int(target[t])
        if d != side and d != 0 and np.isfinite(a[t]):
            stop[t] = opens[t] - d * cfg.stop_mult * a[t]
            side = d
        elif d == 0:
            side = 0
        elif side != 0:
            stop[t] = stop[t - 1]
    return target, stop


class DirectionalRunner:
    """Run the slow-trend perp trade through the production execution chain."""

    def __init__(self, cfg: DirectionalRunnerConfig, exec_cfg: ExecutionConfig | None = None):
        self.cfg = cfg
        ec = exec_cfg or ExecutionConfig()
        self.cost = CostModel(
            maker_fee=ec.maker_fee,
            taker_fee=ec.taker_fee,
            spread_bps=ec.spread_bps,
            slippage_bps=ec.slippage_bps,
        )
        self.broker = PaperBroker(ec)
        self.risk = CentralRiskEngine(
            limits=RiskLimits(
                per_trade_risk=cfg.risk_per_trade,
                per_symbol_notional=cfg.leverage,
                portfolio_notional=cfg.leverage,
                max_leverage=cfg.leverage,
                max_position_notional=cfg.leverage,
                daily_loss_limit=1.0,   # strategy risk is stop-governed; kill switches
                weekly_loss_limit=1.0,  # stay armed via drawdown/reconciliation instead
                max_drawdown_kill=0.50,
                max_strategy_fraction=cfg.leverage,
            )
        )
        self.risk.state.equity = cfg.equity
        self.risk.state.peak_equity = cfg.equity
        self.risk.state.day_start_equity = cfg.equity
        self.execution = ExecutionEngine(
            risk=self.risk, store=OrderStateStore(), broker=self.broker,
            model_version="directional-trend-v1",
        )
        self.reconciler = PositionReconciler(self.risk)
        self.log: list[dict] = []

    # ------------------------------------------------------------------
    def _intent(self, side: int, units: float, price: float, stop: float | None, when: str, post_only: bool) -> OrderIntent:
        return OrderIntent(
            strategy_id="directional_trend",
            symbol=self.cfg.symbol,
            signal_time=when,
            side=side,
            units=units,
            entry_ref_price=price,
            stop_price=stop,
            notional=units * price,
            post_only=post_only,
        )

    def _execute(self, intent: OrderIntent, ref_price: float) -> dict[str, Any] | None:
        self.broker.set_market(ref_price)
        res = self.execution.execute(intent)
        if res.submitted and res.broker_ack and res.broker_ack.get("status") == "filled":
            return res.broker_ack
        self.log.append(
            {"action": "order_not_filled", "side": intent.side, "message": res.message,
             "time": intent.signal_time}
        )
        return None

    # ------------------------------------------------------------------
    def run(self, perp_1h: DataFrame) -> dict:
        """Replay the trend trade over 1h perp history via the production chain."""
        cfg = self.cfg
        df = resample_panel(perp_1h.sort_values("date").reset_index(drop=True), "4h")
        if len(df) < cfg.sma_bars + cfg.donchian_entry + 10:
            raise ValueError(f"need > {cfg.sma_bars + cfg.donchian_entry + 10} 4h bars, got {len(df)}")
        target, stop = build_trend_signals(df, cfg)

        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        dates = df["date"].to_numpy()

        side = 0
        units = 0.0
        entry_price = 0.0
        open_fee = 0.0          # entry fee is accounted in the trade's pnl at close,
        stop_level = np.nan     # so equity matches the research model's per-trade netting
        cooldown = False  # stopped out: no re-entry until the target returns to flat
        fees_total = 0.0
        trades: list[dict] = []
        curve: list[dict] = []

        def close_position(price: float, t: int, reason: str) -> bool:
            nonlocal side, units, entry_price, open_fee, fees_total
            when = str(pd.Timestamp(dates[t]))
            ack = self._execute(
                self._intent(-side, units, price, None, when, post_only=False), price
            )
            if ack is None:
                self.log.append({"action": "close_failed_abort", "time": when, "reason": reason})
                return False
            exit_price = float(ack["avg_price"])
            fee = float(ack.get("fee", 0.0))
            fees_total += fee
            pnl = side * (exit_price - entry_price) * units - (open_fee + fee)
            self.risk.on_position_closed(cfg.symbol, pnl)
            trades.append(
                {"exit_time": dates[t], "side": side, "entry_price": entry_price,
                 "exit_price": exit_price, "units": units, "pnl_net": pnl,
                 "exit_reason": reason}
            )
            side, units = 0, 0.0
            return True

        for t in range(len(df)):
            desired = int(target[t])
            when = str(pd.Timestamp(dates[t]))
            if desired == 0:
                cooldown = False
            # (a) target transitions act at the open (decision was close t-1)
            if desired != side and not (cooldown and side == 0):
                if side != 0 and not close_position(opens[t], t, "signal"):
                    break
                if desired != 0:
                    s = stop[t]
                    if not np.isfinite(s) or abs(opens[t] - s) <= 0:
                        self.log.append({"action": "entry_skipped_no_stop", "time": when})
                    else:
                        # epsilon under the risk budget: float rounding on
                        # |entry-stop|*units otherwise trips the engine's strict
                        # per-trade-risk check at exactly 1.0x the limit
                        risk_units = (
                            self.risk.state.equity * cfg.risk_per_trade * (1.0 - 1e-9)
                            / abs(opens[t] - s)
                        )
                        cap = min(cfg.leverage, MARGIN_SAFE_NOTIONAL)
                        lev_units = self.risk.state.equity * cap / opens[t]
                        u = float(min(risk_units, lev_units))
                        ack = self._execute(
                            self._intent(desired, u, opens[t], float(s), when, cfg.post_only),
                            opens[t],
                        )
                        if ack is not None:
                            side = desired
                            units = float(ack["filled"])
                            entry_price = float(ack["avg_price"])
                            stop_level = float(s)
                            open_fee = float(ack.get("fee", 0.0))
                            fees_total += open_fee
                            self.risk.on_position_opened(
                                self._intent(side, units, entry_price, float(s), when, False)
                            )
            elif cooldown and side == 0:
                pass  # target still mid-episode after a stop-out: stay flat
            # (b) intra-bar protective stop on the held position (open-through-level
            # fills at the open, matching the EventBacktester)
            if side == 1 and lows[t] <= stop_level:
                px = opens[t] if opens[t] <= stop_level else stop_level
                if not close_position(px, t, "stop"):
                    break
                cooldown = True
            elif side == -1 and highs[t] >= stop_level:
                px = opens[t] if opens[t] >= stop_level else stop_level
                if not close_position(px, t, "stop"):
                    break
                cooldown = True
            # mark-to-market equity curve (risk state equity updates on closes only)
            mtm = float(self.risk.state.equity) + (
                side * (closes[t] - entry_price) * units if side != 0 else 0.0
            )
            curve.append({"date": dates[t], "equity": mtm})

            if cfg.recon_every_bars and (t + 1) % cfg.recon_every_bars == 0:
                internal = {cfg.symbol: side * units} if side != 0 else {}
                rr = self.reconciler.reconcile(internal, self.broker.positions())
                if rr.consistent:
                    self.reconciler.resume()
                else:
                    self.log.append(
                        {"action": "reconciliation_failed", "diffs": rr.diffs, "time": when}
                    )

        # end of data: flatten through the same chain - the book must end flat
        if side != 0:
            close_position(closes[len(df) - 1], len(df) - 1, "end_of_data")

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
            "book_flat": side == 0,
            "log": self.log,
        }
