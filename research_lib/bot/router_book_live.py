"""Live loop for the router book (config D): replay-to-target + position diff.

Architecture (4h-bar book - REST polling is sufficient, no websocket needed):

    every poll:
      1. PerpDataFeed appends new CLOSED 1h bars to the local feather panels
      2. if at least one new 4h bar closed:
         a. (re)fit the causal HMM routers at most every ROUTER_REFIT_HOURS
         b. replay the full history through RouterBookRunner
            (flatten_at_end=False) -> the book's TARGET positions right now
         c. diff target vs broker-reported positions -> delta OrderIntents
            through CentralRiskEngine -> ExecutionEngine -> broker
         d. PositionReconciler verifies bot state == broker state;
            divergence trips the risk kill switch (no new orders)

Replay-to-target makes the loop STATELESS: restart safety comes free (any
crash/restart simply replays again and diffs), and every order still takes
the same production chain as the backtest. Paper mode uses PaperBroker; live
mode uses LiveBroker and refuses to start without --yes and API credentials.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.bot.broker import PaperBroker
from research_lib.bot.config import ExecutionConfig
from research_lib.bot.live_feed import PerpDataFeed
from research_lib.bot.router_book_runner import (
    ROUTER_CFG_4H,
    TREND_SYMS,
    RouterBookConfig,
    RouterBookRunner,
)
from research_lib.execution.engine import ExecutionEngine
from research_lib.execution.order_state import OrderStateStore
from research_lib.execution.reconciliation import PositionReconciler
from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits

logger = logging.getLogger(__name__)

ROUTER_REFIT_HOURS = 24
MIN_ORDER_NOTIONAL = 5.0  # exchange dust threshold (USDT)


@dataclass
class CycleReport:
    new_bars: dict[str, int] = field(default_factory=dict)
    new_4h_bar: bool = False
    router_refit: bool = False
    target: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    orders_filled: int = 0
    orders_failed: int = 0
    reconciliation_ok: bool = True
    kill_switch: bool = False
    note: str = ""


class RouterBookLive:
    """Drive config D against the live market (paper or real broker)."""

    def __init__(
        self,
        cfg: RouterBookConfig,
        data_dir: Path,
        exec_cfg: ExecutionConfig | None = None,
        broker=None,
        feed: PerpDataFeed | None = None,
        meta_signals=None,
        funding_map: dict | None = None,
    ):
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        ec = exec_cfg or ExecutionConfig()
        self.broker = broker if broker is not None else PaperBroker(ec)
        self.feed = feed  # None -> constructed lazily (needs network)
        self.meta_signals = meta_signals
        self.funding_map = funding_map
        self.risk = CentralRiskEngine(
            limits=RiskLimits(
                per_trade_risk=cfg.risk_per_trade,
                per_symbol_notional=cfg.leverage,
                portfolio_notional=4.0,
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
            model_version="router-book-live-D-v1",
        )
        self.reconciler = PositionReconciler(self.risk)
        self._routers = None
        self._last_router_refit: pd.Timestamp | None = None
        self._last_4h_close: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    def _feather(self, sym: str) -> Path:
        return self.data_dir / "futures" / f"{sym}_USDT-1h-perp.feather"

    def _load_panels(self) -> dict[str, DataFrame]:
        return {s: pd.read_feather(self._feather(s)) for s in TREND_SYMS}

    def _latest_mark(self, sym: str) -> float:
        df = pd.read_feather(self._feather(sym))
        return float(df["close"].iloc[-1])

    def _refit_routers_if_due(self, panels_4h: dict[str, DataFrame]) -> bool:
        from research_lib.strategies.unified_router import filtered_hmm_router

        now = pd.Timestamp.now(tz="UTC")
        due = (
            self._routers is None
            or self._last_router_refit is None
            or (now - self._last_router_refit) >= pd.Timedelta(hours=ROUTER_REFIT_HOURS)
        )
        if not due:
            return False
        self._routers = {
            s: filtered_hmm_router(panels_4h[s], ROUTER_CFG_4H, seed=42) for s in TREND_SYMS
        }
        self._last_router_refit = now
        return True

    def _derive_target(self) -> tuple[dict[str, float], bool]:
        """Replay the full history (flatten_at_end=False) -> (target positions, refit)."""
        from research_lib.strategies.directional import resample_panel

        perp = self._load_panels()
        panels_4h = {s: resample_panel(df, "4h") for s, df in perp.items()}
        refit = self._refit_routers_if_due(panels_4h)
        live_cfg = RouterBookConfig(**{**self.cfg.__dict__, "flatten_at_end": False})
        runner = RouterBookRunner(
            live_cfg, self._routers, meta_signals=self.meta_signals,
            funding_map=self.funding_map,
        )
        out = runner.run(perp)
        self._last_replay = out
        return out["final_positions"], refit

    def _diff_orders(self, target: dict[str, float], when: str) -> tuple[dict[str, float], int, int]:
        """Issue delta orders so broker positions match the target."""
        current = self.broker.positions()
        deltas: dict[str, float] = {}
        filled = failed = 0
        for sym in sorted(set(target) | set(current)):
            mark = self._latest_mark(sym.split("/")[0])
            d = float(target.get(sym, 0.0)) - float(current.get(sym, 0.0))
            if abs(d) * mark < MIN_ORDER_NOTIONAL:
                continue
            deltas[sym] = d
            intent = OrderIntent(
                strategy_id="router_book_live", symbol=sym, signal_time=when,
                side=1 if d > 0 else -1, units=abs(d), entry_ref_price=mark,
                stop_price=None, notional=abs(d) * mark, post_only=False,
            )
            if hasattr(self.broker, "set_market"):
                self.broker.set_market(mark)  # PaperBroker only; LiveBroker quotes itself
            res = self.execution.execute(intent)
            if res.submitted and res.broker_ack and res.broker_ack.get("status") == "filled":
                filled += 1
            else:
                failed += 1
                logger.warning("delta order not filled: %s %+.6f (%s)", sym, d, res.message)
        return deltas, filled, failed

    # ------------------------------------------------------------------
    def cycle(self) -> CycleReport:
        """One poll cycle: refresh data, and if a new 4h bar closed, re-target."""
        rep = CycleReport()
        if self.feed is None:
            self.feed = PerpDataFeed()
        for sym in TREND_SYMS:
            try:
                r = self.feed.update_panel(sym, self._feather(sym))
                rep.new_bars[sym] = r.new_bars
            except Exception as e:  # noqa: BLE001 - one bad symbol must not kill the loop
                logger.warning("feed update failed for %s: %s", sym, e)
                rep.new_bars[sym] = 0

        latest = max(pd.to_datetime(pd.read_feather(self._feather(s))["date"]).max()
                     for s in TREND_SYMS)
        latest_4h_close = latest.floor("4h")
        if self._last_4h_close is not None and latest_4h_close <= self._last_4h_close:
            rep.note = "no new 4h bar"
            return rep
        self._last_4h_close = latest_4h_close
        rep.new_4h_bar = True

        when = str(pd.Timestamp.now(tz="UTC"))
        target, rep.router_refit = self._derive_target()
        rep.target = target
        deltas, rep.orders_filled, rep.orders_failed = self._diff_orders(target, when)
        rep.deltas = deltas
        rr = self.reconciler.reconcile(target, self.broker.positions())
        rep.reconciliation_ok = rr.consistent
        if rr.consistent:
            self.reconciler.resume()
        else:
            logger.error("reconciliation divergence: %s", rr.diffs)
        rep.kill_switch = self.risk.state.kill_switch
        return rep

    def run(self, poll_seconds: float = 300.0, max_cycles: int | None = None) -> None:
        """Continuous live loop (Ctrl-C to stop)."""
        n = 0
        logger.info("router-book live loop starting (poll=%ss, broker=%s)",
                    poll_seconds, type(self.broker).__name__)
        try:
            while max_cycles is None or n < max_cycles:
                rep = self.cycle()
                n += 1
                logger.info(
                    "cycle %d: new_bars=%s 4h=%s refit=%s target=%d deltas=%d filled=%d failed=%d recon=%s kill=%s",
                    n, sum(rep.new_bars.values()), rep.new_4h_bar, rep.router_refit,
                    len(rep.target), len(rep.deltas), rep.orders_filled,
                    rep.orders_failed, rep.reconciliation_ok, rep.kill_switch,
                )
                if rep.kill_switch:
                    logger.error("kill switch engaged - stopping (manual review required)")
                    break
                if max_cycles is None or n < max_cycles:
                    time.sleep(poll_seconds)
        except KeyboardInterrupt:
            logger.info("stopped by user after %d cycles", n)
