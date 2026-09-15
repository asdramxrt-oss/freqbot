"""Delta-neutral funding-carry runner - the production-path twin of strategies/carry.py.

The research backtest (``run_carry``) proved the edge and passed the acceptance gates
(research/reports/CARRY_ACCEPTANCE_REPORT.md). This runner executes THE SAME trade
through the production chain every order must take in live:

    OrderIntent -> CentralRiskEngine -> ExecutionEngine (idempotency + order state
    machine) -> PaperBroker/LiveBroker -> PositionReconciler

so live-vs-backtest divergence is measurable before anything real is deployed (the
test-suite cross-checks this runner against ``run_carry`` on identical data).

Legs are keyed as separate symbols so risk state, broker positions and reconciliation
never mix them: ``BTC/USDT`` (spot, long) and ``BTC/USDT:PERP`` (perp, short). Funding
is settled through ``CentralRiskEngine.on_funding_payment`` at each 8h print. The short
leg is re-hedged before liquidation (close both legs, reopen at the mark), exactly like
the research model. Known limitation, same as the research path: until a real perp
price series lands, the perp mark is spot * (1 + basis_bps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
from research_lib.strategies.carry import FUNDING_PERIODS_PER_YEAR  # noqa: F401  (re-export)


@dataclass
class CarryRunnerConfig:
    symbol: str = "BTC/USDT"
    equity: float = 100_000.0
    leverage: float = 1.0
    notional_fraction: float = 0.9  # keeps the risk engine's 5% reserve check satisfied
    basis_bps: float = 0.0
    post_only: bool = True
    rebalance_trigger: float = 0.8
    liquidation_buffer: float = 1.0
    recon_every_prints: int | None = 30  # reconciliations per N funding prints (None -> off)


@dataclass
class _Leg:
    symbol: str
    side: int
    units: float = 0.0
    entry_price: float = 0.0
    fees_paid: float = 0.0

    @property
    def open(self) -> bool:
        return self.units > 0.0


class CarryRunner:
    """Run the funding-carry trade through the production execution chain."""

    def __init__(self, cfg: CarryRunnerConfig, exec_cfg: ExecutionConfig | None = None):
        self.cfg = cfg
        ec = exec_cfg or ExecutionConfig()
        self.cost = CostModel(
            maker_fee=ec.maker_fee,
            taker_fee=ec.taker_fee,
            spread_bps=ec.spread_bps,
            slippage_bps=ec.slippage_bps,
        )
        self.broker = PaperBroker(ec)
        # carry-appropriate limits: gross two-leg notional with net delta ~0
        leg_lev = cfg.notional_fraction / (1.0 + 1.0 / cfg.leverage)
        head = leg_lev * 1.02
        self.risk = CentralRiskEngine(
            limits=RiskLimits(
                per_trade_risk=1.0,  # no stops on carry legs - margin model governs
                per_symbol_notional=head,
                portfolio_notional=2.0 * head,
                max_leverage=max(1.0, head),
                max_position_notional=head,
                daily_loss_limit=1.0,
                weekly_loss_limit=1.0,
                max_drawdown_kill=1.0,
                max_strategy_fraction=2.0 * head,
            )
        )
        self.risk.state.equity = cfg.equity
        self.risk.state.peak_equity = cfg.equity
        self.risk.state.day_start_equity = cfg.equity
        self.execution = ExecutionEngine(
            risk=self.risk, store=OrderStateStore(), broker=self.broker, model_version="carry-v1"
        )
        self.reconciler = PositionReconciler(self.risk)
        self.log: list[dict] = []
        # The risk engine's margin-availability check counts gross open notional (both
        # legs) plus a 5% reserve, so N per leg must satisfy 3N <= 0.95E. Cap the
        # configured fraction accordingly instead of failing the second leg at entry.
        self._eff_fraction = min(
            cfg.notional_fraction, 0.3166 * (1.0 + 1.0 / cfg.leverage) * 0.99
        )

    @property
    def perp_symbol(self) -> str:
        return f"{self.cfg.symbol}:PERP"

    # ------------------------------------------------------------------
    def _intent(self, leg: _Leg, side: int, units: float, price: float, when: str, post_only: bool) -> OrderIntent:
        return OrderIntent(
            strategy_id=f"carry_{'spot' if leg.side > 0 else 'perp'}",
            symbol=leg.symbol,
            signal_time=when,
            side=side,
            units=units,
            entry_ref_price=price,
            stop_price=None,
            notional=units * price,
            post_only=post_only,
        )

    def _execute(self, intent: OrderIntent, ref_price: float) -> dict[str, Any] | None:
        """Submit through risk + execution; return the broker ack on a full fill."""
        self.broker.set_market(ref_price)
        res = self.execution.execute(intent)
        if res.submitted and res.broker_ack and res.broker_ack.get("status") == "filled":
            return res.broker_ack
        self.log.append(
            {"action": "order_not_filled", "symbol": intent.symbol, "side": intent.side,
             "message": res.message, "time": intent.signal_time}
        )
        return None

    def _open_legs(self, spot_price: float, perp_price: float, when: str) -> bool:
        """Enter long spot + short perp. If the second leg fails, unwind the first."""
        committed = self.risk.state.equity * self._eff_fraction
        notional = committed / (1.0 + 1.0 / self.cfg.leverage)
        units = notional / spot_price
        legs = (
            _Leg(self.cfg.symbol, +1),
            _Leg(self.perp_symbol, -1),
        )
        acks: list[dict | None] = []
        for leg in legs:
            ref = spot_price if leg.side > 0 else perp_price
            ack = self._execute(self._intent(leg, leg.side, units, ref, when, self.cfg.post_only), ref)
            acks.append(ack)
            if ack is None:
                break
        if acks[0] is not None and acks[1] is None:
            # never run a naked leg: unwind the filled one at taker immediately
            self._execute(
                self._intent(legs[0], -legs[0].side, float(acks[0]["filled"]), spot_price, when, post_only=False),
                spot_price,
            )
            self.log.append({"action": "entry_aborted_second_leg_failed", "time": when})
            return False
        if acks[0] is None:
            return False
        for leg, ack in zip(legs, acks, strict=True):
            leg.units = float(ack["filled"])
            leg.entry_price = float(ack["avg_price"])
            leg.fees_paid += float(ack.get("fee", 0.0))
            self.risk.on_position_opened(self._intent(leg, leg.side, leg.units, leg.entry_price, when, False))
        self._spot, self._perp = legs
        return True

    def _close_legs(self, spot_price: float, perp_price: float, when: str, reason: str) -> float | None:
        """Close both legs at taker; return realised pnl (delta - fees, funding excluded).

        A close that is not confirmed filled leaves the leg OPEN and aborts the run
        (returns None) - silently marking an unclosed leg as closed is exactly the kind
        of bot/broker divergence reconciliation exists to catch, so it must never happen
        here first.
        """
        pnl = 0.0
        for leg in (self._spot, self._perp):
            if not leg.open:
                continue
            ref = spot_price if leg.side > 0 else perp_price
            ack = self._execute(self._intent(leg, -leg.side, leg.units, ref, when, post_only=False), ref)
            if ack is None:
                self.log.append(
                    {"action": "close_failed", "symbol": leg.symbol, "reason": reason, "time": when}
                )
                return None
            exit_price = float(ack["avg_price"])
            leg.fees_paid += float(ack.get("fee", 0.0))
            leg_pnl = leg.side * (exit_price - leg.entry_price) * leg.units - leg.fees_paid
            pnl += leg_pnl
            self.risk.on_position_closed(leg.symbol, leg_pnl)
            leg.units = 0.0
        self.log.append({"action": "closed", "reason": reason, "price": spot_price, "time": when, "pnl": pnl})
        return pnl

    # ------------------------------------------------------------------
    def run(self, spot: DataFrame, funding: DataFrame, perp: DataFrame | None = None) -> dict:
        """Replay the carry trade over spot/funding history via the production chain.

        ``perp`` (optional): the real perpetual-futures series (e.g.
        research/data/futures/{SYM}_USDT-1h-perp.feather). When given, the short leg is
        priced on it; otherwise the spot * (1 + basis_bps) proxy is used.
        """
        cfg = self.cfg
        df = spot.sort_values("date").reset_index(drop=True).copy()
        f = funding.sort_values("date").reset_index(drop=True).copy()
        frames = [df, f] + ([perp] if perp is not None else [])
        for frame in frames:
            frame["date"] = (
                pd.to_datetime(frame["date"], utc=True, format="mixed")
                .dt.tz_localize(None)
                .astype("datetime64[ns]")
            )
        if perp is not None:
            pp = perp.sort_values("date").reset_index(drop=True)
        else:
            pp = None
        if df.empty or f.empty:
            raise ValueError("spot and funding frames must both be non-empty")
        dates = df["date"].to_numpy(dtype="datetime64[ns]")
        closes = df["close"].to_numpy(float)
        basis = cfg.basis_bps / 1e4
        p_dates = pp["date"].to_numpy(dtype="datetime64[ns]") if pp is not None else None
        p_closes = pp["close"].to_numpy(float) if pp is not None else None

        # the run ends where the SHORTEST series ends - past that, one leg would be
        # marked on a frozen price and the position would be naked, not hedged
        end_t64 = dates[-1] if p_dates is None or len(p_dates) == 0 else min(dates[-1], p_dates[-1])

        def perp_mark_at(t64) -> float | None:
            if p_dates is None or len(p_dates) == 0:
                return None
            k = int(np.searchsorted(p_dates, t64, side="right")) - 1
            return float(p_closes[k]) if k >= 0 else None

        def perp_price(t64, spot_ref: float) -> float:
            real = perp_mark_at(t64)
            return real if real is not None else spot_ref * (1.0 + basis)

        self._spot = _Leg(cfg.symbol, +1)
        self._perp = _Leg(self.perp_symbol, -1)
        margin = 0.0
        liquidated = False
        n_rehedges = 0
        gross_funding = 0.0
        fees_at_start = self.broker.total_fees()
        curve: list[dict] = []
        entered = False

        for row in f.itertuples():
            if np.datetime64(row.date.asm8) > end_t64:
                break  # one leg is out of data - the hedge no longer exists
            j = int(np.searchsorted(dates, np.datetime64(row.date.asm8), side="right")) - 1
            if j < 0:
                continue
            mark = closes[j]
            mark_perp = perp_price(np.datetime64(row.date.asm8), mark)
            when = str(row.date)
            if not entered:
                entered = self._open_legs(mark, mark_perp, when)
                if entered:
                    margin = (self._spot.units * self._spot.entry_price) / cfg.leverage
                    self.log.append({"action": "entered", "price": mark, "time": when})
                else:
                    break
                continue
            # funding settles on the perp notional at the mark (short receives when > 0)
            payment = float(row.funding_rate) * self._perp.units * mark_perp
            gross_funding += payment
            self.risk.on_funding_payment(self.perp_symbol, payment)
            short_loss = self._perp.units * (mark_perp - self._perp.entry_price)
            curve.append(
                {"date": row.date, "payment": payment, "equity": float(self.risk.state.equity)}
            )
            if short_loss >= margin * cfg.liquidation_buffer:
                if self._close_legs(mark, mark_perp, when, "liquidation") is None:
                    break
                liquidated = True
                break
            if short_loss >= margin * cfg.liquidation_buffer * cfg.rebalance_trigger:
                if self._close_legs(mark, mark_perp, when, "rehedge") is None:
                    break
                n_rehedges += 1
                if self.risk.state.equity * self._eff_fraction / (1 + 1 / cfg.leverage) < margin * 0.5:
                    liquidated = True  # equity can no longer support the margin
                    break
                if not self._open_legs(mark, mark_perp, when):
                    liquidated = True
                    break
            # periodic reconciliation against broker-reported positions
            if cfg.recon_every_prints and len(curve) % cfg.recon_every_prints == 0:
                internal = {}
                if self._spot.open:
                    internal[self._spot.symbol] = self._spot.units
                if self._perp.open:
                    internal[self._perp.symbol] = -self._perp.units
                rr = self.reconciler.reconcile(internal, self.broker.positions())
                if rr.consistent:
                    self.reconciler.resume()
                else:
                    self.log.append({"action": "reconciliation_failed", "diffs": rr.diffs, "time": when})

        if self._spot.open or self._perp.open:
            end_i = int(np.searchsorted(dates, end_t64, side="right")) - 1
            final_mark_perp = perp_price(dates[end_i], closes[end_i])
            self._close_legs(closes[end_i], final_mark_perp, str(pd.Timestamp(dates[end_i])), "end_of_data")

        # after a well-formed run the book must be flat - anything else is a bug,
        # not a result
        book_flat = not (self._spot.open or self._perp.open)
        final_equity = float(self.risk.state.equity)
        fees = self.broker.total_fees() - fees_at_start
        span_days = max(
            (pd.Timestamp(end_t64) - pd.Timestamp(dates[0])).total_seconds() / 86400.0, 1.0
        )
        growth = max(final_equity / cfg.equity, 1e-9)
        return {
            "equity_curve": DataFrame(curve),
            "net_pnl": final_equity - cfg.equity,
            "final_equity": final_equity,
            "gross_funding": gross_funding,
            "total_fees": fees,
            "liquidated": liquidated,
            "n_rehedges": n_rehedges,
            "annualised_return": growth ** (365.0 / span_days) - 1.0,
            "kill_switch": self.risk.state.kill_switch,
            "reconciliation_ok": self.risk.state.reconciliation_ok,
            "book_flat": book_flat,
            "log": self.log,
        }
