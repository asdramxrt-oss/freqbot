"""Broker adapters (PDF: Execution Engine -> Redis Broker / Lock Adapter).

The execution engine only knows the ``BrokerAdapter`` protocol (``submit`` / ``fetch`` /
``cancel``). This module provides two implementations:

* ``PaperBroker`` - a deterministic in-memory broker that fills at the *next bar open*
  with the project's ``CostModel`` (fee + half-spread + slippage). This is the same fill
  model the research backtests use, so paper results are directly comparable.
* ``LiveBroker`` - a thin ccxt adapter for real exchange orders. It is deliberately
  conservative: it never fabricates a fill, returns ``status='open'`` on submit, and
  requires ``fetch`` to confirm. It raises on missing credentials rather than silently
  degrading to paper.

Both are constructed from the bot's ``ExecutionConfig`` so cost/fee assumptions match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from research_lib.bot.config import ExecutionConfig
from research_lib.risk.costs import CostModel
from research_lib.risk.engine import OrderIntent


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------
@dataclass
class _PaperFill:
    broker_order_id: str
    symbol: str
    side: int
    units: float
    price: float
    fee: float


@dataclass
class _RestingOrder:
    """A post-only limit order working on the book (paper)."""

    broker_order_id: str
    symbol: str
    side: int
    units: float
    limit_price: float


class PaperBroker:
    """Simulated broker: taker orders fill at the next bar open, adverse to the trader.

    The bot sets ``set_market(price, high=..., low=...)`` before each ``execute`` call; the
    broker applies the cost model to that reference price. This keeps the broker stateless
    about time and lets the caller control exactly which bar the fill belongs to
    (next-open semantics).

    Post-only (maker) orders rest at the quoted limit price (``CostModel.maker_quote``):
    they fill at the quote with the maker fee - no slippage, no spread cost - but ONLY if
    the bar trades through the quote (``low <= limit`` for a buy, ``high >= limit`` for a
    sell). If the caller does not provide high/low, the quote is assumed reachable and the
    order fills at the limit price. Unfilled orders stay on the book until ``fetch``
    observes a trade-through or ``cancel`` removes them - mirroring live post-only
    behaviour, where saving the spread costs you fill certainty.
    """

    def __init__(self, cfg: ExecutionConfig | None = None):
        cfg = cfg or ExecutionConfig()
        self.cost = CostModel(
            maker_fee=cfg.maker_fee,
            taker_fee=cfg.taker_fee,
            spread_bps=cfg.spread_bps,
            slippage_bps=cfg.slippage_bps,
        )
        self._market_price: float | None = None
        self._market_high: float | None = None
        self._market_low: float | None = None
        self.fills: list[_PaperFill] = []
        self.open_orders: dict[str, _RestingOrder] = {}
        self.cash: float = cfg.equity
        self._seq = 0

    # -- market state ---------------------------------------------------
    def set_market(
        self,
        price: float | None,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        self._market_price = None if price is None else float(price)
        self._market_high = None if high is None else float(high)
        self._market_low = None if low is None else float(low)

    def _trades_through(self, order: _RestingOrder) -> bool:
        """Did the current bar trade through the resting limit price?"""
        if order.side > 0:  # buy limit fills when the market trades down to it
            return self._market_low is not None and self._market_low <= order.limit_price
        return self._market_high is not None and self._market_high >= order.limit_price

    def _book_fill(
        self, oid: str, intent_symbol: str, side: int, units: float, price: float, fee_rate: float
    ) -> dict[str, Any]:
        notional = price * units
        fee = fee_rate * notional
        self.fills.append(
            _PaperFill(
                broker_order_id=oid,
                symbol=intent_symbol,
                side=side,
                units=units,
                price=price,
                fee=fee,
            )
        )
        # cash accounting: long pays out, short receives
        self.cash -= side * notional
        self.cash -= fee
        return {
            "order_id": oid,
            "status": "filled",
            "filled": units,
            "remaining": 0.0,
            "avg_price": price,
            "fee": fee,
        }

    # -- BrokerAdapter protocol -----------------------------------------
    def submit(self, intent: OrderIntent) -> dict[str, Any]:
        if self._market_price is None or not np.isfinite(self._market_price):
            return {"order_id": "", "status": "error", "reason": "no market price set"}
        if intent.units <= 0 or not np.isfinite(intent.units):
            return {"order_id": "", "status": "rejected", "reason": "non-positive units"}

        self._seq += 1
        oid = f"paper-{self._seq}"
        side_name = "long" if intent.side > 0 else "short"

        if getattr(intent, "post_only", False):
            limit = self.cost.maker_quote(self._market_price, side_name)  # type: ignore[arg-type]
            order = _RestingOrder(
                broker_order_id=oid,
                symbol=intent.symbol,
                side=intent.side,
                units=intent.units,
                limit_price=limit,
            )
            # fill only if the bar traded through the quote; without high/low the
            # reference price is all we have, so assume the quote was reachable
            reachable = (
                self._trades_through(order)
                if (self._market_high is not None or self._market_low is not None)
                else True
            )
            if not reachable:
                self.open_orders[oid] = order
                return {
                    "order_id": oid,
                    "status": "open",
                    "filled": 0.0,
                    "remaining": intent.units,
                    "limit_price": limit,
                }
            return self._book_fill(
                oid, intent.symbol, intent.side, intent.units, limit, self.cost.fee("maker")
            )

        # adverse taker fill relative to the reference market price
        price = self.cost.execution_price(self._market_price, side_name)  # type: ignore[arg-type]
        return self._book_fill(
            oid, intent.symbol, intent.side, intent.units, price, self.cost.fee()
        )

    def fetch(self, broker_order_id: str) -> dict[str, Any]:
        resting = self.open_orders.get(broker_order_id)
        if resting is not None:
            if self._trades_through(resting):
                del self.open_orders[broker_order_id]
                return self._book_fill(
                    resting.broker_order_id,
                    resting.symbol,
                    resting.side,
                    resting.units,
                    resting.limit_price,
                    self.cost.fee("maker"),
                )
            return {
                "order_id": broker_order_id,
                "status": "open",
                "filled": 0.0,
                "remaining": resting.units,
                "limit_price": resting.limit_price,
            }
        for f in self.fills:
            if f.broker_order_id == broker_order_id:
                return {
                    "order_id": broker_order_id,
                    "status": "filled",
                    "filled": f.units,
                    "remaining": 0.0,
                    "avg_price": f.price,
                    "fee": f.fee,
                }
        return {"order_id": broker_order_id, "status": "unknown", "filled": 0.0}

    def cancel(self, broker_order_id: str) -> dict[str, Any]:
        if broker_order_id in self.open_orders:
            del self.open_orders[broker_order_id]
            return {"order_id": broker_order_id, "status": "cancelled", "cancelled": True}
        # taker fills are immediate; nothing to cancel
        return {"order_id": broker_order_id, "status": "cancelled", "cancelled": True}

    # -- helpers ---------------------------------------------------------
    def total_fees(self) -> float:
        return float(sum(f.fee for f in self.fills))

    def positions(self) -> dict[str, float]:
        """Net signed units per symbol, derived from every booked fill."""
        pos: dict[str, float] = {}
        for f in self.fills:
            pos[f.symbol] = pos.get(f.symbol, 0.0) + f.side * f.units
        return {k: v for k, v in pos.items() if abs(v) > 0.0}

    def record_external_exit(self, symbol: str, side: int, units: float, price: float) -> None:
        """Book the closing half of a position whose exit did NOT flow through ``submit``.

        The bot loop closes positions on stops / time stops in its mark-to-market path
        (same convention as the research backtests). Without this record the broker's
        fill-derived ``positions()`` would keep showing the closed leg as open and
        reconciliation would flag a phantom divergence.
        """
        if units <= 0 or not np.isfinite(units):
            return
        self._seq += 1
        # the exit is the opposite side of the position; fee 0 - exit costs are owned
        # by the bot loop's own PnL accounting, the broker only mirrors quantities
        self._book_fill(f"paper-{self._seq}", symbol, -int(side), float(units), float(price), 0.0)


# ---------------------------------------------------------------------------
# Live broker (ccxt)
# ---------------------------------------------------------------------------
@dataclass
class LiveBroker:
    """Minimal ccxt-backed broker. Requires explicit credentials; never fabricates fills.

    ``exchange_id`` is a ccxt exchange name (e.g. "binance"). Taker orders are market
    orders with a fixed quantity - this bot's decision layer determines units, and the
    risk engine has already bounded them. When ``intent.post_only`` is set the order is
    submitted as a post-only limit at the top of book (bid for buys, ask for sells, from
    the live ticker); exchanges that cannot guarantee the maker flag reject it, which the
    execution engine records as a broker rejection rather than a taker fill.
    """

    exchange_id: str = "binance"
    market_type: str = "spot"             # spot | swap
    params: dict[str, Any] = field(default_factory=dict)
    _exchange: Any = None

    def __post_init__(self) -> None:
        try:
            import ccxt  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                "ccxt is required for live trading. Install it with `uv add ccxt`."
            ) from e
        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"Unknown ccxt exchange id: {self.exchange_id}")
        klass = getattr(ccxt, self.exchange_id)
        opts = dict(self.params)
        opts.setdefault("enableRateLimit", True)
        opts.setdefault("options", {})
        opts["options"].setdefault("defaultType", self.market_type)
        self._exchange = klass(opts)
        if not self._exchange.apiKey:
            raise RuntimeError(
                "Live mode requires API credentials. Set them in the exchange config "
                "(apiKey/secret) - refusing to start live trading without keys."
            )

    # -- BrokerAdapter protocol -----------------------------------------
    def submit(self, intent: OrderIntent) -> dict[str, Any]:
        symbol = intent.symbol.replace("/", "")
        side = "buy" if intent.side > 0 else "sell"
        try:
            if getattr(intent, "post_only", False):
                price = self._maker_price(symbol, side)
                if price is None:
                    return {"order_id": "", "status": "error", "reason": "no ticker for post-only quote"}
                order = self._exchange.create_order(
                    symbol, "limit", side, intent.units, price, {"postOnly": True}
                )
            else:
                order = self._exchange.create_order(symbol, "market", side, intent.units)
        except Exception as e:  # noqa: BLE001 - surface to execution engine
            return {"order_id": "", "status": "error", "reason": str(e)}
        return {
            "order_id": str(order.get("id", "")),
            "status": "open",
            "filled": float(order.get("filled") or 0.0),
            "remaining": float(order.get("remaining") or intent.units),
        }

    def _maker_price(self, ccxt_symbol: str, side: str) -> float | None:
        """Top-of-book quote for a post-only order: bid for buys, ask for sells."""
        try:
            ticker = self._exchange.fetch_ticker(ccxt_symbol)
        except Exception:  # noqa: BLE001
            return None
        price = ticker.get("bid") if side == "buy" else ticker.get("ask")
        return None if price is None else float(price)

    def fetch(self, broker_order_id: str) -> dict[str, Any]:
        try:
            o = self._exchange.fetch_order(broker_order_id)
        except Exception as e:  # noqa: BLE001
            return {"order_id": broker_order_id, "status": "unknown", "reason": str(e)}
        return {
            "order_id": broker_order_id,
            "status": o.get("status", "open"),
            "filled": float(o.get("filled") or 0.0),
            "remaining": float(o.get("remaining") or 0.0),
            "avg_price": o.get("average"),
        }

    def cancel(self, broker_order_id: str) -> dict[str, Any]:
        try:
            self._exchange.cancel_order(broker_order_id)
        except Exception as e:  # noqa: BLE001
            return {"order_id": broker_order_id, "status": "error", "reason": str(e)}
        return {"order_id": broker_order_id, "status": "cancelled", "cancelled": True}

    def positions(self) -> dict[str, float]:
        """Broker-reported net signed units per unified symbol (e.g. 'BTC/USDT').

        Spot: base-currency balances mapped back through the exchange's markets (long-only,
        so all quantities are positive). Swap/futures: ``fetch_positions`` contracts with
        their signed amount. Any failure returns an empty dict - reconciliation treats that
        as a divergence and halts new orders, which is the safe direction.
        """
        out: dict[str, float] = {}
        try:
            if self.market_type == "swap":
                for p in self._exchange.fetch_positions() or []:
                    qty = float(p.get("contracts") or 0.0)
                    sign = -1.0 if p.get("side") == "short" else 1.0
                    if qty:
                        out[p["symbol"]] = sign * qty
                return out
            bal = self._exchange.fetch_balance()
            markets = self._exchange.load_markets()
            totals = bal.get("total") or {}
            for currency, amount in totals.items():
                if not amount:
                    continue
                for sym, m in markets.items():
                    if m.get("base") == currency and m.get("quote") in ("USDT", "USDC", "USD"):
                        out[sym] = out.get(sym, 0.0) + float(amount)
                        break
        except Exception:  # noqa: BLE001 - empty dict => reconciliation halts (safe)
            return {}
        return out


def make_broker(cfg: ExecutionConfig, exchange_id: str = "binance", market_type: str = "spot"):
    """Factory: pick the broker for the configured mode."""
    if cfg.mode in ("dry_run", "paper"):
        return PaperBroker(cfg)
    if cfg.mode == "live":
        return LiveBroker(exchange_id=exchange_id, market_type=market_type)
    raise ValueError(f"Unknown execution mode: {cfg.mode!r} (expected dry_run|paper|live)")
