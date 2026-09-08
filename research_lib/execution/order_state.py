"""Order state machine (Phase 32) and idempotent order handling.

States:
  SIGNAL_CREATED -> RISK_APPROVED -> ORDER_PENDING -> ORDER_SUBMITTED
       -> PARTIALLY_FILLED | FILLED | CANCEL_PENDING -> CANCELLED
       | REJECTED | ERROR | RECONCILIATION_REQUIRED

Every transition is persisted. An order is NEVER assumed filled because a submit request
succeeded - only an explicit fill confirmation moves it to FILLED.

Idempotency (spec): every order intent carries a deterministic idempotency key:
  key = sha1(strategy_id | symbol | signal_time | side | model_version)
If two workers receive the same signal, only ONE order may be submitted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class OrderState(str, Enum):
    SIGNAL_CREATED = "SIGNAL_CREATED"
    RISK_APPROVED = "RISK_APPROVED"
    ORDER_PENDING = "ORDER_PENDING"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


# legal transition map
_ALLOWED: dict[OrderState, set[OrderState]] = {
    OrderState.SIGNAL_CREATED: {OrderState.RISK_APPROVED, OrderState.REJECTED, OrderState.ERROR},
    OrderState.RISK_APPROVED: {OrderState.ORDER_PENDING, OrderState.REJECTED, OrderState.ERROR},
    OrderState.ORDER_PENDING: {OrderState.ORDER_SUBMITTED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.ERROR},
    OrderState.ORDER_SUBMITTED: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCEL_PENDING,
        OrderState.REJECTED,
        OrderState.ERROR,
        OrderState.RECONCILIATION_REQUIRED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCEL_PENDING,
        OrderState.ERROR,
        OrderState.RECONCILIATION_REQUIRED,
    },
    OrderState.FILLED: set(),
    OrderState.CANCEL_PENDING: {OrderState.CANCELLED, OrderState.ERROR, OrderState.RECONCILIATION_REQUIRED},
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.ERROR: set(),
    OrderState.RECONCILIATION_REQUIRED: {OrderState.FILLED, OrderState.CANCELLED, OrderState.ERROR},
}


class IllegalTransition(Exception):
    pass


@dataclass
class OrderRecord:
    """Persisted order record with full transition history."""

    order_id: str
    idempotency_key: str
    strategy_id: str
    symbol: str
    signal_time: str
    side: int
    quantity: float
    state: OrderState = OrderState.SIGNAL_CREATED
    transitions: list[dict] = field(default_factory=list)
    broker_order_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def transition(self, new_state: OrderState, reason: str = "") -> None:
        if new_state not in _ALLOWED[self.state]:
            raise IllegalTransition(f"{self.state} -> {new_state} not allowed")
        self.state = new_state
        self.transitions.append(
            {
                "to": new_state.value,
                "reason": reason,
                "ts": datetime.now(UTC).isoformat(),
            }
        )


def make_idempotency_key(
    strategy_id: str, symbol: str, signal_time: str, side: int, model_version: str = ""
) -> str:
    payload = "|".join([strategy_id, symbol, str(signal_time), str(side), model_version])
    return hashlib.sha1(payload.encode()).hexdigest()  # noqa: S324 - idempotency, not crypto


class OrderStateStore:
    """In-memory (or DB-backed) store enforcing one order per idempotency key."""

    def __init__(self) -> None:
        self._orders: dict[str, OrderRecord] = {}

    def create_or_get(self, record: OrderRecord) -> tuple[OrderRecord, bool]:
        """Return (record, created). If an order with the same idempotency key exists, the
        existing one is returned and nothing new is created (single submission)."""
        for existing in self._orders.values():
            if existing.idempotency_key == record.idempotency_key:
                return existing, False
        self._orders[record.order_id] = record
        return record, True

    def get(self, order_id: str) -> OrderRecord | None:
        return self._orders.get(order_id)

    def all(self) -> list[OrderRecord]:
        return list(self._orders.values())
