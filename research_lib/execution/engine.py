"""Execution engine (Phase 30) - signal generation is fully separated from execution.

Chain:
  Signal Engine -> Risk Engine -> Order Intent -> Execution Engine -> Broker Adapter
    -> Order Confirmation -> Position Reconciliation

No research strategy submits broker orders directly. The execution engine:
  * computes an idempotency key per intent
  * claims the Redis idempotency guard (single submission across workers)
  * passes the intent through the CentralRiskEngine
  * drives the OrderStateStore state machine to submission
  * only marks FILLED on an explicit broker fill confirmation
  * on any inconsistency moves the order to RECONCILIATION_REQUIRED and stops new orders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from research_lib.execution.order_state import (
    OrderRecord,
    OrderState,
    OrderStateStore,
    make_idempotency_key,
)
from research_lib.execution.redis_lock import IdempotencyGuard
from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskDecision


class BrokerAdapter(Protocol):
    """Minimal broker surface the execution engine talks to."""

    def submit(self, intent: OrderIntent) -> dict:
        """Return broker ack: {'order_id', 'status': 'open'|'rejected'|'error', ...}."""
        ...

    def fetch(self, broker_order_id: str) -> dict:
        """Return current broker state: {'status', 'filled', 'remaining', ...}."""
        ...

    def cancel(self, broker_order_id: str) -> dict:
        ...


@dataclass
class ExecutionResult:
    order: OrderRecord | None
    decision: RiskDecision | None
    broker_ack: dict | None
    submitted: bool
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "message": self.message,
            "risk_approved": bool(self.decision and self.decision.approved),
            "risk_reasons": (self.decision.reasons if self.decision else []),
            "order_state": self.order.state.value if self.order else None,
            "order_id": self.order.order_id if self.order else None,
            "idempotency_key": self.order.idempotency_key if self.order else None,
            "broker_ack": self.broker_ack,
        }


class ExecutionEngine:
    def __init__(
        self,
        risk: CentralRiskEngine,
        store: OrderStateStore,
        broker: BrokerAdapter,
        idem_guard: IdempotencyGuard | None = None,
        model_version: str = "research-v1",
    ):
        self.risk = risk
        self.store = store
        self.broker = broker
        self.idem = idem_guard
        self.model_version = model_version

    def _new_record(self, intent: OrderIntent) -> OrderRecord:
        key = make_idempotency_key(
            intent.strategy_id,
            intent.symbol,
            str(intent.signal_time),
            intent.side,
            self.model_version,
        )
        return OrderRecord(
            order_id=f"{intent.strategy_id}-{datetime.now(UTC).timestamp():.0f}-{intent.side}",
            idempotency_key=key,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            signal_time=str(intent.signal_time),
            side=intent.side,
            quantity=intent.units,
        )

    def execute(self, intent: OrderIntent) -> ExecutionResult:
        record = self._new_record(intent)
        # idempotency: if a record already exists for this key, do NOT resubmit
        existing, created = self.store.create_or_get(record)
        if not created:
            return ExecutionResult(
                order=existing,
                decision=None,
                broker_ack=None,
                submitted=False,
                message="duplicate idempotency key - order already exists; not resubmitted",
            )
        # distributed single-submission guard
        if self.idem is not None and not self.idem.try_claim(record.idempotency_key):
            return ExecutionResult(
                order=record,
                decision=None,
                broker_ack=None,
                submitted=False,
                message="idempotency guard claim failed (another worker is handling this signal)",
            )
        # risk gate - the ONLY path to a broker order
        decision = self.risk.evaluate(intent)
        if not decision.approved:
            record.transition(OrderState.REJECTED, reason="; ".join(decision.reasons))
            return ExecutionResult(
                order=record,
                decision=decision,
                broker_ack=None,
                submitted=False,
                message="risk rejected: " + "; ".join(decision.reasons),
            )
        record.transition(OrderState.RISK_APPROVED, reason="risk passed")
        record.transition(OrderState.ORDER_PENDING, reason="queued")
        record.transition(OrderState.ORDER_SUBMITTED, reason="calling broker")
        try:
            ack = self.broker.submit(intent)
        except Exception as e:  # noqa: BLE001 - broker failure must not crash the loop
            record.transition(OrderState.ERROR, reason=f"broker submit exception: {e}")
            return ExecutionResult(
                order=record, decision=decision, broker_ack=None, submitted=False,
                message=f"broker submit failed: {e}",
            )
        record.broker_order_id = str((ack or {}).get("order_id", record.order_id))
        # Never assume filled on submit success
        status = (ack or {}).get("status", "")
        if status == "rejected":
            record.transition(OrderState.REJECTED, reason="broker rejected")
            return ExecutionResult(order=record, decision=decision, broker_ack=ack, submitted=False, message="broker rejected order")
        if status == "error":
            record.transition(OrderState.ERROR, reason="broker error")
            return ExecutionResult(order=record, decision=decision, broker_ack=ack, submitted=False, message="broker error")
        if status in ("open", "partially_filled"):
            record.transition(OrderState.PARTIALLY_FILLED, reason="ack open")
        return ExecutionResult(order=record, decision=decision, broker_ack=ack, submitted=True, message="order submitted")

    def reconcile_open(self, record: OrderRecord) -> OrderState:
        """Poll the broker for the true state of an order that is not FILLED/CANCELLED."""
        try:
            state = self.broker.fetch(record.broker_order_id or record.order_id)
        except Exception:  # noqa: BLE001
            record.transition(OrderState.RECONCILIATION_REQUIRED, reason="broker fetch failed")
            return record.state
        bstatus = (state or {}).get("status", "")
        if bstatus == "closed":
            filled = float((state or {}).get("filled", 0) or 0)
            if filled > 0:
                record.transition(OrderState.FILLED, reason="broker confirmed fill")
            else:
                record.transition(OrderState.CANCELLED, reason="broker closed unfilled")
        elif bstatus == "canceled":
            record.transition(OrderState.CANCELLED, reason="broker cancelled")
        elif bstatus == "open":
            record.transition(OrderState.PARTIALLY_FILLED, reason="still open")
        else:
            record.transition(OrderState.RECONCILIATION_REQUIRED, reason=f"unexpected broker status {bstatus}")
        return record.state
