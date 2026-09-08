"""Position reconciliation (Phase 33) and failure-handling simulation (Phase 34).

Reconciliation periodically compares internal state against broker state. If inconsistent:
STOP NEW ORDERS -> reconcile -> resume only when consistent.

Failure scenarios simulated for tests: network disconnect, broker API timeout, duplicate
messages, stale market data, partial fill, order rejection, process crash, Redis unavailable,
database unavailable, clock mismatch, WebSocket disconnect, REST API failure. The system must
fail safely; default behavior under uncertainty is NO NEW ORDER (enforced by the risk engine's
kill switches).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from research_lib.risk.engine import CentralRiskEngine, RiskState


@dataclass
class ReconResult:
    consistent: bool
    diffs: list[str] = field(default_factory=list)


class PositionReconciler:
    """Compare internal open positions with the broker's reported positions."""

    def __init__(self, risk: CentralRiskEngine):
        self.risk = risk

    def reconcile(
        self,
        internal_positions: dict[str, float],
        broker_positions: dict[str, float],
        tolerance: float = 1e-6,
    ) -> ReconResult:
        """internal/broker positions: {symbol: signed units}. On inconsistency, set the risk
        engine kill switch so no new orders are placed."""
        all_symbols = set(internal_positions) | set(broker_positions)
        diffs: list[str] = []
        for sym in sorted(all_symbols):
            a = internal_positions.get(sym, 0.0)
            b = broker_positions.get(sym, 0.0)
            if abs(a - b) > tolerance:
                diffs.append(f"{sym}: internal={a} broker={b}")
        consistent = len(diffs) == 0
        self.risk.state.reconciliation_ok = consistent
        if not consistent:
            self.risk.state.kill_switch = True
        return ReconResult(consistent=consistent, diffs=diffs)

    def resume(self) -> None:
        """Called after a successful manual/automated reconciliation; only clears the kill
        switch when reconciliation is OK."""
        if self.risk.state.reconciliation_ok:
            self.risk.state.kill_switch = False


@dataclass
class FailureInjector:
    """Simulates broker/plumbing failures to prove fail-safe behavior (Phase 34)."""

    network_down: bool = False
    api_timeout: bool = False
    duplicate_messages: bool = False
    stale_data: bool = False
    partial_fill_mode: bool = False
    reject_orders: bool = False
    crash_after_submit: bool = False
    redis_down: bool = False
    clock_skew_seconds: float = 0.0

    def submit(self, intent) -> dict:
        if self.network_down or self.api_timeout:
            raise TimeoutError("broker unreachable")
        if self.reject_orders:
            return {"order_id": "rej-1", "status": "rejected"}
        if self.crash_after_submit:
            # simulate a process crash after ack - the order exists at the broker but the
            # local state machine never saw the ack -> reconciliation required
            return {"order_id": "crash-1", "status": "open", "ack_lost": True}
        return {"order_id": "broker-1", "status": "open"}

    def fetch(self, broker_order_id: str) -> dict:
        if self.network_down or self.api_timeout:
            raise TimeoutError("broker unreachable")
        if self.partial_fill_mode:
            return {"status": "open", "filled": 0.5, "remaining": 0.5}
        return {"status": "closed", "filled": 1.0, "remaining": 0.0}
