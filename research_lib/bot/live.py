"""The bar-by-bar trading loop (PDF: ... -> Risk Engine -> Execution Engine -> Broker).

``TradingBot`` is the orchestrator. One instance owns:

* the loaded + featured market data (anchor instrument + pairs),
* the regime router output (which strategy may trade, when, and the vol-spike halt),
* the trained meta model + bet sizer,
* the central risk engine (single choke point for every order),
* the execution engine (idempotency + order state machine), the broker, and the lock.

Two ways to use it:

* ``run(until_bar=...)`` - walk bars forward one at a time. Used for dry-run / paper and
  for live stepping. It is decision-time-only: at bar ``t`` it sees data up to ``t`` and
  fills at the open of ``t+1`` via the broker, which is the same convention as research.
* ``backtest()`` - replay the identical decision path over the full history and report
  realised trades, so the bot's behaviour is checkable against the research numbers.

Nothing in this module invents a new signal. It routes, sizes, risk-checks and executes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.bot.broker import PaperBroker, make_broker
from research_lib.bot.config import BotConfig
from research_lib.bot.features import (
    PrimarySignal,
    build_feature_frame,
    ou_signals_for_pair,
    pairs_signals,
    turtle_signals,
)
from research_lib.bot.sizing import BetSizerAdapter, MetaModel
from research_lib.execution.engine import ExecutionEngine
from research_lib.execution.order_state import OrderStateStore
from research_lib.execution.reconciliation import PositionReconciler
from research_lib.execution.redis_lock import DistributedLock, IdempotencyGuard
from research_lib.risk.costs import CostModel
from research_lib.risk.engine import CentralRiskEngine, OrderIntent, RiskLimits


def _bar_field(bar: pd.Series | None, name: str) -> float | None:
    """Best-effort float extraction of an OHLC field from a bar (None when absent)."""
    if bar is None:
        return None
    value = bar.get(name)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


class _InProcessRedis:
    """Minimal in-process stand-in for the handful of Redis ops the lock/guard use.

    Only ``set(nx=, px=)``, ``get``, ``delete`` and ``eval`` (the compare-and-delete Lua
    path) are needed. This lets the bot run with zero external services while keeping the
    exact same lock semantics in single-process mode.
    """

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def set(self, name, value, nx=False, px=None, ex=None):  # noqa: ANN001
        key = name.decode() if isinstance(name, bytes) else str(name)
        val = value.encode() if isinstance(value, str) else value
        if nx and key in self._data:
            return False
        self._data[key] = val
        return True

    def get(self, name):  # noqa: ANN001
        key = name.decode() if isinstance(name, bytes) else str(name)
        return self._data.get(key)

    def delete(self, *names):  # noqa: ANN002
        removed = 0
        for n in names:
            key = n.decode() if isinstance(n, bytes) else str(n)
            removed += 1 if self._data.pop(key, None) is not None else 0
        return removed

    def eval(self, script, numkeys, *keys_and_args):  # noqa: ANN001
        # emulate the compare-and-delete release script
        key, token = keys_and_args[0], keys_and_args[1]
        cur = self.get(key)
        tok = token.encode() if isinstance(token, str) else token
        if cur == tok:
            self.delete(key)
            return 1
        return 0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BotResult:
    """Everything a caller needs to judge what the bot did and why."""

    mode: str
    bars_processed: int
    orders_submitted: int
    orders_rejected: int = 0
    trades: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    equity: float = 0.0
    start_equity: float = 0.0
    meta_report: dict = field(default_factory=dict)
    router_summary: dict = field(default_factory=dict)
    signal_counts: dict = field(default_factory=dict)
    equity_curve: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def return_pct(self) -> float:
        if not self.start_equity:
            return 0.0
        return (self.equity / self.start_equity - 1.0) * 100.0

    def summary(self) -> str:
        closed = [t for t in self.trades if t.get("status") == "closed"]
        open_n = len(self.trades) - len(closed)
        wins = sum(1 for t in closed if (t.get("pnl") or 0.0) > 0)
        realized = sum(float(t.get("pnl") or 0.0) for t in closed)
        win_rate = (wins / len(closed) * 100.0) if closed else 0.0
        lines = [
            f"mode={self.mode} bars={self.bars_processed}",
            f"signals={self.signal_counts}",
            f"orders submitted={self.orders_submitted} rejected={self.orders_rejected}",
            f"trades closed={len(closed)} open={open_n} win_rate={win_rate:.1f}% realized_pnl={realized:,.2f}",
            f"equity {self.start_equity:,.2f} -> {self.equity:,.2f} ({self.return_pct:+.2f}%)",
            f"router={self.router_summary}",
            f"meta={self.meta_report}",
        ]
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------
class TradingBot:
    """Configuration-driven orchestrator over the validated research stack."""

    def __init__(self, cfg: BotConfig | None = None):
        self.cfg = cfg or BotConfig()
        self._prepared = False

        # market data (filled by prepare())
        self.anchor: DataFrame | None = None
        self.panels: dict[str, DataFrame] = {}
        self.signals: list[PrimarySignal] = []
        self.router_out: Any = None
        self.turtle_gate: np.ndarray | None = None
        self.pairs_gate: np.ndarray | None = None
        # closed-trade ledger (open positions live in the run loop)
        self.closed_trades: list[dict] = []
        # bars dropped from the head of the panel by prepare(limit_bars=...) warmup
        self._warmup: int = 0

        # decision stack
        self.meta = MetaModel(self.cfg)
        self.sizer = BetSizerAdapter(self.cfg)

        # infrastructure
        cost = CostModel(
            maker_fee=self.cfg.execution.maker_fee,
            taker_fee=self.cfg.execution.taker_fee,
            spread_bps=self.cfg.execution.spread_bps,
            slippage_bps=self.cfg.execution.slippage_bps,
        )
        self.cost = cost
        self.broker = make_broker(self.cfg.execution)
        self.risk = CentralRiskEngine(
            limits=RiskLimits(
                per_trade_risk=self.cfg.risk.per_trade_risk,
                per_symbol_notional=self.cfg.risk.per_symbol_notional,
                portfolio_notional=self.cfg.risk.portfolio_notional,
                max_leverage=self.cfg.risk.max_leverage,
                max_position_notional=self.cfg.risk.max_position_notional,
                daily_loss_limit=self.cfg.risk.daily_loss_limit,
                weekly_loss_limit=self.cfg.risk.weekly_loss_limit,
                max_drawdown_kill=self.cfg.risk.max_drawdown_kill,
                max_strategy_fraction=self.cfg.risk.max_strategy_fraction,
                max_correlation_exposure=self.cfg.risk.max_correlation_exposure,
            )
        )
        self.risk.state.equity = self.cfg.execution.equity
        self.risk.state.peak_equity = self.cfg.execution.equity
        self.risk.state.day_start_equity = self.cfg.execution.equity
        self.reconciler = PositionReconciler(self.risk)

        store = OrderStateStore()
        lock = self._make_lock()
        self.execution = ExecutionEngine(
            risk=self.risk,
            store=store,
            broker=self.broker,
            idem_guard=IdempotencyGuard(client=lock.client),
            model_version=self.cfg.version,
        )
        self.lock = lock

    # ------------------------------------------------------------------
    # infrastructure helpers
    # ------------------------------------------------------------------
    def _make_lock(self) -> DistributedLock:
        """Build the distributed lock (PDF: Lock Adapter). Falls back to in-process."""
        url = self.cfg.execution.redis_url
        redis_client = None
        if url:
            try:
                import redis  # noqa: PLC0415

                redis_client = redis.Redis.from_url(url, decode_responses=False)
                redis_client.ping()
            except Exception:  # noqa: BLE001 - degrade to a fake/single-process lock
                redis_client = None
        if redis_client is None:
            try:
                import fakeredis  # noqa: PLC0415

                redis_client = fakeredis.FakeRedis(decode_responses=False)
            except Exception:  # noqa: BLE001
                redis_client = None
        if redis_client is None:
            redis_client = _InProcessRedis()
        return DistributedLock(
            client=redis_client,
            resource=self.cfg.name,
            ttl_ms=self.cfg.execution.lock_ttl_ms,
        )

    # ------------------------------------------------------------------
    # data preparation
    # ------------------------------------------------------------------
    def load_data(self) -> DataFrame:
        """Load and normalize the anchor + pair panels (PDF: Data Normalizer)."""
        from research_lib.data.loader import load_pair_tf
        from pathlib import Path

        data_dir = Path(self.cfg.data.data_dir) if self.cfg.data.data_dir else None
        pairs = list(dict.fromkeys([self.cfg.data.pairs[0]] + [self.cfg.strategy.pairs_a, self.cfg.strategy.pairs_b]))
        if self.cfg.strategy.pairs_enabled:
            pairs = list(dict.fromkeys(pairs + self.cfg.data.pairs))

        for pair in pairs:
            try:
                self.panels[pair] = load_pair_tf(
                    pair,
                    self.cfg.data.timeframe,
                    exchange=self.cfg.data.exchange,
                    data_dir=data_dir,
                )
            except FileNotFoundError:
                continue

        anchor_pair = self.cfg.data.pairs[0]
        if anchor_pair not in self.panels:
            raise FileNotFoundError(
                f"Anchor pair {anchor_pair} {self.cfg.data.timeframe} not found in the "
                f"configured data directory. Download it first (see bot README)."
            )
        self.anchor = self.panels[anchor_pair]
        return self.anchor

    def _maybe_information_bars(self, df: DataFrame) -> DataFrame:
        """Optionally resample into information bars (PDF: Information Bars)."""
        if not self.cfg.data.use_information_bars:
            return df
        bar_type = self.cfg.data.bar_type
        if bar_type == "time":
            return df
        threshold = self.cfg.data.bar_threshold
        if threshold is None:
            # sensible default: median bar volume (volume bars) / median bar dollar (dollar)
            if bar_type == "volume":
                threshold = float(df["volume"].median())
            else:
                threshold = float((df["close"] * df["volume"]).median())
        try:
            from research_lib.data.infobars import BarSpec, build_bars

            out = build_bars(
                df,
                BarSpec(bar_type=bar_type, threshold=float(threshold)),
                symbol=str(df["symbol"].iloc[0]) if "symbol" in df.columns else "",
                timeframe_label=bar_type,
            )
            return out if out is not None and len(out) > 100 else df
        except Exception:  # noqa: BLE001 - fall back to raw bars rather than fail
            return df

    def prepare(self, limit_bars: int | None = None) -> None:
        """Load data, build features, generate primary signals, train the meta model.

        ``limit_bars`` restricts the panel to the LAST ``limit_bars`` bars plus a
        warmup prefix, which makes short replays fast (the HMM router refit loop
        dominates runtime on full history). Causality is preserved: we only ever
        keep history that PRECEDES the bars we act on, so no future information
        leaks into an earlier decision. ``None`` uses the full panel.
        """
        if self._prepared:
            return
        self.load_data()
        anchor = self._maybe_information_bars(self.anchor)
        assert anchor is not None

        self._warmup = 0
        if limit_bars is not None:
            limit_bars = int(limit_bars)
            total = len(anchor)
            if 0 < limit_bars < total:
                # keep a generous warmup so rolling features/meta training are valid
                warmup = max(
                    int(getattr(self.cfg.data, "warmup_bars", 2000)),
                    int(self.cfg.router.hmm_min_train_bars) + int(self.cfg.router.hmm_refit_bars),
                    int(self.cfg.strategy.ou_lookback) * 2,
                    1500,
                )
                start = max(0, total - limit_bars - warmup)
                self._warmup = total - limit_bars - start
                anchor = anchor.iloc[start:].reset_index(drop=True)
                for pair, panel in list(self.panels.items()):
                    if panel is not None and len(panel) >= total:
                        self.panels[pair] = panel.iloc[start:].reset_index(drop=True)

        # PDF: Information Bars -> Feature Engine
        self.anchor = build_feature_frame(anchor, self.cfg)

        # PDF: Signal Router (regime engine + vol-spike kill switch)
        if self.cfg.router.enable_router:
            self._build_router()
        else:
            n = len(self.anchor)
            self.turtle_gate = np.ones(n, dtype=bool)
            self.pairs_gate = np.ones(n, dtype=bool)

        # PDF: Turtle Engine + OU Engine (+ Pairs)
        self.signals = self._generate_signals()

        # PDF: Triple Barrier -> Meta Model (direction still comes from primaries)
        self.meta.fit(self.anchor, self.signals)

        self._prepared = True

    def _build_router(self) -> None:
        from research_lib.bot.config import RouterConfig as BotRouterConfig  # noqa: F401
        from research_lib.strategies.regime_hmm import HMMParams
        from research_lib.strategies.unified_router import RouterConfig, filtered_hmm_router
        from research_lib.ml.volspike import VolSpikeParams

        assert self.anchor is not None
        rc = RouterConfig(
            hmm=HMMParams(n_states=self.cfg.router.hmm_states, n_iter=150),
            refit_bars=self.cfg.router.hmm_refit_bars,
            min_train_bars=self.cfg.router.hmm_min_train_bars,
            turtle_regimes=tuple(self.cfg.router.turtle_regimes),
            pairs_regimes=tuple(self.cfg.router.pairs_regimes),
            vol_spike=VolSpikeParams(horizon_bars=self.cfg.router.vol_spike_horizon_bars),
            vol_kill_p=self.cfg.router.vol_kill_p,
            vol_refit_bars=self.cfg.router.vol_refit_bars,
        )
        self.router_out = filtered_hmm_router(self.anchor, rc, seed=42)
        self.turtle_gate = self.router_out.turtle_gate
        self.pairs_gate = self.router_out.pairs_gate

    def _generate_signals(self) -> list[PrimarySignal]:
        assert self.anchor is not None
        anchor_pair = self.cfg.data.pairs[0]
        sigs: list[PrimarySignal] = []
        sigs += turtle_signals(self.anchor, self.cfg, anchor_pair)

        for pair in self.cfg.data.pairs:
            panel = self.panels.get(pair)
            if panel is not None and len(panel) > self.cfg.strategy.ou_lookback:
                sigs += ou_signals_for_pair(self.anchor, self.cfg, pair)

        if self.cfg.strategy.pairs_enabled:
            a = self.panels.get(self.cfg.strategy.pairs_a)
            b = self.panels.get(self.cfg.strategy.pairs_b)
            if a is not None and b is not None:
                sigs += pairs_signals(a, b, self.cfg)

        sigs.sort(key=lambda s: (s.bar_index, s.strategy_id))
        return sigs

    # ------------------------------------------------------------------
    # gate lookup
    # ------------------------------------------------------------------
    def _gate_allows(self, signal: PrimarySignal) -> bool:
        """Consult the router gate for this signal's fill bar (causal: uses bar-1)."""
        if self.turtle_gate is None or self.pairs_gate is None:
            return True
        fill_bar = signal.bar_index + 1
        if fill_bar >= len(self.turtle_gate):
            return False
        if signal.strategy_id == "turtle":
            return bool(self.turtle_gate[fill_bar])
        if signal.strategy_id in ("ou", "pairs"):
            return bool(self.pairs_gate[fill_bar])
        return True

    # ------------------------------------------------------------------
    # order construction
    # ------------------------------------------------------------------
    def _intent_for(self, signal: PrimarySignal, eq: float) -> OrderIntent:
        """Turn a routed, sized primary signal into a risk-checkable order intent."""
        price = float(signal.entry_ref)
        stop = signal.stop_price
        risk_per_unit = abs(price - stop) if stop is not None and np.isfinite(stop) else price * 0.02
        risk_per_unit = max(risk_per_unit, price * 1e-4)

        budget = eq * self.cfg.risk.per_trade_risk
        units = budget / risk_per_unit
        # cap notional at the per-symbol limit (spot, no leverage)
        max_notional = eq * min(self.cfg.risk.per_symbol_notional, self.cfg.risk.max_position_notional)
        units = min(units, max_notional / price)
        units = float(max(units, 0.0))

        return OrderIntent(
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            signal_time=signal.signal_time,
            side=int(signal.side),
            units=units,
            entry_ref_price=price,
            stop_price=stop,
            notional=units * price,
            model_p=None,
            bet_fraction=1.0,
            post_only=self.cfg.execution.post_only,
        )

    # ------------------------------------------------------------------
    # the main loop
    # ------------------------------------------------------------------
    def run(self, until_bar: int | None = None, start_bar: int | None = None) -> BotResult:
        """Walk the anchor frame bar by bar, executing the decision path.

        Only signals whose fill bar falls in ``(start_bar, until_bar]`` are actioned, so a
        caller can step the bot forward incrementally (live) or replay a range (dry-run).
        """
        self.prepare(limit_bars=until_bar)
        assert self.anchor is not None and self.turtle_gate is not None

        n = len(self.anchor)
        # when prepare() trimmed a warmup prefix, only act on bars after it so
        # the requested window is exactly what gets traded
        start = int(start_bar if start_bar is not None else self._warmup)
        end = n if until_bar is None else min(int(self._warmup) + int(until_bar), n)

        # optional hard cap for quick smoke runs
        if self.cfg.execution.max_bars is not None and until_bar is None:
            end = min(end, start + int(self.cfg.execution.max_bars))

        result = BotResult(
            mode=self.cfg.execution.mode,
            bars_processed=0,
            orders_submitted=0,
            equity=float(self.risk.state.equity),
            start_equity=float(self.cfg.execution.equity),
        )
        # a fresh ledger per run keeps each report self-contained
        self.closed_trades = []

        # index signals by fill bar so the loop is O(signals) not O(bars*signals)
        by_fill: dict[int, list[PrimarySignal]] = {}
        for s in self.signals:
            by_fill.setdefault(s.bar_index + 1, []).append(s)
        counts: dict[str, int] = {}
        for s in self.signals:
            counts[s.strategy_id] = counts.get(s.strategy_id, 0) + 1
        result.signal_counts = counts

        open_positions: dict[str, dict] = {}
        locked = self.lock.acquire()
        if not locked:
            result.warnings.append(
                "distributed lock not acquired - another bot instance may be running; "
                "proceeding because this run is single-process"
            )

        try:
            for t in range(start, end):
                result.bars_processed += 1
                bar = self.anchor.iloc[t]
                market_price = float(bar["open"])

                # 1. mark-to-market any open position at this bar's open
                self._mark_to_market(open_positions, market_price, bar, t, result)

                # 2. act on signals that fill at this bar's open
                for sig in by_fill.get(t, []):
                    self._handle_signal(sig, t, market_price, result, open_positions, bar=bar)

                # 3. mirror the research cost model
                if isinstance(self.broker, PaperBroker):
                    self.broker.set_market(
                        market_price,
                        high=_bar_field(bar, "high"),
                        low=_bar_field(bar, "low"),
                    )

                # 4. record the equity curve
                result.equity_curve.append(
                    {"bar": t, "date": str(bar.get("date", "")), "equity": float(self.risk.state.equity)}
                )

                # 5. periodic position reconciliation (PDF: Execution -> Reconciliation)
                every = self.cfg.execution.recon_every_bars
                if every is not None and every > 0 and (t + 1) % every == 0:
                    self._reconcile(open_positions, result, t)

                if self.cfg.execution.poll_seconds:
                    time.sleep(self.cfg.execution.poll_seconds)
        finally:
            self.lock.release()

        # final mark-to-market: close everything out so the ledger and equity agree
        if len(self.anchor) and open_positions:
            last_px = float(self.anchor.iloc[end - 1]["close"]) if end > 0 else market_price
            self._mark_to_market(open_positions, last_px, None, end - 1, result, force=True)

        # final reconciliation - a run that ends diverged must say so loudly
        if self.cfg.execution.recon_every_bars:
            self._reconcile(open_positions, result, end - 1)

        result.equity = float(self.risk.state.equity)
        result.meta_report = self.meta.report.to_dict()
        result.router_summary = self._router_summary(end)
        result.trades = list(self.closed_trades) + list(open_positions.values())
        return result

    # ------------------------------------------------------------------
    def _router_summary(self, end: int) -> dict:
        if self.router_out is None or self.turtle_gate is None:
            return {"enabled": False}
        sl = slice(0, end)
        regime = pd.Series(self.router_out.regime[sl])
        return {
            "enabled": True,
            "regime_counts": regime.value_counts().to_dict(),
            "turtle_gate_on": int(np.sum(self.turtle_gate[sl])),
            "pairs_gate_on": int(np.sum(self.pairs_gate[sl])),
            "killed_by_vol_spike": int(np.sum(self.router_out.halt_reason[sl] == "vol_spike")),
        }

    # ------------------------------------------------------------------
    def _reconcile(
        self,
        open_positions: dict[str, dict],
        result: BotResult,
        bar_index: int,
    ) -> None:
        """Compare internal open positions with broker-reported positions.

        On divergence the reconciler sets the risk engine's kill switch, so every
        subsequent order is rejected until a clean reconciliation runs (fail-safe:
        under uncertainty, NO NEW ORDER). On a clean run ``resume()`` clears a
        previously tripped switch.
        """
        positions_fn = getattr(self.broker, "positions", None)
        if not callable(positions_fn):
            return
        internal = {
            sym: float(pos["side"]) * float(pos["units"]) for sym, pos in open_positions.items()
        }
        try:
            broker_pos = positions_fn()
        except Exception as e:  # noqa: BLE001 - unreachable broker => halt, do not trade
            broker_pos = {}
            result.warnings.append(f"reconciliation: broker positions unavailable: {e}")
        rr = self.reconciler.reconcile(internal, {k: float(v) for k, v in broker_pos.items()})
        if rr.consistent:
            self.reconciler.resume()
            result.decisions.append({"bar": bar_index, "action": "reconciliation_ok"})
        else:
            result.warnings.append(
                f"reconciliation failed at bar {bar_index}: {'; '.join(rr.diffs)} - "
                "kill switch set, no new orders"
            )
            result.decisions.append(
                {"bar": bar_index, "action": "reconciliation_failed", "diffs": list(rr.diffs)}
            )

    def _handle_signal(
        self,
        sig: PrimarySignal,
        bar_index: int,
        market_price: float,
        result: BotResult,
        open_positions: dict[str, dict],
        bar: pd.Series | None = None,
    ) -> None:
        # 1. regime / vol-spike gate (PDF: Signal Router)
        if not self._gate_allows(sig):
            result.decisions.append(
                {"bar": bar_index, "strategy": sig.strategy_id, "action": "gated_out"}
            )
            return

        # 2. one position per symbol at a time
        if sig.symbol in open_positions:
            result.decisions.append(
                {"bar": bar_index, "strategy": sig.strategy_id, "action": "already_open"}
            )
            return

        # 3. meta probability -> bet size (PDF: Meta Model -> Bet Sizer)
        eq = float(self.risk.state.equity)
        p = self.meta.predict(self.anchor, sig)
        frac = self.sizer.fraction(p, self.meta.report.enabled)
        if frac <= 0.0:
            result.decisions.append(
                {
                    "bar": bar_index,
                    "strategy": sig.strategy_id,
                    "action": "meta_rejected",
                    "p": p,
                }
            )
            return

        intent = self._intent_for(sig, eq)
        intent.model_p = p
        intent.bet_fraction = frac
        intent.notional = intent.units * intent.entry_ref_price

        # 4. risk-gated execution (PDF: Risk Engine -> Execution Engine -> Broker)
        if isinstance(self.broker, PaperBroker):
            self.broker.set_market(
                market_price,
                high=_bar_field(bar, "high"),
                low=_bar_field(bar, "low"),
            )
        exec_result = self.execution.execute(intent)

        if exec_result.submitted and exec_result.broker_ack:
            ack = exec_result.broker_ack
            if intent.post_only and ack.get("status") != "filled":
                # the post-only quote was not traded through - cancel the resting
                # order (one-shot attempt) and record a missed fill instead of
                # assuming one happened
                self.broker.cancel(str(ack.get("order_id", "")))
                result.decisions.append(
                    {
                        "bar": bar_index,
                        "strategy": sig.strategy_id,
                        "action": "missed_fill",
                        "side": sig.side,
                        "p": p,
                        "bet_fraction": frac,
                    }
                )
                return
            filled = float(ack.get("filled") or intent.units)
            price = float(ack.get("avg_price") or market_price)
            fee = float(ack.get("fee") or 0.0)
            self.risk.on_position_opened(intent)
            open_positions[sig.symbol] = {
                "strategy": sig.strategy_id,
                "symbol": sig.symbol,
                "side": sig.side,
                "units": filled,
                "entry_price": price,
                "entry_bar": bar_index,
                "stop_price": sig.stop_price,
                "fee": fee,
                "model_p": p,
                "bet_fraction": frac,
                "open_time": str(sig.signal_time),
                "status": "open",
            }
            result.orders_submitted += 1
            result.trades.append(dict(open_positions[sig.symbol]))
            result.decisions.append(
                {
                    "bar": bar_index,
                    "strategy": sig.strategy_id,
                    "action": "filled",
                    "side": sig.side,
                    "units": filled,
                    "price": price,
                    "p": p,
                    "bet_fraction": frac,
                }
            )
        else:
            result.orders_rejected += 1
            result.decisions.append(
                {
                    "bar": bar_index,
                    "strategy": sig.strategy_id,
                    "action": "rejected",
                    "message": exec_result.message,
                }
            )

    # ------------------------------------------------------------------
    def _mark_to_market(
        self,
        open_positions: dict[str, dict],
        price: float,
        bar: pd.Series | None,
        bar_index: int | None = None,
        result: BotResult | None = None,
        force: bool = False,
    ) -> None:
        """Update equity for open positions and close them on stop / time stop.

        ``force=True`` closes every remaining position at ``price`` (used for the
        final mark-to-market so the trade ledger reconciles with equity).
        """
        if not open_positions:
            return
        for sym, pos in list(open_positions.items()):
            stop = pos.get("stop_price")
            side = int(pos["side"])
            exit_reason = None

            # stop check uses intrabar low/high when available (conservative)
            if bar is not None and stop is not None and np.isfinite(stop):
                if side > 0 and float(bar["low"]) <= stop:
                    exit_reason = "stop"
                    price = float(stop)
                elif side < 0 and float(bar["high"]) >= stop:
                    exit_reason = "stop"
                    price = float(stop)

            pnl = (price - pos["entry_price"]) * pos["units"] * side - pos["fee"]
            if exit_reason is None:
                if force:
                    exit_reason = "eod"
                else:
                    continue

            # The risk engine owns equity accounting (single source of truth):
            # on_position_closed() applies `realized_pnl` to equity and rolls
            # the daily/weekly counters. Do NOT also mutate equity here - that
            # would double-count every realised PnL.
            self.risk.on_position_closed(sym, pnl)
            # keep the broker's fill-derived positions in sync so reconciliation
            # does not flag the closed leg as still open
            if isinstance(self.broker, PaperBroker):
                self.broker.record_external_exit(sym, side, float(pos["units"]), float(price))
            pos["status"] = "closed"
            pos["exit_price"] = float(price)
            pos["exit_reason"] = exit_reason
            pos["pnl"] = float(pnl)
            if bar_index is not None:
                pos["exit_bar"] = int(bar_index)
            self.closed_trades.append(dict(pos))
            open_positions.pop(sym, None)
            if result is not None:
                result.decisions.append(
                    {
                        "bar": bar_index,
                        "strategy": pos.get("strategy", ""),
                        "symbol": sym,
                        "action": "closed",
                        "exit_reason": exit_reason,
                        "pnl": float(pnl),
                    }
                )

    # ------------------------------------------------------------------
    # reporting helper
    # ------------------------------------------------------------------
    def write_report(self, result: BotResult, path: str | None = None) -> str:
        """Write a machine-readable run report (JSON) and return its path."""
        import json
        from pathlib import Path

        from research_lib.constants import PROJECT_ROOT

        if path is None:
            out_dir = PROJECT_ROOT / "user_data" / "bot_runs"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            path = str(out_dir / f"run_{self.cfg.name}_{stamp}.json")
        Path(path).write_text(
            json.dumps(
                {
                    "config": self.cfg.to_dict(),
                    "summary": result.summary(),
                    "bars_processed": result.bars_processed,
                    "orders_submitted": result.orders_submitted,
                    "orders_rejected": result.orders_rejected,
                    "start_equity": result.start_equity,
                    "equity": result.equity,
                    "return_pct": result.return_pct,
                    "signal_counts": result.signal_counts,
                    "router": result.router_summary,
                    "meta_model": result.meta_report,
                    "trades": result.trades,
                    "equity_curve": result.equity_curve,
                    "decisions": result.decisions[-500:],
                    "warnings": result.warnings,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# convenience entry point
# ---------------------------------------------------------------------------
def run_bot(cfg: BotConfig | None = None, **kwargs: Any) -> BotResult:
    """Build a bot, run it, and write a report. The one-call entry point."""
    bot = TradingBot(cfg)
    result = bot.run(**kwargs)
    path = bot.write_report(result)
    result.warnings.append(f"report: {path}")
    return result
