"""Funding-rate carry backtest (delta-neutral): long spot + short perpetual.

Implements the causal backtest specified in research/FIX_OR_ENHANCE_ASSESSMENT.md §3:
funding is collected/paid at 8h boundaries, both legs are costed through the project's
``CostModel`` (taker by default; maker when ``post_only=True`` - which is how assessment
step 4, "does post-only execution lift net carry", gets measured), the short leg is
marked to market with a liquidation check, and results are reported per quarter.

Conventions match the rest of the project: decisions happen at a funding print, fills
happen at the NEXT spot bar open. Funding prints before entry earn nothing.

Perp pricing: pass ``perp=`` (e.g. research/data/futures/{SYM}_USDT-1h-perp.feather,
produced by research/download_carry_data.py) to mark and settle the short leg on the
REAL perpetual series - this closes the "no perp price series" gap flagged in the
assessment and produces genuine basis pnl. Without ``perp=`` the leg falls back to the
spot * (1 + basis_bps) proxy (constant basis, legs' price pnl cancels by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.costs import CostModel

FUNDING_PERIODS_PER_YEAR = 3 * 365  # 8h funding


def load_funding(path: str | Path) -> DataFrame:
    """Load a funding-rate CSV (``date``, ``funding_rate``) into UTC-sorted form."""
    df = pd.read_csv(path)
    if "date" not in df.columns or "funding_rate" not in df.columns:
        raise ValueError(f"funding file {path} must have 'date' and 'funding_rate' columns")
    df = df[["date", "funding_rate"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed").astype("datetime64[ns, UTC]")
    df["funding_rate"] = df["funding_rate"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


@dataclass
class CarryConfig:
    equity: float = 100_000.0
    notional_fraction: float = 1.0  # fraction of equity committed as capital (spot + margin)
    leverage: float = 1.0           # perp leverage: margin = notional / leverage
    basis_bps: float = 0.0          # constant perp-over-spot premium proxy
    post_only: bool = False         # maker fills on both legs (assessment step 4)
    liquidation_buffer: float = 1.0  # liquidated when short-leg loss >= margin * buffer
    rebalance: bool = True          # top up / re-hedge before liquidation (what real
                                    # carry desks do - the assessment's 19/19 quarters
                                    # assume the position stays on)
    rebalance_trigger: float = 0.8  # rebalance when short loss >= margin*buffer*trigger


@dataclass
class CarryResult:
    equity_curve: DataFrame         # one row per funding print after entry
    quarterly: DataFrame            # net pnl per calendar quarter (costs allocated in time)
    liquidated: bool
    liquidation_time: str | None
    n_rebalances: int
    gross_funding: float            # sum of funding payments received (positive = earned)
    total_costs: float
    net_pnl: float
    final_equity: float
    annualised_return: float        # CAGR-style on committed equity
    n_funding_prints: int
    used_real_perp: bool = False    # True when the short leg ran on a real perp series

    def summary(self) -> dict:
        return {
            "net_pnl": self.net_pnl,
            "gross_funding": self.gross_funding,
            "total_costs": self.total_costs,
            "final_equity": self.final_equity,
            "annualised_return": self.annualised_return,
            "liquidated": self.liquidated,
            "liquidation_time": self.liquidation_time,
            "n_rebalances": self.n_rebalances,
            "n_funding_prints": self.n_funding_prints,
            "used_real_perp": self.used_real_perp,
        }


def _next_open_idx(dates: np.ndarray, t: pd.Timestamp) -> int | None:
    """Index of the first spot bar strictly after ``t`` (decision -> next-open fill)."""
    i = int(np.searchsorted(dates, np.datetime64(t.asm8), side="right"))
    return i if i < len(dates) else None


def _normalize_dates(df: DataFrame) -> DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["date"] = (
        pd.to_datetime(out["date"], utc=True, format="mixed")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    return out


def run_carry(
    spot: DataFrame,
    funding: DataFrame,
    cfg: CarryConfig | None = None,
    cost: CostModel | None = None,
    perp: DataFrame | None = None,
) -> CarryResult:
    """Causal delta-neutral carry backtest.

    ``spot``: OHLCV frame with ``date``/``open``/``close`` (UTC). ``funding``: output of
    ``load_funding``. One static position: enter after the first funding print, hold to
    the end of data or liquidation, re-hedging before liquidation when configured.

    ``perp`` (optional): the real perpetual-futures price series. When given, the short
    leg is marked and settled on ACTUAL perp prices - no basis proxy, real basis pnl.
    When omitted, the perp is proxied by spot * (1 + basis_bps) (documented limitation).
    """
    cfg = cfg or CarryConfig()
    cost = cost or CostModel()
    role = "maker" if cfg.post_only else "taker"

    df = _normalize_dates(spot)
    f = _normalize_dates(funding)
    if df.empty or f.empty:
        raise ValueError("spot and funding frames must both be non-empty")
    dates = df["date"].to_numpy(dtype="datetime64[ns]")
    opens = df["open"].to_numpy(float)
    closes = df["close"].to_numpy(float)

    pp = _normalize_dates(perp) if perp is not None else None
    p_dates = pp["date"].to_numpy(dtype="datetime64[ns]") if pp is not None else None
    p_closes = pp["close"].to_numpy(float) if pp is not None else None

    # never let one leg outlive the other: the run ends where the SHORTEST series ends.
    # Past that point one side would be marked on a frozen price - a naked position,
    # not a hedge (this is not an approximation, it is wrong data).
    end_t64 = dates[-1] if p_dates is None or len(p_dates) == 0 else min(dates[-1], p_dates[-1])

    basis = cfg.basis_bps / 1e4

    def perp_mark_at(t64: np.datetime64) -> float | None:
        """Real perp close at or before ``t64``; None when no perp data exists yet."""
        if p_dates is None or len(p_dates) == 0:
            return None
        k = int(np.searchsorted(p_dates, t64, side="right")) - 1
        return float(p_closes[k]) if k >= 0 else None

    def perp_price(t64: np.datetime64, spot_ref: float) -> float:
        """Perp price at t: the real series when available, else the basis proxy."""
        real = perp_mark_at(t64)
        return real if real is not None else spot_ref * (1.0 + basis)

    entry_i = _next_open_idx(dates, f["date"].iloc[0])
    if entry_i is None:
        raise ValueError("no spot bar after the first funding print - cannot enter")
    entry_spot = opens[entry_i]
    entry_time = pd.Timestamp(dates[entry_i])

    # capital: spot notional + perp margin = committed capital
    committed = cfg.equity * cfg.notional_fraction
    notional = committed / (1.0 + 1.0 / cfg.leverage)
    margin = notional / cfg.leverage
    coins = notional / entry_spot
    anchor_perp = perp_price(np.datetime64(entry_time.asm8), entry_spot)

    entry_cost = cost.one_way_cost(role) * notional * 2.0  # both legs in
    costs = entry_cost
    cum_funding = 0.0
    realized_delta = 0.0   # hedge pnl banked at each re-hedge
    n_rebalances = 0
    anchor_spot = entry_spot
    liquidated = False
    liq_time: pd.Timestamp | None = None
    exit_i: int | None = None
    rows: list[dict] = []

    prints = f[f["date"] > entry_time].reset_index(drop=True)
    for row in prints.itertuples():
        if np.datetime64(row.date.asm8) > end_t64:
            break  # one leg is out of data - the hedge no longer exists
        # mark the perp at the last spot close at or before the print (causal)
        j = int(np.searchsorted(dates, np.datetime64(row.date.asm8), side="right")) - 1
        if j < entry_i:
            continue
        mark_spot = closes[j]
        mark_perp = perp_price(np.datetime64(row.date.asm8), mark_spot)
        payment = float(row.funding_rate) * coins * mark_perp  # short receives when > 0
        cum_funding += payment
        short_loss = coins * (mark_perp - anchor_perp)
        spot_mtm = coins * (mark_spot - anchor_spot)
        hedge_mtm = realized_delta + spot_mtm - short_loss
        equity = cfg.equity + cum_funding + hedge_mtm - costs
        rows.append(
            {
                "date": row.date,
                "funding_rate": float(row.funding_rate),
                "payment": payment,
                "cum_funding": cum_funding,
                "basis_mtm": hedge_mtm,
                "costs_to_date": costs,
                "equity": equity,
            }
        )
        if short_loss >= margin * cfg.liquidation_buffer:
            liquidated = True
            liq_time = pd.Timestamp(row.date)
            exit_i = _next_open_idx(dates, row.date)
            break
        if (
            cfg.rebalance
            and short_loss >= margin * cfg.liquidation_buffer * cfg.rebalance_trigger
        ):
            # re-hedge before liquidation: close both legs at the mark and reopen at the
            # same mark (constant-dollar hedge). Equity must still cover the margin.
            hedge_cost = cost.one_way_cost(role) * coins * mark_spot * 4.0  # out + in, 2 legs
            realized_delta += spot_mtm - short_loss
            costs += hedge_cost
            coins = notional / mark_spot
            anchor_spot = mark_spot
            anchor_perp = mark_perp
            n_rebalances += 1
            if equity - hedge_cost < margin:
                liquidated = True
                liq_time = pd.Timestamp(row.date)
                exit_i = _next_open_idx(dates, row.date)
                break

    # exit: next open after liquidation, else the last bar where BOTH legs have data
    if exit_i is None and not liquidated:
        exit_i = int(np.searchsorted(dates, end_t64, side="right")) - 1
    if exit_i is not None:
        exit_spot = opens[min(exit_i, len(dates) - 1)]
        exit_cost = cost.one_way_cost(role) * coins * exit_spot * 2.0  # both legs out
        exit_t64 = dates[min(exit_i, len(dates) - 1)]
    else:
        # liquidated with no later bar - flattened at the last mark
        exit_spot = closes[-1]
        exit_cost = cost.one_way_cost("taker") * coins * exit_spot * 2.0  # forced = taker
        exit_t64 = dates[-1]
    costs += exit_cost
    exit_perp = perp_price(exit_t64, exit_spot)

    # realise the exit: each leg closes on its own series (real basis pnl when perp given)
    realized_delta += coins * (exit_spot - anchor_spot) - coins * (exit_perp - anchor_perp)
    final_equity = cfg.equity + cum_funding + realized_delta - costs

    curve = DataFrame(rows)
    if len(curve):
        curve["quarter"] = pd.to_datetime(curve["date"]).dt.to_period("Q").astype(str)
        q = curve.groupby("quarter").agg(
            funding=("payment", "sum"),
            n_prints=("payment", "size"),
        )
        # quarterly net = funding + change in basis MtM; costs allocated when paid
        basis_chg = curve.groupby("quarter")["basis_mtm"].last().diff()
        basis_chg.iloc[0] = curve["basis_mtm"].iloc[0]
        q["basis_mtm_change"] = basis_chg
        q["costs"] = 0.0
        # allocate entry/exit costs to the nearest quarter present in the curve so
        # quarterly net sums always add up to the total net pnl
        entry_q = pd.Timestamp(entry_time).to_period("Q").strftime("%YQ%q")
        q.loc[max(entry_q, q.index[0]), "costs"] += entry_cost
        if exit_i is not None:
            exit_q = pd.Timestamp(dates[min(exit_i, len(dates) - 1)]).to_period("Q").strftime("%YQ%q")
            q.loc[min(exit_q, q.index[-1]), "costs"] += exit_cost
        else:
            q.loc[q.index[-1], "costs"] += exit_cost
        q["net"] = q["funding"] + q["basis_mtm_change"] - q["costs"]
        q = q.reset_index()
    else:
        q = DataFrame(columns=["quarter", "funding", "n_prints", "basis_mtm_change", "costs", "net"])

    span_days = max((pd.Timestamp(end_t64) - entry_time).total_seconds() / 86400.0, 1.0)
    growth = max(final_equity / cfg.equity, 1e-9)
    annualised = growth ** (365.0 / span_days) - 1.0

    return CarryResult(
        equity_curve=curve.drop(columns=["quarter"], errors="ignore"),
        quarterly=q,
        liquidated=liquidated,
        liquidation_time=str(liq_time) if liq_time is not None else None,
        n_rebalances=n_rebalances,
        gross_funding=float(cum_funding),
        total_costs=float(costs),
        net_pnl=float(final_equity - cfg.equity),
        final_equity=float(final_equity),
        annualised_return=float(annualised),
        n_funding_prints=int(len(curve)),
        used_real_perp=perp is not None,
    )
