"""Monte Carlo robustness (Phase 18) + parameter robustness (Phase 26 neighborhood tests).

Monte Carlo perturbations over the realized trade set:
  * trade-order reshuffling (random permutation of trade sequence)
  * bootstrap resampling (sample with replacement)
  * slippage variation (add N(0, sigma_slip) bps to each trade)
  * fee variation (multiply fees by a noisy factor)
  * execution delay (drop/skip a fraction of trades to mimic missed fills)
Measures per simulation:
  total PnL, equity drawdown distribution, risk of ruin (equity < ruin_threshold),
  probability of negative performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class MCConfig:
    n_sims: int = 1000
    seed: int = 42
    ruin_fraction: float = 0.5       # ruin if equity < start * ruin_fraction
    slippage_bps_sd: float = 2.0     # extra slippage noise std (bps per trade)
    fee_noise_sd: float = 0.2        # multiplicative fee noise std
    missed_trade_frac: float = 0.0   # fraction of trades dropped (execution delay proxy)


class MonteCarlo:
    def __init__(self, cfg: MCConfig | None = None):
        self.cfg = cfg or MCConfig()
        self.rng = np.random.RandomState(cfg.seed if cfg else 42)

    def _simulate(self, trades: DataFrame, start_equity: float) -> dict:
        cfg = self.cfg
        n = len(trades)
        if n == 0:
            return {"final_equity": start_equity, "pnl": 0.0, "max_dd": 0.0, "ruined": False}
        pnl = trades["pnl_net"].to_numpy(dtype=float).copy()
        # fee/slippage noise applied per trade
        slip = self.rng.normal(0, cfg.slippage_bps_sd / 1e4, n)
        fee_mult = np.exp(self.rng.normal(0, cfg.fee_noise_sd, n))
        notional = trades["notional"].to_numpy(dtype=float)
        # slippage hits notional; fee noise scales the fee component of pnl
        pnl = pnl - notional * slip - (trades["fees_paid"].to_numpy(dtype=float) * (fee_mult - 1.0))
        # missed trades (execution delay)
        if cfg.missed_trade_frac > 0:
            keep = self.rng.random(n) >= cfg.missed_trade_frac
            pnl = pnl[keep]
        # order reshuffle OR bootstrap
        if self.rng.random() < 0.5:
            order = self.rng.permutation(len(pnl))
        else:
            order = self.rng.randint(0, len(pnl), size=len(pnl))
        pnl = pnl[order]
        eq = start_equity + np.cumsum(pnl)
        final_eq = float(eq[-1])
        roll_max = np.maximum.accumulate(eq)
        dd = float((eq / roll_max - 1.0).min())
        ruined = bool(final_eq < start_equity * cfg.ruin_fraction)
        return {"final_equity": final_eq, "pnl": final_eq - start_equity, "max_dd": dd, "ruined": ruined}

    def run(self, trades: DataFrame, start_equity: float) -> DataFrame:
        rows = [self._simulate(trades, start_equity) for _ in range(self.cfg.n_sims)]
        return DataFrame(rows)

    def summarize(self, trades: DataFrame, start_equity: float) -> dict:
        sim = self.run(trades, start_equity)
        pnl = sim["pnl"].to_numpy()
        dd = sim["max_dd"].to_numpy()
        out = {
            "p_pnl_negative": float((pnl < 0).mean()),
            "p_ruin": float(sim["ruined"].mean()),
            "pnl_p5": float(np.percentile(pnl, 5)),
            "pnl_median": float(np.percentile(pnl, 50)),
            "pnl_p95": float(np.percentile(pnl, 95)),
            "dd_p5": float(np.percentile(dd, 5)),      # worst 5% drawdown
            "dd_median": float(np.percentile(dd, 50)),
            "dd_p95": float(np.percentile(dd, 95)),
            "n_sims": self.cfg.n_sims,
        }
        return out


# ---------------------------------------------------------------------------
# parameter neighborhood robustness (Phase 26): a strategy is robust only if performance
# does not collapse immediately outside one exact parameter combination.
# ---------------------------------------------------------------------------
def parameter_grid_robustness(
    run_fn,
    base_params: dict[str, float],
    neighborhood: dict[str, list],
) -> DataFrame:
    """Evaluate run_fn(params) across a neighborhood grid.

    run_fn: callable(params: dict) -> dict of metrics (must include 'sharpe' or 'total_pnl').
    Returns a DataFrame with one row per parameter combination and the metric values; the
    caller decides plateau acceptance (std / worst-case across the grid).
    """
    import itertools

    keys = list(neighborhood.keys())
    combos = list(itertools.product(*(neighborhood[k] for k in keys)))
    rows = []
    for combo in combos:
        params = dict(base_params)
        params.update(dict(zip(keys, combo)))
        try:
            m = run_fn(params)
        except Exception as e:  # noqa: BLE001 - report and continue
            m = {"error": str(e)}
        m = dict(m)
        m.update(zip(keys, combo))
        rows.append(m)
    return DataFrame(rows)


def plateau_acceptance(grid: DataFrame, metric: str, threshold_frac: float = 0.5) -> dict:
    """Check whether performance holds across the grid (plateau) vs a single spike.

    threshold_frac: the fraction of grid points that must remain profitable relative to the
    base case.
    """
    if metric not in grid or len(grid) == 0:
        return {"accepted": False, "reason": "no data"}
    base = grid[metric].median()
    if not np.isfinite(base):
        return {"accepted": False, "reason": "non-finite median"}
    worst = grid[metric].min()
    frac_positive = float((grid[metric] > 0).mean())
    accepted = frac_positive >= threshold_frac and worst > 0
    return {
        "accepted": accepted,
        "median": float(base),
        "worst": float(worst),
        "p25": float(grid[metric].quantile(0.25)),
        "p75": float(grid[metric].quantile(0.75)),
        "frac_positive": frac_positive,
        "n_combos": int(len(grid)),
    }
