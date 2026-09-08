"""Performance metrics for trade logs and equity curves.

Single source of truth for every report: per-trade metrics (win rate, expectancy, profit
factor, MFE/MAE) and time-series metrics (Sharpe, Sortino, Calmar, CAGR, max drawdown)
computed from a trade log or an equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def _annualization(timeframe: str) -> float:
    return {
        "1m": 525600,
        "5m": 105120,
        "15m": 35040,
        "1h": 8760,
        "4h": 2190,
        "1d": 365,
    }.get(timeframe, 8760)


def equity_curve_from_trades(trades: DataFrame, start_equity: float, timeframe: str) -> Series:
    """Build a DAILY-calendar equity curve from a trade log (pnl_net per trade).

    Trades are marked to their exit date; days without exits carry the previous equity
    forward (0 PnL) so returns are measured on a fixed, regular grid. This is the only
    defensible way to compute Sharpe/Sortino on sparse trade streams - computing returns
    over gaps between trade days massively inflates annualized ratios.
    """
    if trades is None or len(trades) == 0:
        idx = pd.date_range("2022-01-01", periods=2, freq="D", tz="UTC")
        return Series(np.nan, index=idx)
    df = trades.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df = df.sort_values("exit_time")
    daily = df.groupby(df["exit_time"].dt.normalize())["pnl_net"].sum()
    # fixed daily calendar spanning first exit .. last exit
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(full_idx, fill_value=0.0)
    eq = start_equity + daily.cumsum()
    eq.index = pd.to_datetime(eq.index, utc=True)
    return eq


def trade_metrics(trades: DataFrame) -> dict:
    """Per-trade summary statistics (numpy-native; works on the trade log)."""
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": np.nan, "expectancy": np.nan,
            "profit_factor": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
            "total_pnl": 0.0, "avg_bars_held": np.nan, "avg_mfe": np.nan, "avg_mae": np.nan,
        }
    pnl = trades["pnl_net"].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "expectancy": float(pnl.mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "total_pnl": float(pnl.sum()),
        "avg_bars_held": float(trades["bars_held"].mean()) if "bars_held" in trades else np.nan,
        "avg_mfe": float(trades["mfe"].mean()) if "mfe" in trades else np.nan,
        "avg_mae": float(trades["mae"].mean()) if "mae" in trades else np.nan,
    }


def series_metrics(eq: Series, timeframe: str = "1h", periods_per_year: int | None = None) -> dict:
    """Risk/return metrics from an equity curve (nan when insufficient data).

    ``eq`` is expected on a REGULAR calendar (see equity_curve_from_trades). When
    ``periods_per_year`` is not given it is inferred from the median spacing of the series
    index; a daily equity curve therefore annualizes by 365 regardless of the bar
    timeframe of the strategy that produced it.
    """
    if eq is None or len(eq.dropna()) < 10:
        return {"sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan, "cagr": np.nan,
                "calmar": np.nan, "total_return": np.nan}
    eq = eq.dropna()
    rets = eq.pct_change().dropna()
    if len(rets) < 10:
        return {"sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan, "cagr": np.nan,
                "calmar": np.nan, "total_return": np.nan}
    if periods_per_year is not None:
        ppy = periods_per_year
    else:
        # infer from the index spacing (daily => 365)
        try:
            spacing = eq.index.to_series().diff().dt.total_seconds().median()
            ppy = 365.25 * 86400 / spacing if spacing and spacing > 0 else 365
        except Exception:  # noqa: BLE001
            ppy = 365
    mu = rets.mean()
    sd = rets.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(ppy)) if sd and sd > 0 else np.nan
    downside = rets[rets < 0].std(ddof=1)
    sortino = float(mu / downside * np.sqrt(ppy)) if downside and downside > 0 else np.nan
    # drawdown from running max
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = float(dd.min())
    n_years = len(eq) / ppy
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 and eq.iloc[0] > 0 else np.nan
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 and np.isfinite(cagr) else np.nan
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1) if eq.iloc[0] > 0 else np.nan
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cagr": cagr,
        "calmar": calmar,
        "total_return": total_return,
        "n_periods": int(len(eq)),
        "annualization": ppy,
    }


def summarize_backtest(
    trades: DataFrame, equity_curve: Series | None, timeframe: str = "1h", start_equity: float = 100_000.0
) -> dict:
    """Full backtest summary: trade metrics + series metrics."""
    out = trade_metrics(trades)
    eq = equity_curve if equity_curve is not None else equity_curve_from_trades(trades, start_equity, timeframe)
    out.update(series_metrics(eq, timeframe))
    # turnover proxy: sum notional / equity (single-asset)
    if trades is not None and len(trades):
        out["turnover"] = float(trades["notional"].sum() / max(start_equity, 1))
        out["fees_total"] = float(trades["fees_paid"].sum())
    else:
        out["turnover"] = 0.0
        out["fees_total"] = 0.0
    return out


def summarize_trades_panel(trades: DataFrame, timeframe: str = "1h", start_equity: float = 100_000.0) -> dict:
    return summarize_backtest(trades, None, timeframe, start_equity)
