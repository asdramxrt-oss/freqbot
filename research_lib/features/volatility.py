"""Unified volatility engine (Phase 3).

Estimators (all configurable):
  * close_to_close   - classic realized vol of log returns
  * ewma            - RiskMetrics-style EWMA vol
  * parkinson        - high/low range vol
  * garman_klass     - OHLC vol (with session handling)
  * rogers_satchell  - drift-adjusted OHLC vol
  * yang_zhang       - overnight+intraday combined vol

Crypto/24-7 session handling:
  Yang-Zhang's power comes from separating *overnight* gaps (close[t-1] -> open[t]) from
  *intraday* movement. For a continuously traded market there is no natural session boundary,
  so the engine takes an explicit ``session`` definition. Two modes are supported:
    - ``session="cont"``  : no overnight gap. open[t] is treated as a continuation of the
                            prior close; the "overnight" term is dropped and YZ degenerates to
                            the Rogers-Satchell + close-to-close blend (documented below).
    - ``session="utc_day"``: explicit daily boundary at 00:00 UTC; open[t] is the first price
                            after the boundary and the standard YZ decomposition applies.
  The choice is a configuration decision, not an assumption, and must be validated OOS.

Every estimator output is strictly causal: value at bar t uses only bars <= t.
The engine also exposes derived causal series:
  volatility_percentile  - rolling percentile rank of vol (window configurable)
  volatility_zscore      - rolling z-score of vol
  volatility_regime      - quantile bucket: LOW / NORMAL / HIGH / EXTREME
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

VOL_REGIME_EDGES = [0.25, 0.75, 0.95]  # LOW < p25 <= NORMAL < p75 <= HIGH < p95 <= EXTREME


def _log_returns(df: DataFrame) -> Series:
    return np.log(df["close"]).diff()


class VolatilityEstimator:
    """Computes a single volatility estimator series over a canonical OHLC frame."""

    def __init__(
        self,
        method: str = "yang_zhang",
        window: int = 20,
        session: str = "cont",
        min_periods: int | None = None,
    ) -> None:
        self.method = method
        self.window = window
        self.session = session
        self.min_periods = min_periods if min_periods is not None else max(2, window // 2)
        if method not in {
            "close_to_close",
            "ewma",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
            "yang_zhang",
        }:
            raise ValueError(f"Unknown volatility method {method}")

    # ------------------------------------------------------------------
    def _period_vol(self, squared: Series) -> Series:
        """Annualization-neutral rolling mean of squared returns (kept in bar units)."""
        return squared.rolling(self.window, min_periods=self.min_periods).mean()

    def _close_to_close(self, df: DataFrame) -> Series:
        r = _log_returns(df)
        return np.sqrt(self._period_vol(r**2))

    def _ewma(self, df: DataFrame) -> Series:
        r = _log_returns(df)
        return r.ewm(span=self.window, min_periods=self.min_periods).std()

    def _parkinson(self, df: DataFrame) -> Series:
        hl = np.log(df["high"] / df["low"])
        # Parkinson: sigma^2 = (1/(4 ln2)) * mean(hl^2)
        return np.sqrt(self._period_vol(hl**2) / (4.0 * np.log(2.0)))

    def _garman_klass(self, df: DataFrame) -> Series:
        o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
        hl = np.log(h / l)
        co = np.log(c / o)
        # GK variance can be negative bar-by-bar; the ROLLED mean is the estimator, clip after.
        var = (0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2).rolling(
            self.window, min_periods=self.min_periods
        ).mean()
        return np.sqrt(var.clip(lower=0))

    def _rogers_satchell(self, df: DataFrame) -> Series:
        o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
        # Standard RS: sigma^2 = E[ ln(h/o)ln(h/c) + ln(l/o)ln(l/c) ]
        ho = np.log(h / o)
        lo = np.log(l / o)
        hc = np.log(h / c)
        lc = np.log(l / c)
        var = (ho * hc + lo * lc).rolling(self.window, min_periods=self.min_periods).mean()
        return np.sqrt(var.clip(lower=0))

    def _yang_zhang(self, df: DataFrame) -> Series:
        o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
        # Rogers-Satchell intraday term (drift-adjusted), correct pairing
        ho = np.log(h / o)
        lo = np.log(l / o)
        hc = np.log(h / c)
        lc = np.log(l / c)
        rs = ho * hc + lo * lc
        if self.session == "cont":
            # No overnight gap: open[t] continues close[t-1], so the classic YZ overnight
            # term is identically zero and YZ reduces to the RS variance estimator. This is
            # the correct degeneration for a 24/7 market and is documented as such.
            var = rs.rolling(self.window, min_periods=self.min_periods).mean()
            return np.sqrt(var.clip(lower=1e-20))
        # Standard YZ with a real session boundary (overnight gap + intraday)
        overnight = np.log(o / c.shift(1))
        o_var = self._period_vol(overnight**2)
        c_var = self._period_vol((np.log(c / o)) ** 2)
        rs_var = rs.rolling(self.window, min_periods=self.min_periods).mean()
        k = 0.34 / (1.0 + (self.window + 1.0) / (self.window - 1.0))
        var = o_var + k * c_var + (1.0 - k) * rs_var
        return np.sqrt(var.clip(lower=1e-20))

    # ------------------------------------------------------------------
    def compute(self, df: DataFrame) -> Series:
        out: Series
        if self.method == "close_to_close":
            out = self._close_to_close(df)
        elif self.method == "ewma":
            out = self._ewma(df)
        elif self.method == "parkinson":
            out = self._parkinson(df)
        elif self.method == "garman_klass":
            out = self._garman_klass(df)
        elif self.method == "rogers_satchell":
            out = self._rogers_satchell(df)
        elif self.method == "yang_zhang":
            out = self._yang_zhang(df)
        else:  # pragma: no cover
            raise ValueError(self.method)
        return out.rename(f"vol_{self.method}").replace(0.0, np.nan)


class VolatilityEngine:
    """Exposes vol + percentile + zscore + regime for a canonical frame (all causal)."""

    def __init__(
        self,
        method: str = "yang_zhang",
        window: int = 20,
        session: str = "cont",
        percentile_window: int = 500,
        regime_window: int = 500,
    ) -> None:
        self.estimator = VolatilityEstimator(method=method, window=window, session=session)
        self.percentile_window = percentile_window
        self.regime_window = regime_window

    def compute(self, df: DataFrame) -> DataFrame:
        vol = self.estimator.compute(df)
        out = DataFrame({"volatility": vol}, index=df.index)
        # Rolling percentile rank (causal: uses only trailing window)
        out["volatility_percentile"] = (
            vol.rolling(self.percentile_window, min_periods=100)
            .rank(pct=True)
        )
        # Rolling z-score
        roll = vol.rolling(self.percentile_window, min_periods=100)
        mu = roll.mean()
        sd = roll.std()
        out["volatility_zscore"] = (vol - mu) / sd.replace(0, np.nan)
        # Regime buckets from the trailing percentile
        regime = pd.cut(
            out["volatility_percentile"],
            bins=[-np.inf, VOL_REGIME_EDGES[0], VOL_REGIME_EDGES[1], VOL_REGIME_EDGES[2], np.inf],
            labels=["LOW", "NORMAL", "HIGH", "EXTREME"],
        )
        out["volatility_regime"] = regime.astype("string")
        return out


def compute_volatility_panel(
    panel: dict[tuple[str, str], DataFrame],
    method: str = "yang_zhang",
    window: int = 20,
    session: str = "cont",
) -> dict[tuple[str, str], DataFrame]:
    return {
        key: VolatilityEngine(method=method, window=window, session=session).compute(df)
        for key, df in panel.items()
    }
