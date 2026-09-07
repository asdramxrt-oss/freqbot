"""Data-quality validation for the canonical market-data contract.

Checks (each returns a per-issue report; a frame with any HARD issue must not feed research):
  * duplicate timestamps
  * missing / out-of-order timestamps
  * zero / negative prices
  * impossible OHLC (high < max(open,close), low > min(open,close), etc.)
  * non-finite values
  * abnormal volume (zero-volume runs, extreme z-score spikes)
  * suspicious flat-price runs (often an artifact of bad fills / synthetic repair)
  * exchange-specific anomaly hooks (extensible)

Rules:
  - Never forward-fill prices blindly (freqtrade's own fill does this at its boundary;
    the research layer must KNOW about every synthetic row).
  - Never use future values to repair historical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.constants import TIMEFRAME_SECONDS


@dataclass
class DataQualityIssue:
    check: str
    severity: str  # "hard" (must fix / refuse) or "soft" (warn)
    index: int | None
    timestamp: pd.Timestamp | None
    message: str


@dataclass
class DataQualityReport:
    frame_name: str
    issues: list[DataQualityIssue] = field(default_factory=list)

    @property
    def hard_issues(self) -> list[DataQualityIssue]:
        return [i for i in self.issues if i.severity == "hard"]

    @property
    def soft_issues(self) -> list[DataQualityIssue]:
        return [i for i in self.issues if i.severity == "soft"]

    @property
    def passed(self) -> bool:
        return len(self.hard_issues) == 0

    def summary(self) -> str:
        return (
            f"{self.frame_name}: {len(self.issues)} issues "
            f"({len(self.hard_issues)} hard, {len(self.soft_issues)} soft) "
            f"- {'PASS' if self.passed else 'FAIL'}"
        )

    def to_frame(self) -> DataFrame:
        return DataFrame(
            [
                {
                    "check": i.check,
                    "severity": i.severity,
                    "index": i.index,
                    "timestamp": i.timestamp,
                    "message": i.message,
                }
                for i in self.issues
            ]
        )


def _tf_seconds(timeframe: str) -> int:
    return TIMEFRAME_SECONDS.get(timeframe, 3600)


class DataQualityValidator:
    """Runs the full battery of quality checks over a canonical frame."""

    def __init__(
        self,
        name: str = "frame",
        timeframe: str = "1h",
        *,
        volume_z_threshold: float = 20.0,
        max_pct_missing: float = 0.02,
        extra_checks: list[Callable[[DataFrame], list[DataQualityIssue]]] | None = None,
    ) -> None:
        self.name = name
        self.timeframe = timeframe
        self.volume_z_threshold = volume_z_threshold
        self.max_pct_missing = max_pct_missing
        self.extra_checks = extra_checks or []

    # ------------------------------------------------------------------
    def _exp_ts(self, df: DataFrame) -> pd.DatetimeIndex:
        """Expected regular timestamp grid for the frame's span."""
        start = df["date"].iloc[0]
        end = df["date"].iloc[-1]
        step = pd.Timedelta(seconds=_tf_seconds(self.timeframe))
        return pd.date_range(start, end, freq=step, tz="UTC")

    def check_duplicates(self, df: DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        dup = df["date"].duplicated(keep="first")
        for idx in np.where(dup)[0][:20]:
            issues.append(
                DataQualityIssue(
                    "duplicate_timestamp",
                    "hard",
                    int(idx),
                    df["date"].iloc[int(idx)],
                    "duplicate timestamp",
                )
            )
        if int(dup.sum()) > 20:
            issues.append(
                DataQualityIssue(
                    "duplicate_timestamp", "hard", None, None, f"...and {int(dup.sum()) - 20} more"
                )
            )
        return issues

    def check_missing_and_order(self, df: DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        if not df["date"].is_monotonic_increasing:
            issues.append(
                DataQualityIssue(
                    "out_of_order", "hard", None, None, "date column is not monotonically increasing"
                )
            )
        expected = self._exp_ts(df)
        missing = expected.difference(pd.DatetimeIndex(df["date"]))
        if len(missing) > 0:
            pct = len(missing) / len(expected)
            sev = "hard" if pct > self.max_pct_missing else "soft"
            issues.append(
                DataQualityIssue(
                    "missing_timestamps",
                    sev,
                    None,
                    None,
                    f"{len(missing)} missing timestamps ({pct:.2%} of expected span)",
                )
            )
        return issues

    def check_prices(self, df: DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        for col in ("open", "high", "low", "close"):
            bad = ~np.isfinite(df[col]) | (df[col] <= 0)
            if bad.any():
                idx = int(np.where(bad)[0][0])
                issues.append(
                    DataQualityIssue(
                        "nonpositive_price",
                        "hard",
                        idx,
                        df["date"].iloc[idx],
                        f"{col} <= 0 or non-finite at row {idx}",
                    )
                )
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        bad_high = h < np.maximum(o, c) - 1e-12
        bad_low = l > np.minimum(o, c) + 1e-12
        for mask, label in ((bad_high, "high < max(open,close)"), (bad_low, "low > min(open,close)")):
            if mask.any():
                idx = int(np.where(mask)[0][0])
                issues.append(
                    DataQualityIssue(
                        "impossible_ohlc",
                        "hard",
                        idx,
                        df["date"].iloc[idx],
                        f"{label} at row {idx}",
                    )
                )
        return issues

    def check_volume(self, df: DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        vol = df["volume"].to_numpy(dtype=float)
        if not np.isfinite(vol).all() or (vol < 0).any():
            issues.append(DataQualityIssue("negative_volume", "hard", None, None, "volume < 0 or NaN"))
        # Extreme volume z-scores (log-normal-ish guard)
        lv = np.log1p(vol)
        med = np.median(lv)
        mad = np.median(np.abs(lv - med)) + 1e-12
        z = 0.6745 * (lv - med) / mad
        bad = np.abs(z) > self.volume_z_threshold
        if bad.sum() > 0:
            idx = int(np.where(bad)[0][0])
            issues.append(
                DataQualityIssue(
                    "abnormal_volume",
                    "soft",
                    idx,
                    df["date"].iloc[idx],
                    f"{int(bad.sum())} extreme volume rows (|MAD-z|>{self.volume_z_threshold})",
                )
            )
        # zero-volume runs longer than 3 bars -> possible data hole
        zero_run = 0
        max_run = 0
        for v in vol:
            zero_run = zero_run + 1 if v == 0 else 0
            max_run = max(max_run, zero_run)
        if max_run > 3:
            issues.append(
                DataQualityIssue(
                    "zero_volume_run", "soft", None, None, f"longest zero-volume run = {max_run} bars"
                )
            )
        return issues

    def check_flat_runs(self, df: DataFrame) -> list[DataQualityIssue]:
        """Suspicious identical consecutive closes (synthetic/repair artifacts)."""
        close = df["close"].to_numpy()
        if len(close) == 0:
            return []
        diffs = np.abs(np.diff(close)) < 1e-12
        max_run = 0
        run = 0
        for d in diffs:
            run = run + 1 if d else 0
            max_run = max(max_run, run)
        if max_run >= 5:
            issues = [
                DataQualityIssue(
                    "flat_price_run",
                    "soft",
                    None,
                    None,
                    f"{max_run + 1} consecutive identical closes (possible synthetic fill)",
                )
            ]
        else:
            issues = []
        return issues

    # ------------------------------------------------------------------
    def validate(self, df: DataFrame) -> DataQualityReport:
        report = DataQualityReport(self.name)
        if len(df) == 0:
            report.issues.append(
                DataQualityIssue("empty_frame", "hard", None, None, "dataframe is empty")
            )
            return report
        for check in (
            self.check_duplicates,
            self.check_missing_and_order,
            self.check_prices,
            self.check_volume,
            self.check_flat_runs,
        ):
            report.issues.extend(check(df))
        for extra in self.extra_checks:
            report.issues.extend(extra(df))
        return report


def validate_panel(
    panel: dict[tuple[str, str], DataFrame], timeframe: str = "1h"
) -> dict[tuple[str, str], DataQualityReport]:
    """Run the full battery across a {(pair, tf): frame} panel."""
    reports: dict[tuple[str, str], DataQualityReport] = {}
    for key, df in panel.items():
        reports[key] = DataQualityValidator(name=f"{key[0]} {key[1]}", timeframe=timeframe).validate(df)
    return reports


def panel_passed(reports: dict[tuple[str, str], DataQualityReport]) -> bool:
    return all(r.passed for r in reports.values())


def drop_hard_issue_rows(df: DataFrame, timeframe: str = "1h") -> DataFrame:
    """Remove rows flagged by hard checks that are row-local (non-positive prices).

    NOTE: this does NOT repair missing timestamps - gaps are left as-is and downstream
    code must treat them as gaps (no blind forward-fill).
    """
    out = df.copy()
    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    bad = (
        ~np.isfinite(o)
        | ~np.isfinite(h)
        | ~np.isfinite(l)
        | ~np.isfinite(c)
        | (o <= 0)
        | (h <= 0)
        | (l <= 0)
        | (c <= 0)
        | (h < np.maximum(o, c) - 1e-12)
        | (l > np.minimum(o, c) + 1e-12)
    )
    return out.loc[~bad].reset_index(drop=True)
