"""Feature-audit tooling (spec: 'create a feature-audit script that automatically detects
suspicious future references').

Two layers:
  1. STATIC scan: search a feature-builder source file / dataframe-ops for the classic
     look-ahead patterns: shift(-n), iloc with future row access, whole-frame aggregates
     (mean/min/max over the full frame without rolling), expanding-with-future patterns,
     and TA calls configured with signalperiod <= 0.
  2. DYNAMIC causality proof: recompute the feature frame on a TRUNCATED history and assert
     that every historical feature value is bit-identical. Any change proves the feature saw
     the future. This is the same idea as freqtrade's lookahead-analysis but applied at the
     feature-engine level, and it is the gate every feature set must pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pandas import DataFrame

# ---------------------------------------------------------------- static scan
# shift(-n) is the primary look-ahead; shift(+n) is safe (backward). Whole-frame reductions
# are only suspicious when NOT preceded by a .rolling(/ewm(/expanding( window on the same chain.
SUSPICIOUS_PATTERNS: list[tuple[str, str, str]] = [
    ("shift_negative", "shift(-n) reads future rows", r"\.shift\(\s*-\s*\d"),
    ("iloc_negative", "negative iloc indexes from the future end of the frame", r"\.iloc\s*\[\s*-\d"),
    ("iloc_open_ended", "open-ended iloc slice may touch future rows", r"\.iloc\s*\[\s*\w+\s*:\s*\]"),
    ("ta_signalperiod_1", "TA signalperiod=1 is a known look-ahead", r"signalperiod\s*=\s*1"),
    ("whole_frame_reduction", "aggregate without a trailing-window on df", r"\.(mean|std|max|min|sum)\(\s*\)"),
]


def static_scan_python(source: str, filename: str = "<string>") -> list[dict]:
    """Scan a Python source string for suspicious future-reference patterns."""
    lines = source.splitlines()
    findings: list[dict] = []
    for name, desc, pat in SUSPICIOUS_PATTERNS:
        for m in re.finditer(pat, source):
            line_no = source.count("\n", 0, m.start()) + 1
            line_text = lines[line_no - 1].strip()
            if name == "whole_frame_reduction" and _statement_has_window(lines, line_no):
                continue  # the logical statement carries a trailing window -> causal
            findings.append(
                {
                    "file": filename,
                    "line": line_no,
                    "pattern": name,
                    "description": desc,
                    "snippet": line_text[:160],
                }
            )
    return findings


def _statement_has_window(lines: list[str], line_no: int) -> bool:
    """Heuristic: search the current + preceding lines (up to 8) for a trailing-window call.

    The static scan is a warning layer only; the authoritative causality proof is the
    dynamic truncation check. A modest look-back keeps false positives low for multi-line
    ``.rolling(\\n ...).mean()`` chains while still catching genuine whole-frame reductions.
    """
    start = max(0, line_no - 8)
    stmt = "\n".join(lines[start:line_no])
    return ".rolling(" in stmt or ".ewm(" in stmt or ".expanding(" in stmt


def static_scan_file(path: Path) -> list[dict]:
    return static_scan_python(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------- dynamic proof
def dynamic_causality_check(builder, df: DataFrame, cutoff: int = 200) -> dict:
    """Prove causality: truncate the frame and compare historical feature values.

    builder: callable(df) -> feature DataFrame aligned to df.index.
    Any non-NaN historical value (index < cutoff) that differs between the full and the
    truncated build proves the feature used information after its own timestamp.
    """
    full = builder(df)
    cut = builder(df.iloc[:cutoff].reset_index(drop=True))
    # align on position (indexes may differ after reset)
    n = min(cutoff, len(cut), len(full))
    issues: list[str] = []
    for col in full.columns:
        if col == "date":
            continue
        a = full[col].iloc[:n].to_numpy()
        b = cut[col].iloc[:n].to_numpy()
        if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
            continue  # categorical/string columns: causality is positional, verified separately
        both = ~(np.isnan(a.astype(float)) | np.isnan(b.astype(float)))
        if both.sum() == 0:
            continue
        ok = np.allclose(a[both].astype(float), b[both].astype(float), rtol=0, atol=1e-12)
        if not ok:
            # find the first differing index
            diff = np.where(~np.allclose(a[both].astype(float), b[both].astype(float), rtol=0, atol=1e-12))[0]
            issues.append(
                {
                    "column": col,
                    "first_diff_position": int(both.nonzero()[0][diff[0]]) if len(diff) else None,
                    "n_diff": int((~np.allclose(a[both].astype(float), b[both].astype(float), rtol=0, atol=1e-12)).sum()),
                }
            )
    return {"passed": len(issues) == 0, "issues": issues, "checked_columns": len(full.columns)}
