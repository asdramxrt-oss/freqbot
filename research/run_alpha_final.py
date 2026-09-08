"""Final alpha-discovery research (DeepSeek brief Step 5): acceptance verdict + report data.

The vol-spike directional alpha was implemented and evaluated with strict walk-forward
discipline. This module assembles the final evidence and writes the acceptance verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from research_lib.constants import RESEARCH_DIR
from research_lib.validation.acceptance import (
    ARTIFACTS_DIR,
    AcceptanceVerdict,
    GateResult,
    evaluate_pairs_acceptance,
)

# Results captured from the WFO runs (semi-annual windows; 6 testable - the 2022H1 window
# has no prior training history so is structurally untestable).
# Threshold sweep totals (gross/net across 6 windows on $100k).
WFO_SUMMARY = {
    "BTC fade pe=0.4": {"gross_pos_windows": "3/6", "total_gross": 18132, "total_net": 5182},
    "BTC fade pe=0.3": {"gross_pos_windows": "3/6", "total_gross": 8461, "total_net": -13131},
    "BTC follow pe=0.4": {"gross_pos_windows": "3/6", "total_gross": -18132, "total_net": -9956},
    "ETH fade pe=0.4": {"gross_pos_windows": "4/6", "total_gross": 564, "total_net": -13248},
    "ETH fade pe=0.3": {"gross_pos_windows": "2/6", "total_gross": 321, "total_net": -20945},
}

# Directional-agreement diagnostics (per-window mean |agreement - 0.5|): ~0.06-0.12
# across p75/p90, both pairs - statistically at chance.


def evaluate_alpha_acceptance() -> AcceptanceVerdict:
    v = AcceptanceVerdict(strategy="volspike_directional_15m")

    v.gates.append(GateResult(1, "no look-ahead", True,
                              "walk-forward: model trained on prior data only per window; "
                              "decisions at close t fill at open t+1 (unit-tested)"))
    v.gates.append(GateResult(2, "no material leakage", True,
                              "train-only spike thresholds; per-window expanding refits"))
    v.gates.append(GateResult(3, "purged validation passed", False,
                              "no directional signal to validate: agreement with trend/momentum "
                              "at chance across windows (mean |agr-0.5| ~0.06-0.12)"))
    v.gates.append(GateResult(4, "walk-forward performance positive", False,
                              "best variant BTC fade 3/6 gross-positive windows; ETH 4/6 but "
                              "total gross only +$564 (noise); no variant reaches 5/6"))
    v.gates.append(GateResult(5, "CPCV distribution acceptable", False,
                              "directional edge absent; CPCV not meaningful"))
    v.gates.append(GateResult(6, "final holdout positive", False,
                              "not tested - candidate failed walk-forward; holdout untouched"))
    v.gates.append(GateResult(7, "cost sensitivity acceptable", False,
                              "net negative for nearly all variants (fees exceed ~zero gross)"))
    v.gates.append(GateResult(8, "parameter perturbation acceptable", False,
                              "no robust parameter pocket exists; sign of gross flips with "
                              "direction rule and window"))
    v.gates.append(GateResult(9, "regime stress test acceptable", False,
                              "spike-conditional returns track window regime: + in bull "
                              "(-83 to +291 bps), - in bear - leveraged beta, not alpha"))
    v.gates.append(GateResult(10, "Monte Carlo risk acceptable", False,
                              "net negative; not evaluated further"))
    v.gates.append(GateResult(11, "drawdown within risk budget", False,
                              "net negative strategy"))
    v.gates.append(GateResult(12, "sufficient trade count", True,
                              "p75 model yields 5k-11k events (adequate power); result is "
                              "absence of signal, not lack of data"))
    v.gates.append(GateResult(13, "execution assumptions realistic", True,
                              "next-open fills, taker costs, documented futures-style shorts"))
    v.gates.append(GateResult(14, "no single small period explains profits", True,
                              "no profits to explain - failure is uniform across windows"))
    v.gates.append(GateResult(15, "no single symbol explains profits", True,
                              "both BTC and ETH show the same absence of directional signal"))
    v.gates.append(GateResult(16, "no extreme parameter dependence", True,
                              "failure is parameter-flat (no hidden good pocket)"))
    v.gates.append(GateResult(17, "no catastrophic tail behavior", True,
                              "gross magnitudes small; no blowup"))

    n_pass = len(v.passed_gates)
    v.final = (
        "PRODUCTION READY" if n_pass == len(v.gates)
        else "PAPER ONLY" if n_pass >= 10
        else "RESEARCH ONLY" if n_pass >= 6
        else "REJECT"
    )
    return v


def main():
    verdict = evaluate_alpha_acceptance()
    print(f"\n=== ACCEPTANCE VERDICT: {verdict.strategy} ===")
    print(f"FINAL: {verdict.final}  ({len(verdict.passed_gates)}/{len(verdict.gates)} gates passed)")
    for g in verdict.gates:
        print(f"  [{'PASS' if g.passed else 'FAIL'}] Gate {g.gate_id:2d} {g.name}: {g.evidence}")
    out = ARTIFACTS_DIR / "acceptance" / "verdict_volspike_alpha.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict.to_dict(), indent=2))
    print(f"\nverdict saved: {out}")


if __name__ == "__main__":
    main()
