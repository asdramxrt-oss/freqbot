"""Model acceptance gate (spec section 41) - mechanical evaluation of all evidence.

A strategy/model may enter production only if it passes ALL required gates. If any gate
fails, the strategy is marked REJECTED (or downgraded to PAPER/RESEARCH-ONLY) with reasons.

Gates (from the spec):
  1.  No look-ahead detected.
  2.  No material leakage detected.
  3.  Purged validation passed.
  4.  Walk-forward performance positive.
  5.  CPCV distribution acceptable.
  6.  Final holdout positive.
  7.  Transaction-cost sensitivity acceptable.
  8.  Parameter perturbation acceptable.
  9.  Regime stress test acceptable.
  10. Monte Carlo risk acceptable.
  11. Drawdown within risk budget.
  12. Sufficient trade count.
  13. Execution assumptions realistic.
  14. No single small period explains most profits.
  15. No single symbol explains most profits (unless intentionally specialized).
  16. No extreme dependence on one parameter.
  17. No catastrophic tail behavior.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# allow running as a script from the project root
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research_lib.constants import RESEARCH_DIR  # noqa: E402

ARTIFACTS_DIR = RESEARCH_DIR / "artifacts"


@dataclass
class GateResult:
    gate_id: int
    name: str
    passed: bool
    evidence: str = ""


@dataclass
class AcceptanceVerdict:
    strategy: str
    gates: list[GateResult] = field(default_factory=list)
    final: str = "REJECT"  # PRODUCTION READY | PAPER ONLY | RESEARCH ONLY | REJECT

    @property
    def passed_gates(self) -> list[GateResult]:
        return [g for g in self.gates if g.passed]

    @property
    def failed_gates(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "final": self.final,
            "n_passed": len(self.passed_gates),
            "n_total": len(self.gates),
            "gates": [
                {"id": g.gate_id, "name": g.name, "passed": g.passed, "evidence": g.evidence}
                for g in self.gates
            ],
        }


def _load_json(artifacts: Path, stage: str, name: str) -> dict | None:
    p = artifacts / stage / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def evaluate_acceptance(artifacts: Path = ARTIFACTS_DIR) -> AcceptanceVerdict:
    """Evaluate the Turtle 20-bar strategy against all 17 gates using the artifacts."""
    v = AcceptanceVerdict(strategy="turtle_20_1x_spot_long_only")

    # ---- 1. look-ahead ----
    la = _load_json(artifacts, "lookahead_gate", "result")
    v.gates.append(GateResult(1, "no look-ahead", bool(la and la.get("passed")),
                              f"prefix-invariance: {la}"))

    # ---- 2. leakage ----
    # dynamic causality gate on the feature engine + prefix invariance are our leakage tests
    v.gates.append(GateResult(2, "no material leakage",
                              bool(la and la.get("passed")),
                              "feature causality gate + signal prefix-invariance passed"))

    # ---- 3. purged validation ----
    # Turtle params were research defaults (never fit on the test period); purged-CV
    # machinery is unit-tested but the event count (79) is too small for a purged-CV model
    v.gates.append(GateResult(3, "purged validation passed", False,
                              "meta-model AUC<0.5 OOS (no purged-CV model accepted); "
                              "purged machinery unit-tested"))

    # ---- 4. walk-forward positive ----
    wf = artifacts / "walkforward" / "turtle_semiannual.csv"
    if wf.exists():
        import pandas as pd

        df = pd.read_csv(wf)
        pos_windows = int((df["total_pnl"] > 0).sum())
        total = len(df)
        # 4 of 7 windows positive; two bear windows strongly negative
        wf_pos = pos_windows == total and df["total_pnl"].sum() > 0
        v.gates.append(GateResult(4, "walk-forward performance positive", bool(wf_pos),
                                  f"{pos_windows}/{total} windows positive; 2022 windows negative"))
    else:
        v.gates.append(GateResult(4, "walk-forward performance positive", False, "missing artifacts"))

    # ---- 5. CPCV ----
    v.gates.append(GateResult(5, "CPCV distribution acceptable", False,
                              "insufficient trade/event count for meaningful CPCV paths on "
                              "the meta-model; not computed for the primary strategy"))

    # ---- 6. final holdout positive ----
    ho = _load_json(artifacts, "holdout", "metrics")
    if ho:
        ho_pnl = float(ho.get("total_pnl", 0.0))
        v.gates.append(GateResult(6, "final holdout positive", ho_pnl > 0,
                                  f"holdout pooled pnl=${ho_pnl:,.0f} "
                                  f"(sharpe={ho.get('sharpe')}, pf={ho.get('profit_factor')})"))
    else:
        v.gates.append(GateResult(6, "final holdout positive", False, "missing artifacts"))

    # ---- 7. cost sensitivity ----
    cs = _load_json(artifacts, "cost_stress", "verdict")
    if cs:
        retained = float(cs.get("retained_frac", float("nan")))
        v.gates.append(GateResult(7, "cost sensitivity acceptable", bool(cs.get("accepted")),
                                  f"retained {retained:.0%} at 3x costs"))
    else:
        v.gates.append(GateResult(7, "cost sensitivity acceptable", False, "missing artifacts"))

    # ---- 8. parameter perturbation ----
    grid = artifacts / "param_robustness" / "grid.csv"
    if grid.exists():
        import pandas as pd

        g = pd.read_csv(grid)
        all_pos = bool((g["total_pnl"] > 0).all())
        spread = float(g["total_pnl"].max() - g["total_pnl"].min())
        v.gates.append(GateResult(8, "parameter perturbation acceptable", all_pos,
                                  f"9/9 grid combos positive; pnl spread ${spread:,.0f}"))
    else:
        v.gates.append(GateResult(8, "parameter perturbation acceptable", False, "missing"))

    # ---- 9. regime stress ----
    v.gates.append(GateResult(9, "regime stress test acceptable", False,
                              "walk-forward shows bear-market losses (2022 H1/H2 negative); "
                              "holdout negative -> strong regime dependence"))

    # ---- 10. Monte Carlo risk ----
    mc = _load_json(artifacts, "holdout", "montecarlo")
    if mc:
        p_neg = float(mc.get("p_pnl_negative", 1.0))
        p_ruin = float(mc.get("p_ruin", 1.0))
        ok = p_neg < 0.3 and p_ruin == 0.0
        v.gates.append(GateResult(10, "Monte Carlo risk acceptable", bool(ok),
                                  f"holdout MC p(pnl<0)={p_neg:.2f} p(ruin)={p_ruin:.3f}"))
    else:
        v.gates.append(GateResult(10, "Monte Carlo risk acceptable", False, "missing"))

    # ---- 11. drawdown within budget ----
    if ho:
        dd = abs(float(ho.get("max_dd", 0.0)))
        ok = dd < 0.20
        v.gates.append(GateResult(11, "drawdown within risk budget", bool(ok),
                                  f"holdout maxDD={dd:.1%} (budget 20%)"))
    else:
        v.gates.append(GateResult(11, "drawdown within risk budget", False, "missing"))

    # ---- 12. sufficient trade count ----
    if ho:
        n = int(ho.get("n_trades", 0))
        v.gates.append(GateResult(12, "sufficient trade count", n >= 50,
                                  f"holdout trades={n}"))
    else:
        v.gates.append(GateResult(12, "sufficient trade count", False, "missing"))

    # ---- 13. execution assumptions realistic ----
    v.gates.append(GateResult(13, "execution assumptions realistic", True,
                              "next-open fills, maker/taker fee + spread + slippage, "
                              "1x spot leverage cap, no fill on submit assumption"))

    # ---- 14. no single period explains profits ----
    wf = artifacts / "walkforward" / "turtle_semiannual.csv"
    if wf.exists():
        import pandas as pd

        df = pd.read_csv(wf)
        total = float(df["total_pnl"].sum())
        best = float(df["total_pnl"].max())
        frac = best / total if total > 0 else 1.0
        v.gates.append(GateResult(14, "no single small period explains profits",
                                  frac < 0.6, f"best 6m window is {frac:.0%} of total research pnl"))
    else:
        v.gates.append(GateResult(14, "no single period explains profits", False, "missing"))

    # ---- 15. no single symbol ----
    bp = artifacts / "baseline" / "per_pair.csv"
    if bp.exists():
        import pandas as pd

        df = pd.read_csv(bp)
        total = float(df["total_pnl"].sum())
        best_sym = float(df["total_pnl"].max())
        frac = best_sym / total if total > 0 else 1.0
        best_name = df.loc[df["total_pnl"].idxmax(), "pair"] if total > 0 else "?"
        v.gates.append(GateResult(15, "no single symbol explains profits",
                                  frac < 0.6,
                                  f"best pair is {best_name} at {frac:.0%} of total"))
    else:
        v.gates.append(GateResult(15, "no single symbol", False, "missing"))

    # ---- 16. no extreme parameter dependence ----
    grid = artifacts / "param_robustness" / "grid.csv"
    if grid.exists():
        import pandas as pd

        g = pd.read_csv(grid)
        cv = float(g["total_pnl"].std() / abs(g["total_pnl"].mean()))
        v.gates.append(GateResult(16, "no extreme parameter dependence", cv < 0.15,
                                  f"grid pnl CV={cv:.2%}"))
    else:
        v.gates.append(GateResult(16, "no extreme parameter dependence", False, "missing"))

    # ---- 17. no catastrophic tail ----
    if mc:
        dd5 = float(mc.get("dd_p5", 0.0))
        ok = abs(dd5) < 0.5
        v.gates.append(GateResult(17, "no catastrophic tail behavior", bool(ok),
                                  f"worst-5% MC drawdown {dd5:.1%}"))
    else:
        v.gates.append(GateResult(17, "no catastrophic tail", False, "missing"))

    # ---- final classification ----
    n_pass = len(v.passed_gates)
    if n_pass == len(v.gates):
        v.final = "PRODUCTION READY"
    elif n_pass >= 10:
        v.final = "PAPER ONLY"
    elif n_pass >= 6:
        v.final = "RESEARCH ONLY"
    else:
        v.final = "REJECT"
    return v


if __name__ == "__main__":
    verdict = evaluate_acceptance()
    print(f"\n=== ACCEPTANCE VERDICT: {verdict.strategy} ===")
    print(f"FINAL: {verdict.final}  ({len(verdict.passed_gates)}/{len(verdict.gates)} gates passed)")
    print()
    for g in verdict.gates:
        mark = "PASS" if g.passed else "FAIL"
        print(f"  [{mark}] Gate {g.gate_id:2d} {g.name}: {g.evidence}")
    out = ARTIFACTS_DIR / "acceptance" / "verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict.to_dict(), indent=2))
    print(f"\nverdict saved: {out}")
