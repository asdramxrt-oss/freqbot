# Research output index

Evidence-backed trading-system research built on this Freqtrade checkout.

## Verdict
**RESEARCH ONLY** (9/17 acceptance gates). No component demonstrated a persistent,
regime-independent out-of-sample edge on the available data. See
`reports/PRODUCTION_READINESS.md` and `reports/FINAL_SYSTEM_SPEC.md`.

## Layout
- `research_lib/` - the research + execution stack (104 unit tests).
- `research/artifacts/` - every raw result (CSV/JSON) produced by the pipeline.
- `research/reports/` - the 18 mandated deliverable documents.
- `research/data/` - research-window market data (gitignored, regenerable).
- `research/data_holdout/` - final UNTOUCHED holdout (gitignored).

## Reproduce
```
.venv\Scripts\python -m pytest research_lib/tests -c research_lib/pytest.ini   # 104 tests
.venv\Scripts\python research\run_research.py     # stages 1-5 (baseline/costs/MC/params)
.venv\Scripts\python research\run_research2.py    # stages 6-8 (regime/OU/meta-label)
.venv\Scripts\python research\run_research3.py    # stages 9-10 (walk-forward/look-ahead)
.venv\Scripts\python research\run_holdout.py      # final untouched holdout (run ONCE)
.venv\Scripts\python research\generate_reports.py # write all 18 reports
.venv\Scripts\python research_lib\validation\acceptance.py  # 17-gate verdict
```

## Key honest results
- Turtle 20-bar long-only 1x: in-sample pooled PF 1.58 / Sharpe 1.10, robust to costs and
  parameter perturbation, causality-clean. Final holdout: Sharpe -0.06, PF 0.91, 83% MC
  probability of loss. The in-sample edge was a 2023-24 bull-market artifact.
- OU mean reversion: correctly rejected on BTC log-price (no stationary MR).
- Meta-model (RF/XGB): AUC < 0.5 OOS; removed.
- HMM regime router: in-window attribution favorable but sample sizes too small; not adopted.
- Phase log: `research/PHASE_LOG.md`. Audit: `research/ARCHITECTURE_AUDIT.md`.
