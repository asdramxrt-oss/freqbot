# Research output index

Evidence-backed trading-system research built on this Freqtrade checkout.

## Verdict
**RESEARCH ONLY.** No component demonstrated a persistent, regime-independent
out-of-sample edge on the available data:
- Turtle 20-bar: 9/17 gates (positive in-sample economics, fails OOS - regime dependent).
- Pairs/spread pivot (blueprint): 8/17 gates (all economic gates fail; negative gross).
- The one validated OOS capability is **volatility-spike prediction** (AUC ~0.82-0.90),
  useful as a risk/filter input, not an alpha source.
See `reports/PRODUCTION_READINESS.md`, `BLUEPRINT_REPORT.md`, `IMPLEMENTATION_REPORT.md`.

## Layout
- `research_lib/` - the research + execution stack (122 unit tests).
- `research/artifacts/` - every raw result (CSV/JSON) produced by the pipeline.
- `research/reports/` - the 18 mandated deliverable documents.
- `research/data/` - research-window market data (gitignored, regenerable).
- `research/data_holdout/` - final UNTOUCHED holdout (gitignored).

## Reproduce
```
.venv\Scripts\python -m pytest research_lib/tests -c research_lib/pytest.ini   # 122 tests
.venv\Scripts\python research\run_research.py     # stages 1-5 (baseline/costs/MC/params)
.venv\Scripts\python research\run_research2.py    # stages 6-8 (regime/OU/meta-label)
.venv\Scripts\python research\run_research3.py    # stages 9-10 (walk-forward/look-ahead)
.venv\Scripts\python research\run_holdout.py      # final untouched holdout (run ONCE)
.venv\Scripts\python research\run_pairs.py        # blueprint: pairs baseline (15m)
.venv\Scripts\python research\run_volspike.py     # blueprint: vol-spike filter research
.venv\Scripts\python research\run_pairs2.py       # blueprint: maker econ + WFO + holdout
.venv\Scripts\python research\generate_reports.py # write all 18 reports
.venv\Scripts\python research_lib\validation\acceptance.py  # 17-gate verdicts (both strategies)
```

## Key honest results
- Turtle 20-bar long-only 1x: in-sample pooled PF 1.58 / Sharpe 1.10, robust to costs and
  parameter perturbation, causality-clean. Final holdout: Sharpe -0.06, PF 0.91, 83% MC
  probability of loss. The in-sample edge was a 2023-24 bull-market artifact.
- Pairs/spread pivot (ETH/BTC, SOL/BTC, BNB/BTC, SOL/ETH on 15m): negative gross in the
  research window, every semi-annual window negative, holdout -$25.7k. Maker savings
  (~$3.8k) cannot rescue negative gross. No market-neutral reversion edge found.
- OU mean reversion: correctly rejected on BTC log-price (no stationary MR).
- Meta-model (RF/XGB): AUC < 0.5 OOS on 79 Turtle events AND on 318 pairs events; removed.
- Vol-spike prediction: VALIDATED OOS capability (AUC 0.82-0.90 on BTC/ETH/SOL 15m); as a
  filter it reduces pairs losses ~63% but does not create edge; on Turtle it reduces profit.
- Phase log: `research/PHASE_LOG.md`. Audit: `research/ARCHITECTURE_AUDIT.md`.
  Blueprint work: `research/BLUEPRINT_REPORT.md`.
