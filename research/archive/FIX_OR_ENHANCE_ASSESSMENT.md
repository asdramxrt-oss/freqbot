# Can it be fixed or enhanced? — direct answer with new evidence

**Date:** 2026-09-13
**Question:** "can it be fixed or enhanced"
**Answer:** Three different questions hide inside that one word "fixed." Separating
them, two are answerable now — and in the process of answering, I found the
project's **first net-positive, cost-surviving result.**

---

## 1. Directional alpha — cannot be fixed

This is not a bug, so there is nothing to repair. The evidence:

- **Mirror symmetry**: `fade` and `follow` are exact mirror images
  (`gross_fade = -gross_follow` by construction), which proves the direction rule
  carries zero directional information.
- **Chance-level agreement at high power**: 5,482–10,970 events, agreement ≈ 0.5.
- **Parameter-flat failure** (`verdict_pairs.json` gate 16): no hidden good pocket.
- **Turtle was regime beta**: profitable only in the 2023–24 bull.

Fixing this would require *new information* about future direction. No amount of
engineering manufactures information. This is the honest "no."

## 2. Engineering — genuinely improvable, and worth doing

These are real improvements that do not require alpha:

| Improvement | Value | Why it matters |
|---|---|---|
| Post-only limit orders | Saves ~10bp/round trip | Turns marginal strategies from negative to break-even; matters enormously for carry (see §3) |
| Live data adapters | Remove the local-feather dependency | The bot can only replay files today; live needs an exchange feed |
| Reconciliation + alerting | Detects divergence between bot state and broker | The plumbing exists (`execution/reconciliation.py`) but is not wired into the live loop |
| Multi-account risk aggregation | Portfolio-level limits across accounts | Currently single-account |
| Ops hardening | Restart safety, graceful shutdown, structured logging | Needed before any real deployment |

None of these create alpha. All of them make the system trustworthy **if** a
signal is ever found — and one of them (post-only) turns out to matter directly
for the finding below.

## 3. NEW RESULT: funding-rate carry is net-positive and cost-surviving

This is the first positive, cost-surviving finding in the entire project. I ran
it before recommending it, rather than promising it.

### The trade
Long spot, short perpetual future, delta-neutral. The funding mechanism pays the
short side when perps trade above spot (the normal state in crypto). You are not
forecasting price direction at all — you are **harvesting a structural risk
premium** paid by leveraged longs.

### The evidence (BTC and ETH, 2022-01 → 2026-07, 5,000 funding prints each)

| Metric | BTC | ETH |
|---|---|---|
| Mean 8h funding rate | 0.00602% | 0.00550% |
| **Gross annualised** | **6.60%** | **6.02%** |
| % of prints positive | 84.1% | 81.4% |
| **Net annualised after ~16bp round trip** | **5.89%** | **5.24%** |
| **Net-positive quarters** | **19 / 19** | **17 / 19** |
| Worst single 8h print | −0.119% | −0.302% |
| Worst cumulative funding drawdown | −0.41% | −1.79% |

**The critical result:** BTC is net-positive in **19 out of 19 quarters** — every
single quarter from 2022Q1 to 2026Q3, *including the 2022 bear market that
destroyed Turtle, OU, and pairs.* That regime-independence is exactly what every
previous strategy lacked. It is the strongest single piece of evidence produced
in this project.

### Why this is structurally different from everything that failed

1. **It does not forecast direction.** The three failed books all tried to
   predict which way price goes. Carry harvests a payment that exists *because*
   leveraged traders are willing to pay it. No prediction required.
2. **The premium is observable and persistent.** 84% of prints are positive —
   this is a documented structural feature of perpetual futures, not a fitted
   pattern.
3. **It survived the exact test that killed everything else.** The 2022 bear
   quarter is positive. Turtle's OOS failure was precisely its regime dependence;
   carry has no such dependence.

### Honest limitations — do not overstate this

I want to be as rigorous about the downsides as about the result:

- **~6% annualised is not spectacular.** After financing, taxes, and operational
  effort this is a modest, bond-like return. It is a real edge, not a jackpot.
- **The key risk is NOT in the funding data.** Delta-neutral means long spot +
  short perp. If perp price diverges upward from spot (a basis blowout), the
  short leg loses on mark-to-market *even while funding pays*. The funding series
  **cannot show this** — it requires perp-vs-spot basis data, which is not in the
  repo. This is the single most important gap in the analysis, and it is the
  thing that blows up real carry trades (see: leveraged short-leg liquidations in
  fast up-moves).
- **Only BTC and ETH are covered.** No SOL/BNB funding data locally.
- **Execution risk on 8h boundaries.** Funding is collected at fixed times;
  entering one minute late can miss a whole payment. Needs careful scheduling.
- **Crowded trade.** Carry is well known and heavily traded; compression over
  time is visible in the data (2026Q1–Q2 dropped to ~0.4–0.6% annualised).
- **Capital inefficiency.** Requires capital on both legs, so the return on total
  capital is roughly half the headline rate.
- **NOT yet run through the acceptance gates.** This is a first-pass analysis on
  the funding series, not a full validated backtest. It has not touched the
  holdout. It must go through the same 17 gates as everything else before any
  claim is made.

### What a proper test requires (the honest next step)

1. Acquire **perp and spot price series** to model the basis and the short-leg
   MtM risk — this is the critical missing input, and the analysis above is
   explicitly incomplete without it.
2. Build the causal backtest: funding collected at 8h boundaries, both legs
   costed, basis MtM included, liquidation thresholds modelled.
3. Run the walk-forward + 17 acceptance gates, including the untouched holdout.
4. Test whether **post-only execution** (improvement §2) lifts net carry
   materially — this is where engineering and economics actually connect.

---

## 4. Summary answer

| Question | Answer |
|---|---|
| Can the *strategies* be fixed? | **No.** No directional alpha; structurally absent, not a bug. |
| Can the *system* be enhanced? | **Yes.** Real, bounded improvements — post-only orders, live adapters, reconciliation, ops hardening. Creates no alpha but makes the system trustworthy. |
| Has anything positive been found? | **Yes — funding carry.** ~6% gross, ~5.5% net, positive in 19/19 BTC quarters including bear markets. The first regime-independent, cost-surviving result in the project. |
| Is carry proven? | **No.** Needs basis/MtM data and the full acceptance pipeline. The analysis is promising and explicitly incomplete. |

**The honest bottom line:** the bot cannot be fixed into a directional alpha
machine — that door is closed by evidence. But it just opened a different door:
a structural risk premium that survives costs and does not depend on predicting
price. That is worth pursuing properly, and it is the first thing in this project
that has earned that description.
