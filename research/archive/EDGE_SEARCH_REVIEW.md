# Can this become a profitable trading bot? — an evidence review

**Date:** 2026-09-13
**Question asked:** "try every possible ways to make it successful trading bot"
**Answer:** The search has already been run, across every direction the existing
blueprint proposes — and none of them work. This document shows the evidence
rather than asserting the conclusion, and states precisely what is and is not
worth doing next.

---

## 1. Why "try everything" is the wrong objective

Tuning a strategy until the backtest looks good is not a path to a profitable
bot; it is the standard mechanism by which people *believe* they have one. With
enough parameter search, any strategy can be made to look profitable on
historical data, because you are fitting the noise. The failure arrives when the
fitting stops and the market starts.

This project has already been through that trap. `research/reports/OOS_REPORT.md`
records a strategy that was **robustly positive in the research window and lost
money on the untouched holdout**:

```
Holdout metrics (pooled BTC+ETH, $100k):
* trades 76, win 22.4%
* net PnL $-4,857 (-4.9% on equity)
* Sharpe -0.06, PF 0.91
* maxDD -28.0%
* MC: p(pnl<0) 0.83
```

The correct objective is therefore not "find a configuration that backtests
well." It is: **search for a genuine statistical edge, and be willing to conclude
there isn't one.** That search has been done.

---

## 2. The prior blueprint, and whether its proposals were tested

`research/bot_success_blueprint.md` (2026-09-08) argued the project was
"halfway complete" — good engineering, missing alpha — and proposed four
directions. Each has since been implemented and evaluated. Results:

| Blueprint proposal | Status | Evidence |
|---|---|---|
| **Market-neutral pairs trading** (§1.1) — "spreads between cointegrated assets do [mean-revert]" | **Tested. Fails.** | `verdict_pairs.json`: `gross=$-127,077 < 0 — fails even at zero cost`; `0/7 windows positive`; holdout `sharpe -2.13, pf 0.39` |
| **Higher frequency / more trades** (§1.2) — "more data for ML" | **Tested. Fails.** | The three books already run on 15m. `ALPHA_DISCOVERY_REPORT` §3.2: the "more trades" hypothesis was directly falsified — at 5,482 and 10,970 events, directional agreement still hugs 0.5 |
| **Maker/limit-order fee savings** (§3.1) — "can turn break-even into profitable" | **Tested. Insufficient.** | `verdict_pairs.json` gate 7: `maker savings ~$3,772 cannot rescue` a −$127k gross |
| **Repurpose ML to predict volatility, not direction** (§2.1) | **Already implemented.** | This *is* the current design. The vol-spike model (OOS AUC 0.82–0.90) is the one validated capability, and it is used as a kill switch |
| **Trade the vol spike itself as the signal** (the follow-on brief) | **Tested across 4 families. Fails.** | `ALPHA_DISCOVERY_REPORT`: H1 momentum-ignition, H2 vol-crush fade, H3 dip-buy, H4 vol-gated breakout — all fail; no config reached 5/6 positive-gross windows |

The blueprint's diagnosis was right — the problem is alpha, not engineering —
but its prescriptions were already tried, and the evidence contradicts them.

---

## 3. The two results that close the question

These are not "the strategy underperformed." They are structural findings that
rule out the whole approach, not a parameter choice.

### 3.1 Direction rules are exact mirror images

`ALPHA_DISCOVERY_REPORT` §3.1: the `fade` and `follow` strategies are
**mathematically mirror-symmetric** — identical trade sets with flipped sides,
so `gross_fade = -gross_follow` by construction. This proves the direction rule
only ever picks a *side*, never *which* spikes carry information. If pre-spike
momentum had directional content, the two could not be exact opposites. A
profitable result from either would necessarily be window luck.

### 3.2 Directional agreement is at chance, with real statistical power

With 5,482–10,970 events, the test has ample power to detect a weak edge.
Measured agreement between the forward move's sign and both trend and momentum
hugged 0.5 and flipped sign window-to-window (27–67%). This is the signature of
**no directional signal** — and because it was measured at high event counts, it
closes the "we just need more trades" escape hatch.

The residual apparent edge is **leveraged regime beta**: spike-conditional
returns simply track the window's own drift. Trading that directionally is a
levered bet on the prevailing regime — the same thing that killed Turtle.

### 3.3 No instrument to monetise the one real capability

The vol-spike model genuinely predicts *magnitude*. But crypto spot/futures, as
configured here, offers no way to monetise predicted magnitude without a
directional view — no options, straddles, or variance swaps. That is a
structural property of the signal and the instrument set, not a parameter miss.

---

## 4. What is genuinely unexplored

Honesty requires separating "tested and failed" from "not tested." Remaining
gaps, and a candid assessment of each:

1. **Higher-frequency data (1m / tick).** Not available locally; only 15m/1h/4h
   exist. This is untested. However, §3.2 weakens the case for it: the
   no-directional-signal finding was established *with statistical power*, not
   for lack of data. More granularity increases the number of noisier
   observations; it does not add directional information that isn't there. It
   also sharply increases fee drag, which is already the binding constraint.
2. **Funding-rate / basis carry.** `research/data/funding/` holds BTC and ETH
   funding. Funding-carry is a *different* return source from directional
   prediction (it harvests a risk premium rather than forecasting price). This
   is the most defensible untested idea in the repo, and it is a genuine
   research question, not a re-tune.
3. **Alternative data sources.** Order-book microstructure, liquidation data,
   on-chain flows. Genuinely unexplored, but requires data acquisition and a new
   research programme — not a change to this codebase.
4. **A different asset class.** The tools are general; a market with more
   exploitable structure (e.g. certain equity/commodity microstructure effects)
   is a legitimate pivot. Again: a new project, not a config change.

Note what is *not* on this list: any "try everything" re-tune of the existing
books. That search space has been covered and the failure was parameter-flat
(`verdict_pairs.json` gate 16: "failure is parameter-flat — no hidden good
pocket").

---

## 5. What I can legitimately do, if you want the search genuinely extended

In rough order of expected value:

1. **Research funding-rate carry** (untested, and a structurally different
   return source). I can build the causal backtest and run it through the
   existing walk-forward and acceptance gates. This is the single most
   promising remaining option, and it does not require new data.
2. **Acquire higher-frequency data and re-run the vol-spike model** to see
   whether magnitude prediction improves — with the explicit expectation that it
   still cannot be monetised directionally.
3. **Harden what exists** — the risk overlay, execution plumbing, and monitoring
   — so the system is trustworthy *if* a real signal is ever found. This is real
   value: it is the part of the project that already works.

What I will not do is run a large parameter sweep until something screens
profitable and then present it as a success. That would be manufacturing exactly
the artifact that the holdout test exists to destroy.

---

## 6. Bottom line

- The project has an **excellent engineering foundation**: causal guarantees,
  leak-free validation, a real risk engine, and a holdout that correctly
  rejected an overfit strategy. That is a genuine achievement, and it is why the
  negative results can be trusted.
- It does **not** have directional alpha, and the search for it has been run
  with statistical power across momentum, mean reversion, pairs, and
  volatility-conditioned entries — at 15m, on four assets, over multiple market
  regimes.
- The one validated capability (volatility prediction) is correctly deployed as
  risk management, and cannot be monetised directionally on these instruments.
- **The honest answer to "can this be made profitable by trying everything" is
  no** — not because the effort is insufficient, but because the evidence says
  the edge isn't there. Further tuning would produce a profitable-looking
  backtest and a losing live account.

The most valuable next step is funding-rate carry: a different return source,
requiring no new data. Say the word and I'll build it and put it through the
same gates — including the holdout, and including the possibility that it fails.
