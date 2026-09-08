"""Combinatorial Purged Cross-Validation (Phase 16).

CPCV splits the ordered sample space into N groups, then for each combination of P test
groups (N-P train groups) fits a model and predicts the P test groups, with purging +
embargo (event-aware). A *path* is a set of combinations whose test groups are pairwise
disjoint and together cover all N groups exactly once - so a path is a full-coverage OOS
backtest where each group is predicted by a model that never saw it (nor anything
overlapping it).

The purpose is DISTRIBUTIONS across paths (Sharpe/Sortino/profit factor/etc.), never a
single point estimate; a model is never selected because its best CPCV path is excellent.

For N=6, P=2 there are C(6,2)=15 combinations and 15 perfect-matchings (paths) of 3
disjoint pairs. The user-facing split() returns all combinations with their group tags;
paths() assembles full-coverage paths from the per-combination OOS trade frames.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from pandas import DataFrame


class CPCV:
    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo_bars: int = 24):
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_bars = embargo_bars

    def _event_bounds(self, events: DataFrame) -> tuple[np.ndarray, np.ndarray]:
        t0 = events["t0"].to_numpy(dtype=int)
        t1 = events["t1"].to_numpy(dtype=int) if "t1" in events.columns else t0.copy()
        return t0, t1

    def combinations(self) -> list[tuple[int, ...]]:
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def split(self, events: DataFrame) -> list[dict]:
        """Return one dict per combination:
        {combo: (g1,g2), train_idx, test_idx, test_groups} with purging + embargo.
        """
        n = len(events)
        t0, t1 = self._event_bounds(events)
        order = np.argsort(t0, kind="stable")
        edges = np.linspace(0, n, self.n_groups + 1).astype(int)
        groups = [order[edges[i] : edges[i + 1]] for i in range(self.n_groups)]
        t0_set_all = {int(x) for x in t0}
        splits: list[dict] = []
        for combo in self.combinations():
            test_pos = np.concatenate([groups[g] for g in combo])
            # purge against EACH test group's own embargoed interval (not the union, which
            # for non-contiguous groups would wrongly purge the samples between them)
            embargoed: list[tuple[int, int]] = []
            for g in combo:
                gs = int(t0[groups[g]].min())
                ge = int(t1[groups[g]].max()) + self.embargo_bars
                embargoed.append((gs, ge))
            train_pos = []
            for i in order:
                s0, s1 = int(t0[i]), int(t1[i])
                # a train sample is purged only if it overlaps ANY test group's interval
                overlaps = any(not (s1 < lo or s0 > hi) for lo, hi in embargoed)
                if not overlaps:
                    train_pos.append(i)
            if len(train_pos) < 10:
                continue
            splits.append(
                {
                    "combo": combo,
                    "train_idx": np.asarray(train_pos),
                    "test_idx": test_pos,
                    "test_groups": set(combo),
                }
            )
        if not splits:
            raise ValueError("CPCV produced no valid splits - reduce test groups / embargo")
        return splits

    def paths(self) -> list[list[dict]]:
        """Enumerate full-coverage paths: combinations whose test groups are disjoint and
        cover all N groups. Returns list of paths, each a list of split dicts (must pair
        with split() output in order by combo)."""
        combos = self.combinations()
        # depth-first search for perfect matchings over the N groups
        full = set(range(self.n_groups))
        paths: list[list[tuple[int, ...]]] = []

        def dfs(covered: set, chosen: list[tuple[int, ...]]) -> None:
            if len(chosen) * self.n_test_groups == self.n_groups:
                if covered == full:
                    paths.append(list(chosen))
                return
            for c in combos:
                if set(c) & covered:
                    continue
                dfs(covered | set(c), chosen + [c])

        dfs(set(), [])
        # paths may be duplicated due to combination ordering - dedupe by sorted combo set
        seen: set = set()
        unique: list[list[tuple[int, ...]]] = []
        for p in paths:
            key = tuple(sorted(p))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        # map back to split dicts
        return unique

    def build_paths(self, results: list[tuple[tuple[int, ...], DataFrame]]) -> DataFrame:
        """Combine per-combination OOS trade frames into full-coverage paths.

        results: list of (combo, trades_df) in the same order as combinations().
        Returns a DataFrame indexed by path with columns path_n, oos_pnl_series collapsed:
        each row is one path; column 'trades' holds the concatenated OOS trades.
        """
        combo_map = {c: df for c, df in results}
        rows = []
        for i, path in enumerate(self.paths()):
            frames = [combo_map[c] for c in path if c in combo_map]
            if len(frames) != len(path):
                continue
            concat = pd.concat(frames, ignore_index=True).sort_values("entry_time")
            rows.append({"path": f"path_{i}", "trades": concat})
        return DataFrame(rows)
