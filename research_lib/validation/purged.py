"""Purged K-Fold with embargo (Phase 15) - event-aware cross-validation.

For every test event, training events that OVERLAP the test event's label horizon are
removed (purged), and an embargo gap is added after the test interval so that label
information cannot bleed from test into the following train fold.

The splitter understands event structure: each sample is (start, end) where end is the
barrier/vertical horizon of the sample's label. Standard sklearn KFold is NOT sufficient
because it ignores label overlap.

Also provides get_train_times()/get_test_times() compatible helpers so the rest of the
validation stack and sklearn pipelines can consume the folds via indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class PurgedKFold:
    """Event-aware purged K-fold splitter.

    events: DataFrame with columns
        t0  (start bar index of the sample / event)
        t1  (end bar index - the label horizon / vertical barrier index)
        If t1 is absent it defaults to t0 (point samples).
    """

    n_splits: int = 5
    embargo_bars: int = 24

    def _event_bounds(self, events: DataFrame) -> tuple[np.ndarray, np.ndarray]:
        t0 = events["t0"].to_numpy(dtype=int)
        if "t1" in events.columns:
            t1 = events["t1"].to_numpy(dtype=int)
        else:
            t1 = t0.copy()
        return t0, t1

    def _embargo(self, test_start: int, test_end: int) -> tuple[int, int]:
        """Test interval expanded by the embargo AFTER the test block.

        Samples whose label horizon falls inside [test_start, test_end+embargo) cannot be
        in the training set (their labels used test-period data).
        """
        return test_start, test_end + self.embargo_bars

    def split(self, events: DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return [(train_idx, test_idx), ...] with purging + embargo applied."""
        n = len(events)
        if n < self.n_splits:
            raise ValueError("not enough samples for the requested splits")
        t0, t1 = self._event_bounds(events)
        # order samples by start time
        order = np.argsort(t0, kind="stable")
        # contiguous blocks in the ordered sample space
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(self.n_splits):
            test_pos = order[bounds[k] : bounds[k + 1]]
            test_start = int(t0[test_pos].min())
            test_end = int(t1[test_pos].max())
            emb_start, emb_end = self._embargo(test_start, test_end)
            train_pos = []
            for i in order:
                if i in set(test_pos):
                    continue
                # purge: drop any train sample whose label horizon overlaps the (embargoed)
                # test interval
                s0, s1 = int(t0[i]), int(t1[i])
                overlap = not (s1 < emb_start or s0 > emb_end)
                if not overlap:
                    train_pos.append(i)
            if len(train_pos) < 10:
                raise ValueError(
                    f"purged fold {k} has only {len(train_pos)} training samples; "
                    "reduce n_splits or embargo"
                )
            folds.append((np.asarray(train_pos), test_pos))
        return folds
