"""Walk-forward engine (Phase 17).

Discipline:
  TRAIN -> VALIDATE -> TEST windows that roll forward. Parameters are optimized only inside
  TRAIN; VALIDATION is used for model/parameter selection; TEST is honest evaluation. Later
  windows never influence earlier ones. A final untouched holdout is enforced by the caller
  (the data loader already separates research/data from research/data_holdout).

This engine is strategy-agnostic: the caller supplies a fit/select/predict closure that is
re-invoked per window with only that window's data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass
class WFConfig:
    train_bars: int = 4000
    val_bars: int = 1000
    test_bars: int = 1000
    step_bars: int = 1000  # roll forward by this much (train grows if expand=True)
    expand_train: bool = True


@dataclass
class WFFold:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    test_trades: DataFrame = field(default_factory=DataFrame)
    test_metrics: dict = field(default_factory=dict)


class WalkForward:
    """Rolling TRAIN/VALIDATE/TEST evaluation.

    fit_predict: Callable[[DataFrame, np.ndarray, np.ndarray, np.ndarray], tuple[dict, DataFrame]]
        - receives (frame, train_idx, val_idx, test_idx) where frame is the FULL causal
          feature/market frame.
        - returns (params_dict, test_trades_df). The closure is responsible for fitting ONLY
          on train_idx data and emitting trades only over test_idx (using strictly causal
          signals). The engine slices the frame so the closure CANNOT see test rows during
          fit unless it deliberately misuses the indices.
    """

    def __init__(self, config: WFConfig | None = None):
        self.cfg = config or WFConfig()

    def _make_slices(self, n: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        c = self.cfg
        total = c.train_bars + c.val_bars + c.test_bars
        slices = []
        start = 0
        fid = 0
        while start + total <= n:
            tr = np.arange(start, start + c.train_bars)
            va = np.arange(start + c.train_bars, start + c.train_bars + c.val_bars)
            te = np.arange(
                start + c.train_bars + c.val_bars,
                start + c.train_bars + c.val_bars + c.test_bars,
            )
            slices.append((fid, tr, va, te))
            start += c.step_bars
            fid += 1
        return slices

    def run(
        self,
        frame: DataFrame,
        fit_predict: Callable[[DataFrame, np.ndarray, np.ndarray, np.ndarray], tuple[dict, DataFrame]],
    ) -> list[WFFold]:
        n = len(frame)
        folds: list[WFFold] = []
        for fid, tr, va, te in self._make_slices(n):
            params, trades = fit_predict(frame, tr, va, te)
            fold = WFFold(
                fold_id=fid,
                train_idx=tr,
                val_idx=va,
                test_idx=te,
                params=params,
                test_trades=trades if trades is not None else DataFrame(),
            )
            folds.append(fold)
        return folds
