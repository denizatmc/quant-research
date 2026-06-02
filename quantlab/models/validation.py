"""Time-series-aware cross-validation.

Standard k-fold CV is actively misleading on financial data: shuffling rows lets the model
train on the future and test on the past, and overlapping label windows leak information
across the train/test boundary. Both inflate scores and produce strategies that die in
production.

This module provides the two corrections I actually use:

  * `walk_forward_splits` — expanding (or rolling) train window, always testing on a later
    block. This is just "honest" time-series CV.
  * a `purge`/`embargo` gap between train and test, à la López de Prado, so that a label
    computed over the next N days can't overlap the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n: int,
    n_splits: int = 5,
    test_size: int | None = None,
    purge: int = 0,
    expanding: bool = True,
) -> Iterator[WalkForwardSplit]:
    """Yield (train, test) index arrays walking forward through `n` ordered observations.

    `purge` drops the last `purge` observations of the train window so a forward-looking
    label (e.g. next-21-day return) can't bleed into the test block. `expanding=True` grows
    the train window each fold (use all history so far); `False` uses a fixed-length
    rolling window, which is better when the data-generating process drifts.
    """
    test_size = test_size or n // (n_splits + 1)
    if test_size <= 0:
        raise ValueError("Not enough observations for the requested number of splits.")

    indices = np.arange(n)
    first_test_start = n - n_splits * test_size
    for k in range(n_splits):
        test_start = first_test_start + k * test_size
        test_end = test_start + test_size
        train_end = test_start - purge          # leave a gap before the test block
        if train_end <= 0:
            continue
        train_start = 0 if expanding else max(0, train_end - (first_test_start))
        yield WalkForwardSplit(
            train_idx=indices[train_start:train_end],
            test_idx=indices[test_start:test_end],
        )


def information_coefficient(predictions: np.ndarray, realised: np.ndarray) -> float:
    """Rank (Spearman) correlation between forecast and outcome — the quant's R².

    For alpha signals the *ranking* is what gets traded (long the top, short the bottom),
    so rank correlation is a more faithful score than MSE. An IC of 0.03-0.05 sustained
    out-of-sample is already a respectable equity signal; anything much higher on this kind
    of data usually means a leak.
    """
    from scipy.stats import spearmanr

    mask = np.isfinite(predictions) & np.isfinite(realised)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(predictions[mask], realised[mask])
    return float(rho)
