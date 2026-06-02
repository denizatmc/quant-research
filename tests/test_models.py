"""Tests for the statistical machinery: stationarity detection, mean-reversion maths, and
— crucially — that the walk-forward splitter never lets training data touch the future."""

import numpy as np
import pandas as pd
import pytest

from quantlab.models.timeseries import adf_test, half_life
from quantlab.models.validation import walk_forward_splits, information_coefficient


def test_adf_distinguishes_random_walk_from_stationary():
    rng = np.random.default_rng(1)
    # A random walk has a unit root -> should NOT test as stationary.
    rw = pd.Series(np.cumsum(rng.normal(size=500)))
    # White noise is stationary -> should.
    noise = pd.Series(rng.normal(size=500))
    assert adf_test(rw).is_stationary is False
    assert adf_test(noise).is_stationary is True


def test_half_life_of_known_ar1():
    # Build an AR(1) with known phi; half-life should be close to ln(2)/(-ln(phi)).
    rng = np.random.default_rng(2)
    phi = 0.9
    x = np.zeros(5000)
    for t in range(1, len(x)):
        x[t] = phi * x[t - 1] + rng.normal(scale=0.1)
    hl = half_life(pd.Series(x))
    expected = np.log(2) / (-np.log(phi))
    assert hl == pytest.approx(expected, rel=0.25)


def test_walk_forward_has_no_leakage():
    # Every training index must come strictly before its test indices, with the purge gap.
    purge = 5
    for split in walk_forward_splits(200, n_splits=4, purge=purge):
        assert split.train_idx.max() < split.test_idx.min()
        # The purge gap must actually be there.
        assert split.test_idx.min() - split.train_idx.max() >= purge


def test_kalman_recovers_constant_hedge_ratio():
    # If y = 3*x + noise with a fixed ratio, the filter's beta should converge near 3.
    from quantlab.models.kalman import KalmanHedgeRatio

    rng = np.random.default_rng(3)
    x = pd.Series(np.cumsum(rng.normal(size=800)) + 100)
    y = 3.0 * x + rng.normal(scale=0.5, size=len(x))
    out = KalmanHedgeRatio(delta=1e-4).run(x, y)
    assert out["beta"].iloc[-1] == pytest.approx(3.0, abs=0.1)


def test_information_coefficient_sign():
    # Perfectly ranked predictions -> IC of 1; reversed -> -1.
    y = np.arange(50, dtype=float)
    assert information_coefficient(y, y) == pytest.approx(1.0)
    assert information_coefficient(y, y[::-1]) == pytest.approx(-1.0)
