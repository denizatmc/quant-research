"""Value-at-Risk, Expected Shortfall, and drawdown.

I implement VaR three ways on purpose, because the disagreement between them is the
interesting part:

  * Historical    — empirical quantile of past returns. No distributional assumption, but
                    it can only ever know about losses it has already seen.
  * Parametric    — Gaussian (variance-covariance). Clean and fast, but normal tails are
                    too thin, so it *understates* tail risk exactly when it matters.
  * Monte Carlo   — simulate from a fitted distribution. Here I draw from a Student-t, whose
                    fat tails are a better match for real return data than a Gaussian.

Whatever the method, I prefer Expected Shortfall (a.k.a. CVaR — the average loss *given* you
breach the VaR) as the headline number. VaR tells you a threshold; ES tells you how bad it
is on the other side of it, and unlike VaR it's a coherent risk measure (it's sub-additive,
so diversification can't make it look worse than the sum of parts).

Sign convention: these return *positive* numbers representing the magnitude of a loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series, confidence: float = 0.99) -> float:
    """Empirical VaR: the loss at the (1 - confidence) quantile of the return distribution."""
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    q = np.quantile(returns, 1.0 - confidence)
    return float(-q)


def parametric_var(returns: pd.Series, confidence: float = 0.99) -> float:
    """Gaussian VaR from the sample mean and std. Fast, but thin-tailed by construction —
    useful mainly as the baseline the other methods get compared against."""
    returns = returns.dropna()
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1.0 - confidence)
    return float(-(mu + z * sigma))


def monte_carlo_var(
    returns: pd.Series, confidence: float = 0.99, n_sims: int = 100_000, seed: int = 42
) -> float:
    """VaR by simulating from a Student-t fit to the returns.

    Fitting the degrees of freedom lets the data tell us how fat the tails are; a low df
    means heavy tails and a materially larger VaR than the Gaussian would give.
    """
    returns = returns.dropna()
    if len(returns) < 30:
        return float("nan")
    df, loc, scale = stats.t.fit(returns)
    rng = np.random.default_rng(seed)
    sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims, random_state=rng)
    return float(-np.quantile(sims, 1.0 - confidence))


def expected_shortfall(returns: pd.Series, confidence: float = 0.99) -> float:
    """Expected Shortfall / CVaR: the mean loss in the tail beyond the historical VaR."""
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if not tail.empty else float("nan")


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Running peak-to-trough drawdown as a (non-positive) series — useful for plotting the
    underwater curve and for spotting how *long*, not just how deep, the drawdowns ran."""
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0
