"""Classical time-series tools: stationarity, cointegration, mean-reversion speed, GARCH.

This module is the statistical backbone behind the stat-arb strategy and the volatility
work. I lean on statsmodels and arch rather than re-implementing the estimators — the value
here is in framing the tests correctly and reading them honestly, not in rewriting an ADF
routine that's been battle-tested for thirty years.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


@dataclass
class StationarityResult:
    statistic: float
    pvalue: float
    used_lag: int
    n_obs: int
    critical_values: dict[str, float]

    @property
    def is_stationary(self) -> bool:
        # Convention: reject the unit-root null at 5%. Cast to a real Python bool — statsmodels
        # hands back a numpy float, and a numpy bool surprises callers doing `is False` checks.
        return bool(self.pvalue < 0.05)


def adf_test(series: pd.Series, regression: str = "c") -> StationarityResult:
    """Augmented Dickey-Fuller test for a unit root.

    Null hypothesis: the series has a unit root (is non-stationary). A small p-value lets
    us reject that and treat the series as stationary — which, applied to a spread, is the
    whole basis for mean-reversion trading.
    """
    series = series.dropna()
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(series, regression=regression, autolag="AIC")
    return StationarityResult(stat, pvalue, used_lag, nobs, crit)


@dataclass
class CointegrationResult:
    beta: float                  # hedge ratio (y ~ beta * x)
    alpha: float                 # intercept
    adf: StationarityResult      # ADF on the residual spread
    spread: pd.Series

    @property
    def is_cointegrated(self) -> bool:
        return self.adf.is_stationary


def engle_granger_cointegration(y: pd.Series, x: pd.Series) -> CointegrationResult:
    """Engle-Granger two-step cointegration test.

    Step 1: regress y on x (with intercept) to get the cointegrating vector.
    Step 2: ADF-test the residuals. If the residual spread is stationary, the two I(1)
    series are cointegrated and the spread is tradeable.

    Caveat I keep front of mind: Engle-Granger is asymmetric (regressing x on y can give a
    slightly different verdict) and assumes a single cointegrating relationship. For two
    legs that's fine; for a basket I'd reach for the Johansen test.
    """
    df = pd.concat([y, x], axis=1).dropna()
    yy, xx = df.iloc[:, 0], df.iloc[:, 1]
    X = sm.add_constant(xx)
    model = sm.OLS(yy, X).fit()
    alpha, beta = model.params.iloc[0], model.params.iloc[1]
    spread = yy - (alpha + beta * xx)
    # ADF without a constant: the spread is already de-meaned by the regression intercept.
    adf = adf_test(spread, regression="n")
    return CointegrationResult(beta=float(beta), alpha=float(alpha), adf=adf, spread=spread)


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life via an Ornstein-Uhlenbeck / AR(1) fit.

    Regress Δspread on the lagged level: Δs_t = λ s_{t-1} + c. The mean-reversion speed is
    -λ, and the half-life (how long to close half the gap to the mean) is ln(2)/(-λ). It's
    the single number I most want when sizing a pairs trade — it tells you how long you'll
    be holding, and therefore whether the costs are worth it.
    """
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    delta = (spread - spread.shift(1)).dropna()
    common = lagged.index.intersection(delta.index)
    X = sm.add_constant(lagged.loc[common])
    lam = sm.OLS(delta.loc[common], X).fit().params.iloc[1]
    if lam >= 0:
        return float("inf")  # not mean-reverting on this sample
    return float(-np.log(2) / lam)


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t"):
    """Fit a GARCH(p, q) to a return series and return the fitted arch result.

    Volatility clusters — calm begets calm, turmoil begets turmoil — and a constant-vol
    assumption misses that entirely. GARCH(1,1) captures the clustering with two
    parameters, and a Student-t innovation distribution handles the fat tails that a
    Gaussian would understate (which is exactly where risk lives). Returns are scaled to
    percent because arch's optimiser is much happier away from tiny numbers.
    """
    from arch import arch_model  # local import keeps module import cheap

    r = returns.dropna() * 100.0
    model = arch_model(r, vol="GARCH", p=p, q=q, dist=dist, mean="Constant")
    return model.fit(disp="off")
