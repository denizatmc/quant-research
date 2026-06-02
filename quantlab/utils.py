"""Small shared helpers — the functions that would otherwise get copy-pasted everywhere.

Two principles I try to hold to across the codebase and that show up here:
  1. Be explicit about simple vs log returns. They're nearly equal for small moves and
     subtly different for large ones; conflating them is a real source of bugs.
  2. Annualisation always goes through one place, so the trading-days convention can't
     drift between modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Arithmetic returns. These are what aggregate correctly *across* assets (a portfolio
    return is the weighted sum of simple returns), so this is the default for backtests."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns. These aggregate correctly *over time* and are roughly Gaussian, so
    they're what I use for volatility modelling and most statistical tests."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def annualize_return(daily_returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric annualisation of a daily return series."""
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return float("nan")
    growth = (1.0 + daily_returns).prod()
    years = len(daily_returns) / periods
    return float(growth ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def annualize_vol(daily_returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised volatility under the usual sqrt-time scaling (i.e. assuming roughly iid
    daily returns — fine as a summary stat, even though we know vol clusters)."""
    return float(daily_returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe. `rf` is an annual risk-free rate, converted to per-period here."""
    daily_returns = daily_returns.dropna()
    if daily_returns.std(ddof=1) == 0 or daily_returns.empty:
        return float("nan")
    excess = daily_returns - rf / periods
    return float(np.sqrt(periods) * excess.mean() / excess.std(ddof=1))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline of a cumulative equity curve (a negative number)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def to_equity_curve(daily_returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound a return series into an equity curve."""
    return initial * (1.0 + daily_returns.fillna(0.0)).cumprod()
