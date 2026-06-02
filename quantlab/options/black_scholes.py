"""Black-Scholes-Merton pricing, Greeks, and an implied-vol solver.

Everything here is closed-form (or a 1-D root-find for IV), vectorised over NumPy so it
prices a whole option chain at once. I derive the Greeks analytically rather than bumping
the price numerically — it's exact, faster, and frankly the derivations are half the reason
to include this module. The continuous-dividend-yield `q` keeps it general (set q=0 for a
non-dividend payer, q=r for a future).

Sign/convention notes I keep explicit because they bite people:
  * Theta is returned *per calendar day* (the raw model gives per-year), because that's how
    a trader reads it.
  * Vega and rho are per 1.00 change in vol / rate (i.e. per 100 vol points / 100 bps);
    divide by 100 for the "per 1%" convention if you prefer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholes:
    """Price and risk a European option under Black-Scholes-Merton.

    Parameters use the standard names: S spot, K strike, T time-to-expiry in years,
    r risk-free rate, sigma volatility, q continuous dividend yield.
    """

    @staticmethod
    def _d1_d2(S, K, T, r, sigma, q):
        # Guard the degenerate T->0 / sigma->0 corner so we don't divide by zero at expiry.
        sqrtT = np.sqrt(np.maximum(T, 1e-12))
        vol = np.maximum(sigma, 1e-12)
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqrtT)
        d2 = d1 - vol * sqrtT
        return d1, d2

    @classmethod
    def price(cls, S, K, T, r, sigma, q=0.0, option_type: OptionType = OptionType.CALL):
        d1, d2 = cls._d1_d2(S, K, T, r, sigma, q)
        disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
        if option_type == OptionType.CALL:
            return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)

    @classmethod
    def greeks(cls, S, K, T, r, sigma, q=0.0, option_type: OptionType = OptionType.CALL) -> Greeks:
        d1, d2 = cls._d1_d2(S, K, T, r, sigma, q)
        disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
        pdf_d1 = norm.pdf(d1)
        sqrtT = np.sqrt(np.maximum(T, 1e-12))

        # Gamma and vega are the same for calls and puts.
        gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
        vega = S * disc_q * pdf_d1 * sqrtT

        if option_type == OptionType.CALL:
            delta = disc_q * norm.cdf(d1)
            theta_yr = (
                -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                - r * K * disc_r * norm.cdf(d2)
                + q * S * disc_q * norm.cdf(d1)
            )
            rho = K * T * disc_r * norm.cdf(d2)
        else:
            delta = -disc_q * norm.cdf(-d1)
            theta_yr = (
                -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                + r * K * disc_r * norm.cdf(-d2)
                - q * S * disc_q * norm.cdf(-d1)
            )
            rho = -K * T * disc_r * norm.cdf(-d2)

        price = cls.price(S, K, T, r, sigma, q, option_type)
        return Greeks(
            price=float(price),
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta_yr / 365.0),  # per calendar day
            rho=float(rho),
        )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Back out the volatility that reprices the option to its market price.

    There's no closed form, so we root-find. Vega is strictly positive, so price is
    monotone in vol and a bracketed solver (Brent) is robust and fast — I prefer it to
    Newton here because it can't shoot off to a negative vol on a bad initial guess. Returns
    NaN if the quote is below intrinsic / outside an arbitrage-free range, which is the
    honest answer rather than a garbage number.
    """
    intrinsic = max(0.0, (S - K) if option_type == OptionType.CALL else (K - S)) * np.exp(-q * T)
    if market_price < intrinsic - 1e-8 or market_price <= 0:
        return float("nan")

    objective = lambda sig: BlackScholes.price(S, K, T, r, sig, q, option_type) - market_price  # noqa: E731
    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=100, xtol=1e-8))
    except ValueError:
        return float("nan")
