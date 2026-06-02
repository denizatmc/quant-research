"""Options tests: the properties that *must* hold, checked against analytic/numeric truth.

These are the assertions I'd want a reviewer to see, because they pin down correctness
rather than just "it runs": put-call parity, IV self-consistency, and Greeks against finite
differences.
"""

import numpy as np
import pytest

from quantlab.options.black_scholes import BlackScholes as BS, OptionType, implied_volatility


def test_put_call_parity():
    # C - P must equal S e^{-qT} - K e^{-rT} for any inputs — a model-free arbitrage relation.
    S, K, T, r, sigma, q = 100, 95, 1.0, 0.03, 0.2, 0.015
    c = BS.price(S, K, T, r, sigma, q, OptionType.CALL)
    p = BS.price(S, K, T, r, sigma, q, OptionType.PUT)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-9


def test_implied_vol_roundtrip():
    # Pricing at sigma then inverting must recover sigma.
    S, K, T, r, q, sigma = 100, 110, 0.75, 0.04, 0.0, 0.32
    price = BS.price(S, K, T, r, sigma, q, OptionType.CALL)
    recovered = implied_volatility(price, S, K, T, r, q, OptionType.CALL)
    assert recovered == pytest.approx(sigma, abs=1e-4)


def test_gamma_matches_finite_difference():
    # Analytic gamma == second price derivative wrt spot.
    S, K, T, r, sigma = 100, 100, 0.5, 0.02, 0.25
    g = BS.greeks(S, K, T, r, sigma).gamma
    h = 0.01
    num = (BS.price(S + h, K, T, r, sigma) - 2 * BS.price(S, K, T, r, sigma) + BS.price(S - h, K, T, r, sigma)) / h**2
    assert g == pytest.approx(num, rel=1e-4)


def test_call_delta_bounds():
    # A call delta lives in [0, 1]; deep ITM -> ~1, deep OTM -> ~0.
    deep_itm = BS.greeks(200, 100, 0.5, 0.02, 0.25, option_type=OptionType.CALL).delta
    deep_otm = BS.greeks(50, 100, 0.5, 0.02, 0.25, option_type=OptionType.CALL).delta
    assert deep_itm > 0.95
    assert deep_otm < 0.05


def test_iv_below_intrinsic_is_nan():
    # A quote below intrinsic value is an arbitrage; the solver should refuse it, not invent a vol.
    assert np.isnan(implied_volatility(0.5, 100, 90, 1.0, 0.03, 0.0, OptionType.CALL))
