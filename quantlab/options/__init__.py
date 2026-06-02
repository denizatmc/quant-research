"""Options pricing, Greeks, and implied-volatility tools.

The role mentions VİOP (the Turkish derivatives exchange) and options pricing. The maths is
identical on US listed options, so that's what this module works with — Black-Scholes-Merton
for European-style pricing, the full first/second-order Greeks, and an implied-vol solver.
"""

from quantlab.options.black_scholes import (
    BlackScholes,
    OptionType,
    implied_volatility,
)

__all__ = ["BlackScholes", "OptionType", "implied_volatility"]
