"""Backtester tests — most importantly, that it doesn't look into the future.

The no-lookahead test is the one that matters: it's the single assumption a backtest can
violate to produce beautiful, fictional returns. We check it directly by feeding a strategy
that records what history it was shown and asserting it never saw a price dated after the
decision timestamp.
"""

import numpy as np
import pandas as pd

from quantlab.backtest.engine import Backtest
from quantlab.backtest.execution import CostModel, SimulatedExecutionHandler
from quantlab.backtest.events import OrderEvent, OrderSide
from quantlab.backtest.strategy import Strategy


def _toy_prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    paths = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=(n, 2)), axis=0)
    return pd.DataFrame(paths, index=idx, columns=["AAA", "BBB"])


class _SpyStrategy(Strategy):
    """Records the last timestamp it saw in history vs the decision timestamp."""
    name = "spy"

    def __init__(self):
        self.violations = 0

    def generate_signals(self, timestamp, history):
        # The engine hands us a Timestamp; history must never extend past it.
        if history.index.max() > timestamp:
            self.violations += 1
        return {"AAA": 0.05}


def test_no_lookahead():
    prices = _toy_prices()
    strat = _SpyStrategy()
    Backtest(prices, strat, warmup=20).run()
    assert strat.violations == 0


def test_equity_curve_starts_at_capital():
    prices = _toy_prices()

    class Flat(Strategy):
        def generate_signals(self, timestamp, history):
            return {}

    res = Backtest(prices, Flat(), initial_capital=1_000_000, warmup=20).run()
    # With no trades the book is all cash and never moves from the starting capital.
    assert res.equity_curve.iloc[0] == 1_000_000
    assert res.equity_curve.nunique() == 1


def test_costs_are_charged_against_the_trader():
    # A buy must fill above the reference price; a sell below it. Slippage never helps.
    cm = CostModel(commission_bps=1.0, slippage_bps=5.0)
    handler = SimulatedExecutionHandler(cm)
    buy = handler.execute(OrderEvent("X", OrderSide.BUY, 100), ref_price=100.0, adv=1e9)
    sell = handler.execute(OrderEvent("X", OrderSide.SELL, 100), ref_price=100.0, adv=1e9)
    assert buy.fill_price > 100.0
    assert sell.fill_price < 100.0
    assert buy.commission > 0
