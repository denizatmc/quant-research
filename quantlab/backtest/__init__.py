"""An event-driven backtester.

The design is the textbook event loop: a data handler emits market events, a strategy
turns those into signals, a portfolio turns signals into orders, and an execution handler
turns orders into fills — all passed through a single event queue. It's more machinery
than a vectorised backtest, but it's the same shape as a live trading loop, so the path
from research to production is short and the cost/latency assumptions are explicit.
"""

from quantlab.backtest.events import (
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from quantlab.backtest.engine import Backtest, BacktestResult
from quantlab.backtest.strategy import Strategy

__all__ = [
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "Backtest",
    "BacktestResult",
    "Strategy",
]
