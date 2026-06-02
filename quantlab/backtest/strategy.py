"""Strategy base class.

A strategy's only job is to look at data up to "now" and emit target weights. It owns no
cash, holds no positions, and never sees a fill — that separation is what keeps strategies
testable in isolation and lets the same strategy run unchanged in research and (in
principle) live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from quantlab.backtest.events import SignalEvent


class Strategy(ABC):
    """Subclass and implement `generate_signals`.

    The engine calls `generate_signals` once per bar, handing over the full price history
    *up to and including* the current bar (never the future). Return a dict of
    {symbol: target_weight}; the engine wraps these into SignalEvents. Returning an empty
    dict means "no change requested".
    """

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, timestamp: date, history: pd.DataFrame) -> dict[str, float]:
        """`history` is a wide adj-close panel indexed by date, sliced to <= timestamp."""
        raise NotImplementedError

    def to_events(self, timestamp: date, weights: dict[str, float]) -> list[SignalEvent]:
        return [
            SignalEvent(symbol=sym, target_weight=float(w), timestamp=timestamp)
            for sym, w in weights.items()
        ]
