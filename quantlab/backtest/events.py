"""The event types that flow through the backtester's queue.

Keeping these as small, frozen dataclasses makes the event log trivially serialisable and
easy to assert against in tests. The four-event cycle (Market → Signal → Order → Fill) is
the same vocabulary an OMS uses, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class Event:
    """Marker base class so the queue can hold a heterogeneous stream."""

    type: str = "BASE"


@dataclass
class MarketEvent(Event):
    """A new bar has arrived. Carries the timestamp; the strategy/portfolio read prices
    from the shared data handler rather than copying them onto every event."""

    type: str = "MARKET"
    timestamp: date | None = None


@dataclass
class SignalEvent(Event):
    """A strategy's view on one symbol: a target portfolio weight in [-1, 1].

    I chose target weights rather than buy/sell verbs because it makes position sizing the
    portfolio's job (one place), keeps strategies stateless about share counts, and makes
    rebalancing logic fall out naturally.
    """

    symbol: str
    target_weight: float
    timestamp: date | None = None
    type: str = "SIGNAL"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderEvent(Event):
    """A concrete instruction to trade `quantity` shares of `symbol`.

    Quantities are signed-positive with an explicit side; market orders only, which is the
    honest level of fidelity for a daily backtest filled at the next close.
    """

    symbol: str
    side: OrderSide
    quantity: float
    timestamp: date | None = None
    type: str = "ORDER"


@dataclass
class FillEvent(Event):
    """The result of executing an order: the price actually paid (incl. slippage) and the
    commission charged. The portfolio reconciles cash and positions off these."""

    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    commission: float
    slippage_cost: float
    timestamp: date | None = None
    type: str = "FILL"
