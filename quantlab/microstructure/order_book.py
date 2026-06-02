"""A price-time-priority limit order book.

This is a from-scratch matching engine — the thing that sits underneath every exchange,
BISTECH included. I wrote it out rather than mocking it because the mechanics *are* the
microstructure: how price-time priority resolves competing orders, how a marketable order
walks the book and pays successively worse prices (the queue's revenge on impatience), and
how the touch moves as liquidity is consumed.

Design:
  * Two sides, each a dict {price -> FIFO queue of resting orders}. Best bid/ask are found
    over the live price levels. (A heap or a sorted structure would be the move for a hot
    path; for a teaching/simulation engine the dict keeps the logic legible.)
  * Matching is strict price-time priority: best price first, and within a price level the
    order that arrived earliest fills first.
  * A market or crossing limit order is matched immediately against the opposite side; any
    unfilled remainder of a limit order rests in the book.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import count


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    order_id: int
    side: Side
    price: float | None       # None => market order
    quantity: float
    seq: int                  # arrival sequence, for time priority


@dataclass
class Trade:
    price: float
    quantity: float
    aggressor: Side
    resting_id: int
    incoming_id: int


@dataclass
class LimitOrderBook:
    bids: dict[float, deque] = field(default_factory=dict)   # price -> queue of Orders
    asks: dict[float, deque] = field(default_factory=dict)
    _ids: count = field(default_factory=lambda: count(1))
    _seq: count = field(default_factory=lambda: count(1))

    # --- inspection -------------------------------------------------------------------
    def best_bid(self) -> float | None:
        live = [p for p, q in self.bids.items() if q]
        return max(live) if live else None

    def best_ask(self) -> float | None:
        live = [p for p, q in self.asks.items() if q]
        return min(live) if live else None

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        return 0.5 * (b + a) if b is not None and a is not None else None

    def spread(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        return (a - b) if b is not None and a is not None else None

    def depth(self, side: Side, price: float) -> float:
        book = self.bids if side == Side.BUY else self.asks
        return sum(o.quantity for o in book.get(price, []))

    # --- mutation ---------------------------------------------------------------------
    def add_limit(self, side: Side, price: float, quantity: float) -> tuple[int, list[Trade]]:
        """Submit a limit order. Returns (order_id, trades it caused on the way in)."""
        order = Order(next(self._ids), side, price, quantity, next(self._seq))
        trades = self._match(order)
        if order.quantity > 0:  # rest whatever didn't fill
            book = self.bids if side == Side.BUY else self.asks
            book.setdefault(price, deque()).append(order)
        return order.order_id, trades

    def add_market(self, side: Side, quantity: float) -> list[Trade]:
        """Submit a market order — matches immediately, never rests. Walks the book and
        pays progressively worse prices until filled or the book runs dry."""
        order = Order(next(self._ids), side, None, quantity, next(self._seq))
        return self._match(order)

    def cancel(self, order_id: int) -> bool:
        for book in (self.bids, self.asks):
            for q in book.values():
                for o in list(q):
                    if o.order_id == order_id:
                        q.remove(o)
                        return True
        return False

    # --- matching core ----------------------------------------------------------------
    def _match(self, incoming: Order) -> list[Trade]:
        trades: list[Trade] = []
        # An incoming buy lifts the asks; an incoming sell hits the bids.
        opposite = self.asks if incoming.side == Side.BUY else self.bids

        def best_price():
            live = [p for p, q in opposite.items() if q]
            if not live:
                return None
            return min(live) if incoming.side == Side.BUY else max(live)

        while incoming.quantity > 0:
            bp = best_price()
            if bp is None:
                break
            # Respect the limit price: a buy won't pay above its limit, a sell won't sell below.
            if incoming.price is not None:
                if incoming.side == Side.BUY and bp > incoming.price:
                    break
                if incoming.side == Side.SELL and bp < incoming.price:
                    break

            level = opposite[bp]
            while incoming.quantity > 0 and level:
                resting = level[0]              # FIFO: oldest at this price fills first
                fill = min(incoming.quantity, resting.quantity)
                trades.append(
                    Trade(
                        price=bp,
                        quantity=fill,
                        aggressor=incoming.side,
                        resting_id=resting.order_id,
                        incoming_id=incoming.order_id,
                    )
                )
                incoming.quantity -= fill
                resting.quantity -= fill
                if resting.quantity <= 0:
                    level.popleft()             # fully consumed, leave the queue
            if not level:
                del opposite[bp]                # price level emptied
        return trades
