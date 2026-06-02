"""Order-book tests: price-time priority and the mechanics of walking the book."""

from quantlab.microstructure.order_book import LimitOrderBook, Side


def test_spread_and_touch():
    lob = LimitOrderBook()
    lob.add_limit(Side.BUY, 99.0, 100)
    lob.add_limit(Side.SELL, 101.0, 100)
    assert lob.best_bid() == 99.0
    assert lob.best_ask() == 101.0
    assert lob.spread() == 2.0
    assert lob.mid() == 100.0


def test_market_order_walks_the_book():
    # A market buy bigger than the touch should consume successive levels at worse prices.
    lob = LimitOrderBook()
    lob.add_limit(Side.SELL, 101.0, 100)
    lob.add_limit(Side.SELL, 102.0, 100)
    trades = lob.add_market(Side.BUY, 150)
    assert [t.price for t in trades] == [101.0, 102.0]
    assert [t.quantity for t in trades] == [100, 50]
    # 101 fully consumed; 102 partially -> best ask is now 102 with 50 left.
    assert lob.best_ask() == 102.0
    assert lob.depth(Side.SELL, 102.0) == 50


def test_time_priority_within_level():
    # Two resting sells at the same price: the one that arrived first must fill first.
    lob = LimitOrderBook()
    first, _ = lob.add_limit(Side.SELL, 101.0, 100)
    second, _ = lob.add_limit(Side.SELL, 101.0, 100)
    trades = lob.add_market(Side.BUY, 100)
    assert trades[0].resting_id == first  # FIFO honoured


def test_limit_order_rests_when_not_crossing():
    lob = LimitOrderBook()
    oid, trades = lob.add_limit(Side.BUY, 99.0, 100)
    assert trades == []                # didn't cross, so no trade
    assert lob.best_bid() == 99.0      # it's resting
    assert lob.cancel(oid) is True
    assert lob.best_bid() is None      # gone after cancel
