"""Portfolio accounting: the single source of truth for cash, positions, and equity.

This is where target weights become share counts and where fills are reconciled. A few
design decisions worth flagging:

  * Sizing is done off the *current* portfolio equity, so a strategy that says "10% in
    AAPL" gets 10% of the live book, not 10% of the starting capital. This is what keeps
    leverage controlled as the book grows or shrinks.
  * We only trade the difference between target and current shares (rebalance to target),
    which avoids churning the whole book every bar and keeps costs realistic.
  * A small no-trade band suppresses dust trades that would otherwise rack up commission
    for a one-share drift. Real desks do exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantlab.backtest.events import FillEvent, OrderEvent, OrderSide, SignalEvent


@dataclass
class Portfolio:
    initial_capital: float
    max_position_weight: float = 0.10
    # Don't bother trading if the target differs from current by less than this fraction
    # of equity. Kills dust trades; tune against your cost model.
    rebalance_band: float = 0.005

    cash: float = field(init=False)
    positions: dict[str, float] = field(init=False)  # symbol -> shares (signed)
    target_weights: dict[str, float] = field(init=False)
    equity_history: list[tuple] = field(init=False)  # (date, equity)
    trade_log: list[dict] = field(init=False)

    def __post_init__(self):
        self.cash = float(self.initial_capital)
        self.positions = {}
        self.target_weights = {}
        self.equity_history = []
        self.trade_log = []

    # --- valuation --------------------------------------------------------------------
    def market_value(self, prices: pd.Series) -> float:
        """Mark positions to the given prices and add cash."""
        mv = self.cash
        for sym, shares in self.positions.items():
            px = prices.get(sym)
            if px is not None and pd.notna(px):
                mv += shares * px
        return mv

    def record_equity(self, timestamp, prices: pd.Series) -> None:
        self.equity_history.append((timestamp, self.market_value(prices)))

    # --- signal -> orders -------------------------------------------------------------
    def generate_orders(self, signals: list[SignalEvent], prices: pd.Series) -> list[OrderEvent]:
        """Translate target weights into the orders needed to reach them.

        Clamps each name to `max_position_weight` (a hard per-name risk cap that no
        strategy can override), then trades only the gap beyond the rebalance band.
        """
        for sig in signals:
            capped = max(-self.max_position_weight, min(self.max_position_weight, sig.target_weight))
            self.target_weights[sig.symbol] = capped

        equity = self.market_value(prices)
        orders: list[OrderEvent] = []
        for sym, target_w in self.target_weights.items():
            px = prices.get(sym)
            if px is None or pd.isna(px) or px <= 0:
                continue
            target_shares = target_w * equity / px
            current_shares = self.positions.get(sym, 0.0)
            delta = target_shares - current_shares

            # No-trade band, measured in notional relative to equity.
            if abs(delta * px) < self.rebalance_band * equity:
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(
                OrderEvent(symbol=sym, side=side, quantity=abs(delta), timestamp=sig.timestamp if signals else None)
            )
        return orders

    # --- fill -> state ----------------------------------------------------------------
    def apply_fill(self, fill: FillEvent) -> None:
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) + signed_qty
        # Cash out: pay notional + commission on a buy, receive notional - commission on a sell.
        notional = fill.fill_price * fill.quantity
        self.cash -= signed_qty * fill.fill_price  # buy reduces cash, sell increases it
        self.cash -= fill.commission
        self.trade_log.append(
            {
                "date": fill.timestamp,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "fill_price": fill.fill_price,
                "commission": fill.commission,
                "slippage_cost": fill.slippage_cost,
            }
        )

    def equity_curve(self) -> pd.Series:
        if not self.equity_history:
            return pd.Series(dtype=float)
        dates, vals = zip(*self.equity_history)
        return pd.Series(vals, index=pd.DatetimeIndex(dates), name="equity")

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)
