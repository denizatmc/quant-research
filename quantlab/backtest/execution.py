"""Execution simulation: turning orders into fills with a defensible cost model.

Transaction costs are where optimistic backtests go to die, so I'd rather over-model them
than under-model them. The cost of a fill has three pieces:

  1. Commission      — a flat per-notional fee (bps).
  2. Spread/slippage — a fixed half-spread cost (bps), the price of demanding liquidity.
  3. Market impact   — a square-root term in participation rate. This is the empirically
                       motivated bit (Almgren et al.): impact grows like sqrt(Q/ADV), not
                       linearly, so large orders are penalised but not absurdly so.

All three are charged *against* the trader (you always cross the spread, impact always
moves price the wrong way), which keeps the simulation conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantlab.backtest.events import FillEvent, OrderEvent, OrderSide


@dataclass
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    # Coefficient on the square-root impact term. 0.1 is a deliberately modest default;
    # the participation rate (order size / ADV) does most of the work.
    impact_coef: float = 0.1

    def fill_price(self, mid_price: float, side: OrderSide, participation: float) -> tuple[float, float]:
        """Return (fill_price, slippage_cost_per_share).

        `participation` is order quantity as a fraction of average daily volume. The
        slippage is a fraction of price: a fixed half-spread plus the sqrt-impact term.
        """
        half_spread = self.slippage_bps / 1e4
        impact = self.impact_coef * (max(participation, 0.0) ** 0.5)
        slip_frac = half_spread + impact
        # Buys pay up, sells receive less — slippage always hurts.
        signed = slip_frac if side == OrderSide.BUY else -slip_frac
        fill = mid_price * (1.0 + signed)
        return fill, abs(fill - mid_price)


class SimulatedExecutionHandler:
    """Fills market orders at the bar's reference price plus modelled costs.

    The reference price passed in is the execution price *before* costs — in the daily
    engine that's the same bar's close, i.e. we decide on data up to the close and fill at
    that close with slippage. No next-bar magic, no fills at the low of the day.
    """

    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def execute(self, order: OrderEvent, ref_price: float, adv: float) -> FillEvent:
        participation = order.quantity / adv if adv and adv > 0 else 0.0
        fill_price, slip_per_share = self.cost_model.fill_price(ref_price, order.side, participation)
        notional = fill_price * order.quantity
        commission = notional * self.cost_model.commission_bps / 1e4
        slippage_cost = slip_per_share * order.quantity
        return FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage_cost,
            timestamp=order.timestamp,
        )
