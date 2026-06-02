"""The backtest driver: the event loop and the result/statistics object.

The loop walks the calendar one bar at a time. At each bar it pushes a MarketEvent and
drains the queue to completion before advancing — so all of a bar's signals, orders, and
fills settle before the next bar arrives. That ordering is what guarantees no lookahead:
the strategy only ever sees history up to the current bar, and fills happen at the current
bar's price.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.backtest.events import FillEvent, MarketEvent, OrderEvent
from quantlab.backtest.execution import CostModel, SimulatedExecutionHandler
from quantlab.backtest.portfolio import Portfolio
from quantlab.backtest.strategy import Strategy
from quantlab.utils import (
    annualize_return,
    annualize_vol,
    max_drawdown,
    sharpe_ratio,
)


@dataclass
class BacktestResult:
    """Everything a run produces, plus the performance summary computed off it."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    benchmark: pd.Series | None = None

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    def stats(self) -> dict[str, float]:
        r = self.returns
        if r.empty:
            return {}
        eq = self.equity_curve
        downside = r[r < 0]
        sortino = (
            np.sqrt(252) * r.mean() / downside.std(ddof=1)
            if not downside.empty and downside.std(ddof=1) > 0
            else float("nan")
        )
        out = {
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
            "cagr": annualize_return(r),
            "ann_vol": annualize_vol(r),
            "sharpe": sharpe_ratio(r),
            "sortino": float(sortino),
            "max_drawdown": max_drawdown(eq),
            "calmar": (
                annualize_return(r) / abs(max_drawdown(eq))
                if max_drawdown(eq) != 0
                else float("nan")
            ),
            "n_trades": int(len(self.trades)),
            "total_costs": float(
                self.trades[["commission", "slippage_cost"]].sum().sum()
            )
            if not self.trades.empty
            else 0.0,
        }
        # If a benchmark is supplied, add the information ratio of excess returns.
        if self.benchmark is not None:
            bench_r = self.benchmark.reindex(r.index).pct_change().dropna()
            common = r.index.intersection(bench_r.index)
            active = r.loc[common] - bench_r.loc[common]
            if active.std(ddof=1) > 0:
                out["info_ratio"] = float(np.sqrt(252) * active.mean() / active.std(ddof=1))
        return out

    def summary(self) -> str:
        s = self.stats()
        lines = ["Performance summary", "-" * 32]
        fmt = {
            "total_return": "{:.1%}", "cagr": "{:.2%}", "ann_vol": "{:.2%}",
            "sharpe": "{:.2f}", "sortino": "{:.2f}", "max_drawdown": "{:.1%}",
            "calmar": "{:.2f}", "info_ratio": "{:.2f}", "n_trades": "{:d}",
            "total_costs": "${:,.0f}",
        }
        labels = {
            "total_return": "Total return", "cagr": "CAGR", "ann_vol": "Ann. vol",
            "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max drawdown",
            "calmar": "Calmar", "info_ratio": "Info ratio", "n_trades": "# trades",
            "total_costs": "Total costs",
        }
        for k, v in s.items():
            lines.append(f"  {labels.get(k, k):14s} {fmt.get(k, '{}').format(v)}")
        return "\n".join(lines)


class Backtest:
    """Wire together a price panel, a strategy, and the cost model, then run the loop."""

    def __init__(
        self,
        prices: pd.DataFrame,            # wide adj_close, index=date, cols=symbols
        strategy: Strategy,
        volume: pd.DataFrame | None = None,  # wide volume, for the impact model's ADV
        initial_capital: float = 1_000_000,
        cost_model: CostModel | None = None,
        warmup: int = 252,
        max_position_weight: float = 0.10,
        benchmark_col: str | None = None,
    ):
        self.prices = prices.sort_index()
        self.strategy = strategy
        self.volume = volume.sort_index() if volume is not None else None
        self.cost_model = cost_model or CostModel()
        self.warmup = warmup
        self.benchmark_col = benchmark_col

        self.portfolio = Portfolio(
            initial_capital=initial_capital, max_position_weight=max_position_weight
        )
        self.execution = SimulatedExecutionHandler(self.cost_model)
        self.events: deque = deque()

    def _adv(self, symbol: str, upto_idx: int, window: int = 21) -> float:
        """Average daily volume over a trailing window, for the participation calc.
        Falls back to a large number (≈ no impact) when volume isn't available."""
        if self.volume is None or symbol not in self.volume.columns:
            return 1e12
        lo = max(0, upto_idx - window)
        v = self.volume[symbol].iloc[lo:upto_idx]
        adv = v.mean()
        return float(adv) if adv and adv > 0 else 1e12

    def run(self) -> BacktestResult:
        dates = self.prices.index
        for i in range(self.warmup, len(dates)):
            t = dates[i]
            row = self.prices.iloc[i]                       # today's close per symbol
            history = self.prices.iloc[: i + 1]             # everything up to & incl. today

            # 1) Bar arrives.
            self.events.append(MarketEvent(timestamp=t))

            while self.events:
                event = self.events.popleft()

                if event.type == "MARKET":
                    weights = self.strategy.generate_signals(t, history)
                    if weights:
                        signals = self.strategy.to_events(t, weights)
                        # Size all signals together so weights are consistent with one equity snapshot.
                        for order in self.portfolio.generate_orders(signals, row):
                            self.events.append(order)

                elif event.type == "ORDER":
                    order: OrderEvent = event
                    ref_price = row.get(order.symbol)
                    if ref_price is None or pd.isna(ref_price):
                        continue
                    adv = self._adv(order.symbol, i)
                    self.events.append(self.execution.execute(order, float(ref_price), adv))

                elif event.type == "FILL":
                    self.portfolio.apply_fill(event)

            # 2) Bar fully settled — mark the book.
            self.portfolio.record_equity(t, row)

        benchmark = None
        if self.benchmark_col and self.benchmark_col in self.prices.columns:
            benchmark = self.prices[self.benchmark_col].iloc[self.warmup :]

        return BacktestResult(
            equity_curve=self.portfolio.equity_curve(),
            trades=self.portfolio.trades_frame(),
            benchmark=benchmark,
        )
