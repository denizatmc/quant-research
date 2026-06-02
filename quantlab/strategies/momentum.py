"""Cross-sectional momentum — the classic Jegadeesh-Titman effect.

The idea: rank the universe by past return over a lookback (skipping the most recent month
to avoid the well-documented short-term reversal), go long the winners and short the
losers, dollar-neutral. It's a factor that has paid out for decades and then bled in
others, which makes it a good, honest test of the whole pipeline — including whether costs
eat the edge.

Rebalancing monthly rather than daily is deliberate: momentum decays slowly, so daily
rebalancing just pays extra costs for noise.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantlab.backtest.strategy import Strategy


class CrossSectionalMomentum(Strategy):
    name = "xs_momentum"

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        top_n: int = 3,
        rebalance_days: int = 21,
        exclude: tuple[str, ...] = ("SPY",),
    ):
        self.lookback = lookback
        self.skip = skip          # skip the most recent `skip` days (reversal)
        self.top_n = top_n        # long the top_n, short the bottom_n
        self.rebalance_days = rebalance_days
        self.exclude = set(exclude)
        self._last_rebalance: date | None = None

    def _is_rebalance_day(self, history: pd.DataFrame) -> bool:
        # Rebalance on a fixed bar cadence. Using bar count rather than calendar months
        # keeps the schedule independent of holidays.
        n = len(history)
        return (n % self.rebalance_days) == 0

    def generate_signals(self, timestamp: date, history: pd.DataFrame) -> dict[str, float]:
        if len(history) < self.lookback + self.skip:
            return {}
        if not self._is_rebalance_day(history):
            return {}  # hold between rebalances

        cols = [c for c in history.columns if c not in self.exclude]
        # Momentum = return from (t - lookback - skip) to (t - skip).
        past = history[cols].iloc[-(self.lookback + self.skip)]
        recent = history[cols].iloc[-(self.skip + 1)]
        mom = (recent / past - 1.0).dropna()
        if len(mom) < 2 * self.top_n:
            return {}

        ranked = mom.sort_values()
        losers = ranked.index[: self.top_n]
        winners = ranked.index[-self.top_n :]

        # Dollar-neutral, equal weight each side. Gross exposure ≈ 1.0.
        long_w = 0.5 / self.top_n
        short_w = -0.5 / self.top_n
        weights = {sym: 0.0 for sym in cols}  # flatten anything not selected
        for s in winners:
            weights[s] = long_w
        for s in losers:
            weights[s] = short_w
        return weights
