"""Statistical arbitrage: a cointegration-based pairs trade.

This is the strategy I'd point a reviewer to first, because it's where the statistics
actually matter. The premise: two economically related stocks (say KO and PG) share a
common stochastic trend, so while each price is individually non-stationary (I(1)), a
particular linear combination of them — the spread — is stationary (I(0)). When the spread
wanders far from its mean we bet on reversion.

The hedge ratio comes from an OLS regression of one leg on the other, re-estimated on a
trailing window so it can drift. We trade the z-score of the residual spread: short the
spread when it's rich, long it when it's cheap, flatten near the mean. A wide stop guards
against the regime where cointegration simply breaks (the spread becomes a random walk and
never comes back — the way most pairs trades actually lose money).

The formal cointegration testing (Engle-Granger / ADF) lives in quantlab.models.timeseries
and is exercised in the research notebook; here we use the rolling hedge ratio and trade
the standardised spread.
"""

from __future__ import annotations

from collections import deque
from datetime import date

import numpy as np
import pandas as pd

from quantlab.backtest.strategy import Strategy
from quantlab.models.kalman import KalmanHedgeRatio


class PairsTradingStrategy(Strategy):
    name = "pairs_statarb"

    def __init__(
        self,
        symbol_y: str,
        symbol_x: str,
        lookback: int = 63,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        gross: float = 1.0,
    ):
        self.y = symbol_y
        self.x = symbol_x
        self.lookback = lookback
        self.entry_z = entry_z      # enter when |z| exceeds this
        self.exit_z = exit_z        # exit when |z| falls back inside this
        self.stop_z = stop_z        # bail if |z| blows past this (cointegration broke)
        self.gross = gross
        self._position = 0          # -1 short spread, 0 flat, +1 long spread

    def _hedge_ratio_and_z(self, hist: pd.DataFrame) -> tuple[float, float]:
        """Rolling OLS hedge ratio beta (y ~ beta*x) and the current spread z-score.

        Spread_t = y_t - beta * x_t. We standardise it over the same window. Using a
        rolling beta (rather than one fixed in-sample beta) is what keeps this honest
        out-of-sample — the relationship is allowed to evolve.
        """
        window = hist.iloc[-self.lookback :]
        y = window[self.y].values
        x = window[self.x].values
        # beta via covariance / variance — equivalent to OLS slope without an intercept on
        # the demeaned series, which is what we want for a spread.
        x_d = x - x.mean()
        y_d = y - y.mean()
        beta = float((x_d @ y_d) / (x_d @ x_d)) if (x_d @ x_d) != 0 else 0.0
        spread = window[self.y].values - beta * window[self.x].values
        mu, sigma = spread.mean(), spread.std(ddof=1)
        z = (spread[-1] - mu) / sigma if sigma > 0 else 0.0
        return beta, float(z)

    def generate_signals(self, timestamp: date, history: pd.DataFrame) -> dict[str, float]:
        if self.y not in history.columns or self.x not in history.columns:
            return {}
        if len(history) < self.lookback:
            return {}

        beta, z = self._hedge_ratio_and_z(history)

        # State machine on the z-score. "Long spread" = long y, short beta*x.
        prev = self._position
        if self._position == 0:
            if z > self.entry_z:
                self._position = -1   # spread rich -> short it
            elif z < -self.entry_z:
                self._position = +1   # spread cheap -> long it
        else:
            # Exit on reversion to the mean, or stop out if it ran away.
            if abs(z) < self.exit_z or abs(z) > self.stop_z:
                self._position = 0

        if self._position == prev and self._position == 0:
            return {}  # nothing to do, stay flat

        if self._position == 0:
            return {self.y: 0.0, self.x: 0.0}

        # Scale the two legs so the dollar exposure of each side is balanced by beta and
        # the total gross is roughly `gross`. Normalise by (1 + |beta|) so a big beta
        # doesn't silently lever the book up.
        scale = self.gross / (1.0 + abs(beta))
        y_w = self._position * scale
        x_w = -self._position * beta * scale
        return {self.y: float(y_w), self.x: float(x_w)}


class KalmanPairsStrategy(Strategy):
    """The same pairs trade, but with the hedge ratio tracked by a Kalman filter.

    Two practical wins over the rolling-OLS version: there's no arbitrary lookback window to
    pick, and the standardised innovation from the filter is the z-score directly, so the
    spread's scale adapts on its own. The filter keeps its state *across* bars — the engine
    feeds bars in order, exactly once each, so I update with only the newest observation
    (O(1) per bar) rather than re-filtering all of history every time.
    """

    name = "pairs_kalman"

    def __init__(
        self,
        symbol_y: str,
        symbol_x: str,
        delta: float = 1e-4,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 5.0,
        gross: float = 1.0,
        warmup_bars: int = 30,
    ):
        self.y = symbol_y
        self.x = symbol_x
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.gross = gross
        self.warmup_bars = warmup_bars
        self.kf = KalmanHedgeRatio(delta=delta)
        self._position = 0
        self._last_seen = None  # guard against double-updating the same bar
        self._n_updates = 0
        # EWMA variance of the innovation, used to standardise the spread. The filter's own
        # predicted variance turns out to be a poor scale here (it over-states uncertainty),
        # so in practice it's more robust to z-score the innovation by its realised vol.
        self._ewma_var = None
        self._lambda = 0.04  # ~17-bar half-life
        self._last_e = 0.0   # most recent *pre-update* innovation = the spread

    def generate_signals(self, timestamp, history):
        if self.y not in history.columns or self.x not in history.columns:
            return {}
        row = history.iloc[-1]
        beta = float(self.kf.beta[1])
        # Update the filter once per new bar, capturing the one-step prediction error. This
        # innovation is computed *before* the state absorbs the new point, so it's the genuine
        # surprise in the spread — using the post-update residual instead would be ~0 by
        # construction (the filter just fit itself to that point) and signal nothing.
        if self._last_seen != history.index[-1]:
            e, _, b = self.kf.update(float(row[self.x]), float(row[self.y]))
            beta = float(b[1])
            self._last_e = e
            self._last_seen = history.index[-1]
            self._n_updates += 1
            # Online EWMA of the squared innovation -> a self-scaling spread vol.
            self._ewma_var = e * e if self._ewma_var is None else (
                (1 - self._lambda) * self._ewma_var + self._lambda * e * e
            )

        # Let the filter settle before trusting its innovation as a signal.
        if self._n_updates < self.warmup_bars or not self._ewma_var:
            return {}

        # Standardise the innovation by its realised EWMA vol.
        z = self._last_e / np.sqrt(max(self._ewma_var, 1e-12))

        prev = self._position
        if self._position == 0:
            if z > self.entry_z:
                self._position = -1
            elif z < -self.entry_z:
                self._position = +1
        else:
            if abs(z) < self.exit_z or abs(z) > self.stop_z:
                self._position = 0

        if self._position == prev and self._position == 0:
            return {}
        if self._position == 0:
            return {self.y: 0.0, self.x: 0.0}

        scale = self.gross / (1.0 + abs(beta))
        return {self.y: float(self._position * scale), self.x: float(-self._position * beta * scale)}
