"""A Kalman filter for a time-varying hedge ratio.

The pairs strategy in `quantlab.strategies.stat_arb` re-estimates its hedge ratio with a
rolling OLS window. That works, but it has two annoyances: the window length is an arbitrary
choice, and every estimate weights the whole window equally then drops the oldest point off a
cliff. A Kalman filter fixes both by treating the hedge ratio as a *hidden state that drifts*,
updated smoothly one observation at a time.

The setup (the standard dynamic linear regression, e.g. Chan's formulation):

    state    x_t = [alpha_t, beta_t]      — intercept and hedge ratio, assumed to follow a
                                            random walk:  x_t = x_{t-1} + w_t,  w_t ~ N(0, Vw)
    observe  y_t = [1, p^x_t] · x_t + v_t — leg y's price, regressed on leg x's price + noise

The filter's one-step prediction error e_t *is* the spread, and its variance S_t comes for
free — so the trading signal is simply the standardised innovation e_t / sqrt(S_t), with no
separate rolling-window standardisation needed. The single knob `delta` sets how fast the
hedge ratio is allowed to move: small delta → a nearly-static beta, larger delta → a beta
that chases the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class KalmanHedgeRatio:
    # delta controls the state (process) noise: Vw = delta/(1-delta) * I. This is the
    # standard parameterisation — it keeps the one free parameter in an interpretable [0,1).
    delta: float = 1e-4
    obs_cov: float = 1e-3  # measurement noise variance Ve

    beta: np.ndarray = field(init=False)   # [alpha, beta]
    P: np.ndarray = field(init=False)      # state covariance
    _Vw: np.ndarray = field(init=False)
    _initialized: bool = field(init=False, default=False)

    def __post_init__(self):
        self.beta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self._Vw = self.delta / (1.0 - self.delta) * np.eye(2)

    def update(self, x_price: float, y_price: float) -> tuple[float, float, np.ndarray]:
        """Process one observation. Returns (spread, spread_std, [alpha, beta]).

        `spread` is the innovation e_t = y_t - ŷ_t and `spread_std` = sqrt(S_t) is its
        predicted standard deviation, so the caller's z-score is just spread / spread_std.
        """
        H = np.array([1.0, x_price])  # observation matrix (1 x 2)

        if not self._initialized:
            # Seed beta with a sensible first guess (price ratio) and a diffuse prior, so the
            # filter doesn't spend its first hundred points crawling away from zero.
            self.beta = np.array([0.0, y_price / x_price if x_price else 0.0])
            self.P = np.eye(2) * 1.0
            self._initialized = True

        # Predict: state is a random walk (F = I), so we only inflate the covariance.
        R = self.P + self._Vw
        y_hat = H @ self.beta
        e = y_price - y_hat                       # innovation == the spread
        S = float(H @ R @ H.T + self.obs_cov)     # innovation variance (scalar)

        # Update.
        K = (R @ H) / S                           # Kalman gain (2,)
        self.beta = self.beta + K * e
        self.P = R - np.outer(K, H) @ R
        return float(e), float(np.sqrt(max(S, 1e-12))), self.beta.copy()

    def run(self, x_series: pd.Series, y_series: pd.Series) -> pd.DataFrame:
        """Filter two aligned price series offline; handy for research/plots.

        Returns a frame with the filtered alpha/beta, the spread, and its z-score over time.
        """
        df = pd.concat([y_series.rename("y"), x_series.rename("x")], axis=1).dropna()
        out = []
        for ts, row in df.iterrows():
            e, s, b = self.update(row["x"], row["y"])
            out.append({"date": ts, "alpha": b[0], "beta": b[1], "spread": e, "zscore": e / s})
        return pd.DataFrame(out).set_index("date")
