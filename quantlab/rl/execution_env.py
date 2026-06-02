"""The optimal-liquidation environment (Almgren-Chriss as an MDP).

We must liquidate `total_shares` over `n_steps` discrete intervals. Every interval we choose
how many units to sell. The tension is the classic one:

  * sell too fast and you pay convex **temporary impact** (η·n²) for demanding liquidity;
  * sell too slow and you carry **inventory risk** — the price can drift away while you wait.

Dynamics over one step, selling n units at price P:
    temporary impact:   execution price S = P − η·n        (you receive less, the more you rush)
    permanent impact:   P_next = P + σ·ξ − γ·n             (your selling pushes the price down)
    implementation cost vs the arrival price P₀:  n·(P₀ − S)

The reward is the *negative* of (realised cost + a mean-variance risk penalty on the
inventory you're still holding). With zero risk aversion the cost is just the convex impact,
which is minimised by selling in equal slices — i.e. **TWAP is optimal**. Turn risk aversion
up and the optimum **front-loads**: get the risk off the book early. A good agent should
reproduce exactly that shift, which is what makes this a clean, checkable RL demo rather than
an unfalsifiable one.

State and action are kept integer and small so a tabular agent can solve it exactly:
    state  = (step index k, remaining inventory)        both integers
    action = units to sell this step, in [0, remaining]  (forced to clear at the last step)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnvParams:
    total_shares: int = 10        # inventory in integer "units" (e.g. lots)
    n_steps: int = 5              # number of trading intervals
    eta: float = 0.02             # temporary-impact coefficient (per unit traded)
    gamma: float = 0.005          # permanent-impact coefficient
    sigma: float = 0.30           # per-step price volatility (price units)
    risk_aversion: float = 0.0    # weight on inventory risk; 0 -> TWAP optimal
    start_price: float = 100.0


class ExecutionEnv:
    """A minimal Gym-style environment: reset() / step(action) with an integer state."""

    def __init__(self, params: EnvParams | None = None, seed: int = 0):
        self.p = params or EnvParams()
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def n_actions(self) -> int:
        # Actions 0..total_shares (we mask the illegal ones per state).
        return self.p.total_shares + 1

    def reset(self) -> tuple[int, int]:
        self.k = 0
        self.inventory = self.p.total_shares
        self.price = self.p.start_price
        self.arrival = self.p.start_price
        return self._state()

    def _state(self) -> tuple[int, int]:
        return (self.k, self.inventory)

    def legal_actions(self) -> list[int]:
        # On the final step you must sell everything left; otherwise anything up to remaining.
        if self.k == self.p.n_steps - 1:
            return [self.inventory]
        return list(range(self.inventory + 1))

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        p = self.p
        n = int(np.clip(action, 0, self.inventory))
        if self.k == p.n_steps - 1:
            n = self.inventory  # forced liquidation at the horizon

        # Temporary impact: the price we actually transact at this step.
        exec_price = self.price - p.eta * n
        # Cost of this slice, measured against the arrival price (implementation shortfall).
        slice_cost = n * (self.arrival - exec_price)

        self.inventory -= n
        # Permanent impact + a random walk in the mid for the next step.
        shock = self.rng.normal(0.0, p.sigma)
        self.price = self.price - p.gamma * n + shock

        # Mean-variance risk penalty on the inventory still exposed to the next move.
        risk_penalty = p.risk_aversion * (p.sigma ** 2) * (self.inventory ** 2)

        reward = -(slice_cost + risk_penalty)
        self.k += 1
        done = self.k >= p.n_steps or self.inventory <= 0
        return self._state(), float(reward), done
