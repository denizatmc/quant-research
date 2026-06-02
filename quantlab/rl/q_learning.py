"""Tabular Q-learning for the execution environment, plus a TWAP baseline to beat.

Q-learning is the right tool here: the state space is small and discrete, so we can learn the
exact action-value function without function approximation and actually *inspect* the policy
it finds. The agent learns purely from experience — it's never told the Almgren-Chriss
solution — so recovering TWAP at zero risk aversion and a front-loaded schedule at positive
risk aversion is genuine evidence it learned the trade-off rather than memorised an answer.

ε-greedy exploration with a decaying ε; standard off-policy TD(0) update
    Q(s,a) ← Q(s,a) + α · [ r + γ·max_a' Q(s',a') − Q(s,a) ].
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from quantlab.rl.execution_env import EnvParams, ExecutionEnv


def twap_cost(params: EnvParams, n_episodes: int = 2000, seed: int = 1) -> float:
    """Mean cost of the naive TWAP schedule (sell equal-sized slices every step).

    This is the benchmark. Because temporary impact is convex, TWAP is the cost-minimising
    schedule when you don't care about risk — so at risk_aversion=0 a good agent should match
    it, not beat it, and the honest thing is to report that rather than manufacture an edge.
    """
    env = ExecutionEnv(params, seed=seed)
    # Spread the shares as evenly as integers allow.
    base, extra = divmod(params.total_shares, params.n_steps)
    schedule = [base + (1 if i < extra else 0) for i in range(params.n_steps)]
    total = 0.0
    for ep in range(n_episodes):
        env.rng = np.random.default_rng(seed + ep)
        env.reset()
        done = False
        i = 0
        ep_reward = 0.0
        while not done:
            _, r, done = env.step(schedule[i])
            ep_reward += r
            i += 1
        total += -ep_reward  # reward is negative cost; flip back to a cost
    return total / n_episodes


class QLearner:
    def __init__(self, params: EnvParams, alpha: float = 0.1, gamma: float = 1.0,
                 epsilon: float = 1.0, epsilon_min: float = 0.02, epsilon_decay: float = 0.9995,
                 seed: int = 0):
        self.params = params
        self.alpha = alpha
        self.gamma = gamma          # discount (1.0 — the horizon is short and finite)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        # Q[state] -> array over all actions; illegal actions stay at -inf so they're never picked.
        self.Q: dict = defaultdict(lambda: np.full(params.total_shares + 1, 0.0))

    def _choose(self, state, legal: list[int]) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        q = self.Q[state].copy()
        mask = np.full_like(q, -np.inf)
        mask[legal] = q[legal]
        return int(np.argmax(mask))

    def train(self, n_episodes: int = 20000, seed: int = 0) -> list[float]:
        env = ExecutionEnv(self.params, seed=seed)
        history: list[float] = []
        for ep in range(n_episodes):
            env.rng = np.random.default_rng(seed + ep)  # fresh price path each episode
            state = env.reset()
            done = False
            ep_cost = 0.0
            while not done:
                legal = env.legal_actions()
                action = self._choose(state, legal)
                next_state, reward, done = env.step(action)
                ep_cost += -reward

                # TD update. The terminal state's bootstrap value is 0.
                next_legal = env.legal_actions() if not done else []
                best_next = max((self.Q[next_state][a] for a in next_legal), default=0.0)
                td_target = reward + self.gamma * best_next
                self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])
                state = next_state

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            history.append(ep_cost)
        return history

    def greedy_schedule(self) -> list[int]:
        """The learned policy rolled out deterministically (no exploration, no price noise),
        returned as the units sold per step — i.e. the execution schedule the agent prefers."""
        env = ExecutionEnv(self.params, seed=0)
        env.rng = np.random.default_rng(0)
        state = env.reset()
        schedule, done = [], False
        while not done:
            legal = env.legal_actions()
            q = self.Q[state].copy()
            mask = np.full_like(q, -np.inf)
            mask[legal] = q[legal]
            action = int(np.argmax(mask))
            schedule.append(action)
            state, _, done = env.step(action)
        return schedule

    def evaluate(self, n_episodes: int = 2000, seed: int = 7) -> float:
        """Mean realised cost of the greedy policy over fresh random price paths."""
        env = ExecutionEnv(self.params, seed=seed)
        total = 0.0
        for ep in range(n_episodes):
            env.rng = np.random.default_rng(seed + ep)
            state = env.reset()
            done = False
            ep_cost = 0.0
            while not done:
                legal = env.legal_actions()
                q = self.Q[state].copy()
                mask = np.full_like(q, -np.inf)
                mask[legal] = q[legal]
                state, r, done = env.step(int(np.argmax(mask)))
                ep_cost += -r
            total += ep_cost
        return total / n_episodes
