"""Reinforcement learning for optimal trade execution.

A self-contained, tabular RL treatment of the optimal-liquidation problem: an agent learns
*how fast* to work a large order, trading off market impact (go slow) against timing risk
(go fast). The environment is the Almgren-Chriss world; the agent is plain Q-learning. No
deep nets and no GPU — the point is to show the problem framed correctly as an MDP and an
agent that recovers the known-optimal behaviour, which is far more convincing than a black
box that happens to train.
"""

from quantlab.rl.execution_env import ExecutionEnv, EnvParams
from quantlab.rl.q_learning import QLearner, twap_cost

__all__ = ["ExecutionEnv", "EnvParams", "QLearner", "twap_cost"]
