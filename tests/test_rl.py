"""Tests for the RL execution agent.

The properties I care about aren't 'the loss went down' — they're economic: a risk-averse
agent should (a) front-load its selling relative to a risk-neutral one, and (b) actually beat
TWAP when timing risk is what's being penalised. Both are checked here with a short training
run so the suite stays fast.
"""

from quantlab.rl import EnvParams, QLearner, twap_cost


def _train(risk_aversion: float):
    p = EnvParams(total_shares=8, n_steps=4, sigma=0.4, risk_aversion=risk_aversion)
    agent = QLearner(p, seed=0)
    agent.train(n_episodes=8000, seed=0)
    return p, agent


def test_risk_aversion_front_loads():
    # A risk-averse agent sells more in the first interval than a risk-neutral one.
    _, lo = _train(0.0)
    _, hi = _train(0.2)
    assert hi.greedy_schedule()[0] >= lo.greedy_schedule()[0]


def test_agent_beats_twap_when_risk_matters():
    # With a real risk penalty, the learned policy should cost less than naive TWAP.
    p_hi, hi = _train(0.2)
    assert hi.evaluate(n_episodes=1500) < twap_cost(p_hi, n_episodes=1500)


def test_schedule_liquidates_everything():
    # Whatever it learns, the agent must end flat — the order has to get done.
    p, agent = _train(0.05)
    assert sum(agent.greedy_schedule()) == p.total_shares
