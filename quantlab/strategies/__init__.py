"""Concrete trading strategies built on the backtest framework."""

from quantlab.strategies.momentum import CrossSectionalMomentum
from quantlab.strategies.stat_arb import PairsTradingStrategy, KalmanPairsStrategy

__all__ = ["CrossSectionalMomentum", "PairsTradingStrategy", "KalmanPairsStrategy"]
