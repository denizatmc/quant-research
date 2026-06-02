"""quantlab — a compact research toolkit for systematic trading.

The package is organised by research concern rather than by asset class:

    data            ingestion + a SQL storage layer (SQLite/Postgres)
    features        statistical and technical feature construction
    models          time-series (ARIMA/GARCH), ML, and a torch forecaster
    backtest        an event-driven backtester with an explicit cost model
    strategies      concrete alphas (stat-arb, momentum, ML)
    options         Black-Scholes pricing, Greeks, implied-vol surface
    microstructure  a synthetic limit order book + execution-quality analytics
    risk            VaR/ES, drawdown, and a live PnL/risk monitor
    execution       a minimal FIX 4.2 encoder/decoder

Nothing here is meant to be the fastest possible implementation — the goal is
to be correct, readable, and statistically honest.
"""

from quantlab.config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
