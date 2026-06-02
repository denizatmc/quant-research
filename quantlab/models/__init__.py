"""Modelling layer: classical time-series, ML, and a deep forecaster.

Kept separate from `strategies` on purpose — a model produces a forecast or a statistical
verdict; a strategy decides what to do about it. The deep-learning piece imports torch
lazily so the rest of the package works without it installed.
"""

from quantlab.models.timeseries import (
    adf_test,
    engle_granger_cointegration,
    fit_garch,
    half_life,
)

__all__ = ["adf_test", "engle_granger_cointegration", "half_life", "fit_garch"]
