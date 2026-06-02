"""Technical / statistical features, computed per symbol on a tidy long frame.

These are deliberately standard, well-understood predictors — momentum, volatility,
mean-reversion z-scores, RSI, volume pressure. The point of the project isn't to invent
exotic features; it's to construct them without lookahead, stack them into a clean panel,
and let the modelling and validation layers decide what (if anything) actually predicts.

The single most important discipline here: every feature at time t uses information
available *at or before* t. Returns are shifted appropriately when they become targets,
not here, but rolling windows use only past data by construction (pandas `.rolling` is
trailing), and nothing peeks forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Window choices, all in trading days. Defaults are conventional rather than tuned —
    tuning windows on the same data you evaluate on is how you fool yourself."""

    momentum_windows: tuple[int, ...] = (21, 63, 126, 252)  # ~1m, 3m, 6m, 12m
    vol_window: int = 21
    zscore_window: int = 21
    rsi_window: int = 14
    volume_window: int = 21
    feature_cols: list[str] = field(default_factory=list)  # populated by build_*


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI. Bounded 0-100; ~50 is neutral. Included partly because it's a
    nonlinear, bounded feature, which is a useful contrast to the unbounded z-scores."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _features_for_symbol(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute the feature set for one symbol. `df` is sorted by date with adj_close/volume."""
    out = pd.DataFrame(index=df.index)
    px = df["adj_close"]
    log_px = np.log(px)
    daily_logret = log_px.diff()

    # Momentum: cumulative log-return over each lookback. Log-space so multi-horizon
    # momentum is just a difference of log prices — clean and numerically stable.
    for w in cfg.momentum_windows:
        out[f"mom_{w}"] = log_px - log_px.shift(w)

    # Realised volatility (annualised) over the short window.
    out[f"vol_{cfg.vol_window}"] = daily_logret.rolling(cfg.vol_window).std() * np.sqrt(252)

    # Mean-reversion z-score: how many rolling-sigmas is price from its rolling mean.
    roll_mean = px.rolling(cfg.zscore_window).mean()
    roll_std = px.rolling(cfg.zscore_window).std()
    out[f"zscore_{cfg.zscore_window}"] = (px - roll_mean) / roll_std

    out[f"rsi_{cfg.rsi_window}"] = _rsi(px, cfg.rsi_window)

    # Volume pressure: today's volume vs its recent average (a crude liquidity/attention
    # proxy). Logged because volume is heavy-tailed.
    vol_ma = df["volume"].rolling(cfg.volume_window).mean()
    out[f"vol_ratio_{cfg.volume_window}"] = np.log(df["volume"] / vol_ma)

    # 1-day return is a feature too (short-term reversal tends to live here).
    out["ret_1d"] = daily_logret
    return out


def build_feature_panel(long_prices: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Build a multi-index (date, symbol) feature panel from a tidy long price frame.

    Expects columns: symbol, date, adj_close, volume. Returns a panel indexed by
    (date, symbol) with one column per feature — the natural shape for cross-sectional
    ML (group by date) and for joining a forward-return target.
    """
    cfg = cfg or FeatureConfig()
    required = {"symbol", "date", "adj_close", "volume"}
    missing = required - set(long_prices.columns)
    if missing:
        raise ValueError(f"build_feature_panel missing columns: {sorted(missing)}")

    frames = []
    for sym, grp in long_prices.sort_values("date").groupby("symbol"):
        g = grp.set_index("date")
        feats = _features_for_symbol(g, cfg)
        feats["symbol"] = sym
        frames.append(feats)

    panel = pd.concat(frames).set_index("symbol", append=True)
    panel.index = panel.index.set_names(["date", "symbol"])
    cfg.feature_cols = [c for c in panel.columns]
    # Drop the warm-up rows where the longest window hasn't filled yet.
    return panel.dropna()
