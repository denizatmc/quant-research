"""Macro / alternative-data ingestion.

The role asks for forecasting with macroeconomic and alternative data, so this pulls a small
panel of macro *market* indicators — all available from yfinance without a key, which keeps
the project reproducible. They're observable in real time (unlike, say, FRED releases that
arrive with a lag and get revised), so a model built on them isn't quietly cheating on
timing.

  ^VIX   implied vol of the S&P 500 — the market's fear gauge / forward risk appetite
  ^TNX   10-year Treasury yield — the level of the curve / discount rate
  ^IRX   13-week T-bill yield — the short end
  HYG    high-yield credit ETF — a clean credit-spread / risk-on proxy
  TLT    long Treasury ETF — duration / flight-to-quality
  GLD    gold — real rates / haven demand
  UUP    US dollar index ETF — global liquidity / risk sentiment

The yield-curve slope (10y minus short) is the single most-watched recession signal, so we
derive it here too.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

MACRO_TICKERS = {
    "^VIX": "vix",
    "^TNX": "tnx_10y",
    "^IRX": "irx_13w",
    "HYG": "hyg",
    "TLT": "tlt",
    "GLD": "gld",
    "UUP": "uup",
}


def fetch_macro(start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Download the macro panel and return a wide, forward-filled daily frame.

    Forward-fill (not interpolate) because on a non-trading day for one series the right
    assumption is "the last print still stands", and interpolation would invent information
    that wasn't observable at the time.
    """
    cols = {}
    for ticker, name in MACRO_TICKERS.items():
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty:
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # MultiIndex guard
            close = close.iloc[:, 0]
        cols[name] = close
    panel = pd.DataFrame(cols).sort_index().ffill().dropna(how="all")
    # Yield-curve slope: a derived alpha/macro feature in its own right.
    if {"tnx_10y", "irx_13w"}.issubset(panel.columns):
        panel["curve_slope"] = panel["tnx_10y"] - panel["irx_13w"]
    return panel
