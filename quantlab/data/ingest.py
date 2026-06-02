"""Pull daily OHLCV from yfinance and land it in the SQL store.

yfinance is good enough for a daily-frequency research project and, crucially, needs no
API key — anyone can reproduce the dataset. The cost is the occasional schema quirk
(MultiIndex columns, the odd missing day), which this module normalises away so the rest
of the codebase only ever sees a clean, tidy frame.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import yfinance as yf

from quantlab.config import Config, load_config
from quantlab.data.database import MarketDB


def _normalise_one(symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a single-symbol yfinance frame into our tidy long schema.

    yfinance has changed its column layout a few times; handling both the flat and the
    MultiIndex cases here means an upstream library update doesn't quietly break ingestion.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Collapse the (field, ticker) MultiIndex down to just the field.
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    # Some yfinance versions already return an adjusted "Close" and drop "Adj Close".
    # If so, treat close as the adjusted series rather than inventing data.
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
    df["symbol"] = symbol
    keep = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def ingest_universe(config: Config | None = None, verbose: bool = True) -> int:
    """Download the configured universe and upsert it into the store.

    Returns the number of rows written. Safe to re-run: the upsert is idempotent, so a
    daily cron that re-pulls the trailing window just refreshes adjustments.
    """
    cfg = config or load_config()
    db = MarketDB.from_config(cfg)
    db.create_schema()

    symbols = cfg.universe
    start = cfg.get("data.start", "2015-01-01")
    end = cfg.get("data.end")  # None -> today

    total = 0
    for sym in symbols:
        # One symbol at a time. Slower than a batch call, but the per-symbol frames are
        # unambiguous and a single failed ticker can't poison the whole pull.
        raw = yf.download(
            sym, start=start, end=end, auto_adjust=False, progress=False, threads=False
        )
        tidy = _normalise_one(sym, raw)
        n = db.upsert_prices(tidy)
        total += n
        if verbose:
            span = f"{tidy['date'].min()} → {tidy['date'].max()}" if n else "no data"
            print(f"  {sym:6s} {n:5d} rows  [{span}]")
        time.sleep(0.2)  # be polite to the endpoint; avoids the occasional rate-limit

    if verbose:
        print(f"Done. {total} rows across {len(symbols)} symbols into {db.dialect}.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the configured universe into the SQL store.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to repo config).")
    args = parser.parse_args()
    ingest_universe(load_config(args.config))


if __name__ == "__main__":
    main()
