"""A Polars implementation of the core features.

Why have two implementations? Two reasons, both honest:

  1. The JD asks for Polars, and the right way to show you know it is to actually use its
     idioms — lazy frames, window expressions with `.over()`, and the expression API —
     not to wrap a pandas call.
  2. On a daily, 16-name universe pandas is plenty. But the same code path scales to
     intraday/tick data where Polars' multithreaded, out-of-core engine genuinely matters,
     so this is the version I'd reach for when the data outgrows memory.

This computes a representative subset (multi-horizon momentum, realised vol, a z-score,
and a volume ratio) to demonstrate the approach without duplicating every last feature.
"""

from __future__ import annotations

import polars as pl


def build_features_polars(
    long_prices: pl.DataFrame | "object",
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252),
    vol_window: int = 21,
    zscore_window: int = 21,
) -> pl.DataFrame:
    """Build features with Polars window expressions.

    Accepts a Polars DataFrame or anything Polars can ingest (e.g. a pandas frame) with
    columns: symbol, date, adj_close, volume. All rolling stats are partitioned per symbol
    via `.over("symbol")`, which is the Polars equivalent of a pandas groupby-rolling but
    expressed declaratively and run in parallel across groups.
    """
    if not isinstance(long_prices, pl.DataFrame):
        long_prices = pl.DataFrame(long_prices)

    lf = (
        long_prices.lazy()
        .sort(["symbol", "date"])
        .with_columns(pl.col("adj_close").log().alias("log_px"))
    )

    # Per-horizon momentum = log_px - log_px shifted by the window, within each symbol.
    momentum_exprs = [
        (pl.col("log_px") - pl.col("log_px").shift(w).over("symbol")).alias(f"mom_{w}")
        for w in momentum_windows
    ]

    lf = lf.with_columns(
        # daily log return, per symbol
        (pl.col("log_px") - pl.col("log_px").shift(1).over("symbol")).alias("ret_1d"),
        *momentum_exprs,
    )

    lf = lf.with_columns(
        # annualised realised vol over the short window
        (
            pl.col("ret_1d").rolling_std(vol_window).over("symbol") * (252 ** 0.5)
        ).alias(f"vol_{vol_window}"),
        # mean-reversion z-score on price
        (
            (pl.col("adj_close") - pl.col("adj_close").rolling_mean(zscore_window).over("symbol"))
            / pl.col("adj_close").rolling_std(zscore_window).over("symbol")
        ).alias(f"zscore_{zscore_window}"),
        # log volume ratio vs its own rolling mean
        (
            pl.col("volume") / pl.col("volume").rolling_mean(zscore_window).over("symbol")
        ).log().alias(f"vol_ratio_{zscore_window}"),
    )

    return lf.drop("log_px").collect().drop_nulls()
