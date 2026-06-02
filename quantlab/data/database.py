"""SQL storage layer.

The brief asked for SQL/PostgreSQL competence, but a public repo should also run with
zero setup. The compromise: write everything through SQLAlchemy Core against a backend
chosen in config. SQLite is the default (a single file, nothing to install); Postgres is
one config flag away and is what I'd actually use in production for the concurrency and
the window functions.

I use SQLAlchemy Core rather than the ORM on purpose — for a market-data store we're doing
bulk upserts and analytical reads, not object graphs, and Core keeps the generated SQL
close to what I'd write by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    Float,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from quantlab.config import Config, load_config

_METADATA = MetaData()

# One wide-ish OHLCV table keyed by (symbol, date). Adjusted close is stored separately
# from raw close because total-return vs price-return matters and conflating them is a
# classic, silent source of backtest bugs.
prices = Table(
    "prices",
    _METADATA,
    Column("symbol", String(16), nullable=False),
    Column("date", Date, nullable=False),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adj_close", Float),
    Column("volume", Float),
    UniqueConstraint("symbol", "date", name="uq_prices_symbol_date"),
)


class MarketDB:
    """Thin wrapper around a SQLAlchemy engine for the market-data store.

    Usage:
        db = MarketDB.from_config()
        db.create_schema()
        db.upsert_prices(df)
        wide = db.load_prices(["AAPL", "MSFT"], field="adj_close")
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    # --- construction -----------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Config | None = None) -> "MarketDB":
        cfg = config or load_config()
        backend = cfg.get("database.backend", "sqlite")
        if backend == "sqlite":
            path = cfg.sqlite_path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path}"
        elif backend == "postgres":
            url = cfg.get("database.postgres_url")
        else:
            raise ValueError(f"Unknown database backend: {backend!r}")
        # future=True keeps us on the 2.0-style API; echo stays off to spare the logs.
        return cls(create_engine(url, future=True))

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    def create_schema(self) -> None:
        _METADATA.create_all(self.engine)

    # --- writes -----------------------------------------------------------------------
    def upsert_prices(self, df: pd.DataFrame) -> int:
        """Insert-or-update a tidy (long) price frame.

        Expected columns: symbol, date, open, high, low, close, adj_close, volume.
        Idempotent by construction — re-running ingestion never duplicates rows, which
        means I can re-pull the last few days to pick up adjustments without fear.
        """
        required = {"symbol", "date", "open", "high", "low", "close", "adj_close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"upsert_prices is missing columns: {sorted(missing)}")

        records = df[list(required)].to_dict(orient="records")
        if not records:
            return 0

        # The ON CONFLICT clause is dialect-specific, so branch on it. Both paths key on
        # the (symbol, date) unique constraint and refresh the OHLCV fields.
        update_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        with self.engine.begin() as conn:
            if self.dialect == "postgresql":
                stmt = pg_insert(prices).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "date"],
                    set_={c: stmt.excluded[c] for c in update_cols},
                )
            else:  # sqlite (and a sane default for anything else)
                stmt = sqlite_insert(prices).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "date"],
                    set_={c: getattr(stmt.excluded, c) for c in update_cols},
                )
            conn.execute(stmt)
        return len(records)

    # --- reads ------------------------------------------------------------------------
    def load_prices(
        self,
        symbols: list[str] | None = None,
        field: str = "adj_close",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return a wide price panel: index = date, columns = symbols.

        Wide form is what every downstream model wants (returns, covariance, signals),
        so we pivot here once rather than in a dozen call sites.
        """
        valid_fields = {"open", "high", "low", "close", "adj_close", "volume"}
        if field not in valid_fields:
            raise ValueError(f"field must be one of {sorted(valid_fields)}")

        col = prices.c[field]
        query = select(prices.c.symbol, prices.c.date, col.label("value"))
        if symbols:
            query = query.where(prices.c.symbol.in_(symbols))
        if start:
            query = query.where(prices.c.date >= pd.Timestamp(start).date())
        if end:
            query = query.where(prices.c.date <= pd.Timestamp(end).date())

        with self.engine.connect() as conn:
            long_df = pd.read_sql(query, conn, parse_dates=["date"])

        if long_df.empty:
            return pd.DataFrame()
        wide = long_df.pivot(index="date", columns="symbol", values="value").sort_index()
        wide.columns.name = None
        return wide

    def load_long(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return the tidy long frame (one row per symbol-date) the feature layer expects.

        Wide form is convenient for portfolio maths but awkward for per-symbol feature
        construction, so this gives callers the raw long table back.
        """
        query = select(prices)
        if symbols:
            query = query.where(prices.c.symbol.in_(symbols))
        if start:
            query = query.where(prices.c.date >= pd.Timestamp(start).date())
        if end:
            query = query.where(prices.c.date <= pd.Timestamp(end).date())
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, parse_dates=["date"])
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)

    def available_symbols(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(prices.c.symbol).distinct()).fetchall()
        return sorted(r[0] for r in rows)

    def run_sql(self, sql: str) -> pd.DataFrame:
        """Escape hatch for ad-hoc analytical SQL (window functions, etc.).

        Handy in notebooks when I want to demonstrate a rolling computation in SQL rather
        than pandas — e.g. a 20-day moving average via `AVG(...) OVER (...)`.
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
