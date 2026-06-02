"""A streaming PnL / risk monitor.

This is the small piece of "real-time risk infrastructure" the role asks about, modelled as
a stateful object you feed mark updates to, one timestamp at a time — exactly how it would
sit in a live loop consuming a market-data feed. On each update it re-marks the book,
attributes PnL, recomputes exposures, and checks them against a set of hard limits, emitting
a breach the instant one trips.

The limits encode the kind of pre-trade / intraday controls a desk runs under (gross and net
leverage caps, per-name concentration, a max-drawdown kill-switch). In a regulated setting
(SPK in Turkey, or the SEC's Market Access Rule 15c3-5 in the US) these aren't optional —
they're the automated controls that have to sit between a strategy and the exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RiskLimits:
    max_gross_leverage: float = 2.0     # sum(|position value|) / equity
    max_net_leverage: float = 1.0       # |sum(position value)| / equity
    max_position_weight: float = 0.10   # any single name as a fraction of equity
    max_drawdown: float = 0.20          # kill-switch on peak-to-trough equity decline


@dataclass
class Breach:
    timestamp: object
    kind: str
    value: float
    limit: float

    def __str__(self) -> str:
        return f"[{self.timestamp}] BREACH {self.kind}: {self.value:.3f} > {self.limit:.3f}"


@dataclass
class RiskMonitor:
    initial_equity: float
    limits: RiskLimits = field(default_factory=RiskLimits)

    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    positions: dict[str, float] = field(init=False)   # symbol -> shares
    last_prices: dict[str, float] = field(init=False)
    history: list[dict] = field(init=False)
    breaches: list[Breach] = field(init=False)

    def __post_init__(self):
        self.equity = float(self.initial_equity)
        self.peak_equity = float(self.initial_equity)
        self.positions = {}
        self.last_prices = {}
        self.history = []
        self.breaches = []

    def update(self, timestamp, prices: dict[str, float], positions: dict[str, float] | None = None) -> list[Breach]:
        """Process one mark. Optionally accept a new position vector (e.g. after a fill),
        re-mark the book, snapshot risk, and return any limits breached on this tick."""
        if positions is not None:
            self.positions = dict(positions)
        self.last_prices.update(prices)

        # Mark to market: cash isn't tracked here (the backtester owns that); this monitor
        # works in exposures and PnL relative to the starting equity, which is what a risk
        # desk actually watches in real time.
        position_values = {
            s: self.positions.get(s, 0.0) * self.last_prices.get(s, 0.0)
            for s in self.positions
        }
        gross = sum(abs(v) for v in position_values.values())
        net = sum(position_values.values())
        # PnL since inception is the change in marked book value of positions.
        self.equity = self.initial_equity + net  # net exposure stands in for cumulative PnL on a fully-invested book
        self.peak_equity = max(self.peak_equity, self.equity)

        gross_lev = gross / self.equity if self.equity > 0 else float("inf")
        net_lev = abs(net) / self.equity if self.equity > 0 else float("inf")
        drawdown = self.equity / self.peak_equity - 1.0
        max_weight = (
            max((abs(v) / self.equity for v in position_values.values()), default=0.0)
            if self.equity > 0
            else float("inf")
        )

        tick_breaches: list[Breach] = []

        def check(kind, value, limit):
            if value > limit:
                b = Breach(timestamp, kind, value, limit)
                tick_breaches.append(b)
                self.breaches.append(b)

        check("gross_leverage", gross_lev, self.limits.max_gross_leverage)
        check("net_leverage", net_lev, self.limits.max_net_leverage)
        check("position_concentration", max_weight, self.limits.max_position_weight)
        check("drawdown", -drawdown, self.limits.max_drawdown)

        self.history.append(
            {
                "timestamp": timestamp,
                "equity": self.equity,
                "gross_leverage": gross_lev,
                "net_leverage": net_lev,
                "drawdown": drawdown,
                "max_weight": max_weight,
                "n_breaches": len(tick_breaches),
            }
        )
        return tick_breaches

    def report(self) -> pd.DataFrame:
        """The time series of risk snapshots — the data behind a live dashboard."""
        return pd.DataFrame(self.history).set_index("timestamp")
