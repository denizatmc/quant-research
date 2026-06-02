"""Execution scheduling (TWAP / VWAP) and execution-quality measurement.

When you have to move a large parent order, *how* you slice it across the day is its own
optimisation: trade too fast and you pay market impact, too slow and you carry timing risk.
The two workhorse schedules are TWAP (spread evenly over time) and VWAP (trade in proportion
to expected volume, so you participate more when the market can absorb you).

The quality of a fill is then judged against benchmarks. The one that matters most is
*implementation shortfall* — the gap between the price you assumed when you decided to trade
(the arrival/decision price) and the price you actually achieved, including impact and any
adverse drift while you waited. It captures the full cost of execution, not just the visible
spread, which is why desks are measured on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def twap_schedule(n_slices: int) -> np.ndarray:
    """Equal weights — trade the same quantity in every interval. Simple and robust; the
    right default when you have no reliable volume forecast."""
    return np.full(n_slices, 1.0 / n_slices)


def vwap_schedule(expected_volume: np.ndarray) -> np.ndarray:
    """Weights proportional to expected per-interval volume.

    Following the volume curve means your participation rate stays roughly constant through
    the day, which is what keeps market impact even rather than concentrated. Intraday
    equity volume is famously U-shaped — heavy at the open and into the close — so VWAP
    front- and back-loads relative to TWAP.
    """
    total = expected_volume.sum()
    if total <= 0:
        return twap_schedule(len(expected_volume))
    return expected_volume / total


def synthetic_session(n_slices: int = 13, base_price: float = 100.0, drift_bps: float = 5.0,
                      vol_bps: float = 40.0, seed: int = 0) -> pd.DataFrame:
    """A toy intraday session: a U-shaped volume profile and a noisy price path.

    Stands in for real tick/bar data so the execution analytics are runnable out of the box.
    The U-shape is a parabola in interval index; the price is a small drift plus Gaussian
    noise. Deterministic given the seed so results reproduce.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n_slices)
    volume = (0.6 + x**2) * 1e5                      # heavy at both ends, light midday
    steps = rng.normal(drift_bps / 1e4 / n_slices, vol_bps / 1e4 / np.sqrt(n_slices), n_slices)
    price = base_price * np.cumprod(1 + steps)
    return pd.DataFrame({"interval": np.arange(n_slices), "price": price, "volume": volume})


@dataclass
class ExecutionReport:
    algo: str
    avg_exec_price: float
    arrival_price: float
    interval_vwap: float
    shortfall_bps: float       # vs arrival price — the headline number
    vs_vwap_bps: float         # vs the day's VWAP — the standard agency benchmark
    participation: float       # our share of total volume over the window

    def __str__(self) -> str:
        return (
            f"{self.algo:5s} | avg={self.avg_exec_price:.4f} arrival={self.arrival_price:.4f} "
            f"IS={self.shortfall_bps:+.1f}bp vs-VWAP={self.vs_vwap_bps:+.1f}bp "
            f"part={self.participation:.1%}"
        )


def simulate_execution(
    session: pd.DataFrame,
    parent_qty: float,
    weights: np.ndarray,
    side: str = "BUY",
    impact_coef: float = 0.1,
    algo: str = "ALGO",
) -> ExecutionReport:
    """Execute a sliced parent order against an intraday session and score the fills.

    Each slice trades `weights[i] * parent_qty` in interval i, paying that interval's price
    plus a square-root impact penalty in its own participation rate (consistent with the
    backtester's cost model). We then compare the size-weighted average fill price to the
    arrival price (implementation shortfall) and to interval VWAP.
    """
    px = session["price"].to_numpy()
    vol = session["volume"].to_numpy()
    sign = 1.0 if side.upper() == "BUY" else -1.0

    child_qty = weights * parent_qty
    part_rate = np.divide(child_qty, vol, out=np.zeros_like(child_qty), where=vol > 0)
    impact = impact_coef * np.sqrt(np.maximum(part_rate, 0.0))
    exec_px = px * (1.0 + sign * impact)             # impact always works against us

    filled = child_qty.sum()
    avg_exec = float((exec_px * child_qty).sum() / filled) if filled > 0 else float("nan")
    arrival = float(px[0])                            # decision price = price when we started
    interval_vwap = float((px * vol).sum() / vol.sum())

    # Shortfall is signed so positive always means "worse than benchmark", for either side.
    shortfall_bps = sign * (avg_exec - arrival) / arrival * 1e4
    vs_vwap_bps = sign * (avg_exec - interval_vwap) / interval_vwap * 1e4
    participation = float(filled / vol.sum()) if vol.sum() > 0 else float("nan")

    return ExecutionReport(
        algo=algo,
        avg_exec_price=avg_exec,
        arrival_price=arrival,
        interval_vwap=interval_vwap,
        shortfall_bps=float(shortfall_bps),
        vs_vwap_bps=float(vs_vwap_bps),
        participation=participation,
    )
