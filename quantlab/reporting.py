"""Plotting helpers for the research report.

Nothing fancy — matplotlib with a restrained style. The figures exist to make the results
legible at a glance (equity curves, the underwater plot, a vol smile), which is how I'd
present this work in a research note rather than a wall of numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, never try to open a window
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "figure.figsize": (9, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_curves(curves: dict[str, pd.Series], path: Path, title: str) -> Path:
    """Overlay one or more equity curves, each rebased to 1.0 so they're comparable."""
    fig, ax = plt.subplots()
    for label, curve in curves.items():
        rebased = curve / curve.iloc[0]
        ax.plot(rebased.index, rebased.values, label=label, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    return _save(fig, path)


def plot_underwater(equity: pd.Series, path: Path, title: str = "Drawdown") -> Path:
    """The underwater curve — peak-to-trough drawdown through time. I find it tells you more
    about whether a strategy is *livable* than any single max-drawdown number."""
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots()
    ax.fill_between(dd.index, dd.values, 0.0, color="firebrick", alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    return _save(fig, path)


def plot_conditional_vol(realised: pd.Series, garch_vol: pd.Series, path: Path) -> Path:
    """Realised vs GARCH conditional volatility — shows the model tracking vol clustering."""
    fig, ax = plt.subplots()
    ax.plot(realised.index, realised.values, color="grey", alpha=0.6, label="realised (21d)")
    ax.plot(garch_vol.index, garch_vol.values, color="navy", linewidth=1.2, label="GARCH(1,1)")
    ax.set_title("Volatility: realised vs GARCH conditional")
    ax.set_ylabel("Annualised vol")
    ax.legend()
    return _save(fig, path)


def plot_vol_smile(surface: pd.DataFrame, path: Path) -> Path:
    """Implied vol vs moneyness, one line per expiry — the smile/skew."""
    fig, ax = plt.subplots()
    for expiry, grp in surface.groupby("expiry"):
        g = grp.sort_values("moneyness")
        ax.plot(g["moneyness"], g["implied_vol"], marker="o", markersize=3, label=str(expiry.date()))
    ax.axvline(1.0, color="black", linestyle=":", alpha=0.5)
    ax.set_title("Implied volatility smile")
    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Implied vol")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_ic_by_fold(fold_ic: list[float], path: Path) -> Path:
    """Information coefficient per walk-forward fold — stability matters as much as level."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(fold_ic) + 1), fold_ic, color="seagreen", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cross-sectional IC by walk-forward fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Spearman IC")
    return _save(fig, path)
