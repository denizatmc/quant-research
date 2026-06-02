"""Build an implied-volatility surface from a live option chain.

This ties the pricer to real market data: pull a chain from yfinance, invert every quote to
an implied vol, and lay the result out as a (strike/moneyness × expiry) surface. The two
things worth seeing fall straight out of it — the volatility *smile* across strikes (deep
OTM options trade richer than Black-Scholes flat-vol would say, because the market prices in
fat tails) and the *term structure* across expiries.

Network-dependent by nature, so it's isolated here and the rest of the package never imports
it implicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.options.black_scholes import OptionType, implied_volatility


def fetch_chain(ticker: str, max_expiries: int = 4) -> pd.DataFrame:
    """Pull calls+puts for the nearest few expiries and return a tidy quote frame.

    We take the mid of bid/ask where available (last-traded prices can be stale on illiquid
    strikes), and carry spot + days-to-expiry through so the IV inversion has everything it
    needs downstream.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    rows = []
    for expiry in tk.options[:max_expiries]:
        chain = tk.option_chain(expiry)
        for opt_type, table in [(OptionType.CALL, chain.calls), (OptionType.PUT, chain.puts)]:
            df = table.copy()
            mid = np.where(
                (df["bid"] > 0) & (df["ask"] > 0),
                0.5 * (df["bid"] + df["ask"]),
                df["lastPrice"],
            )
            rows.append(
                pd.DataFrame(
                    {
                        "expiry": pd.Timestamp(expiry),
                        "type": opt_type.value,
                        "strike": df["strike"].values,
                        "mid": mid,
                        "spot": spot,
                    }
                )
            )
    out = pd.concat(rows, ignore_index=True)
    out = out[out["mid"] > 0].reset_index(drop=True)
    return out


def build_iv_surface(chain: pd.DataFrame, r: float = 0.04, q: float = 0.0) -> pd.DataFrame:
    """Invert each quote to an implied vol and add moneyness / time-to-expiry columns.

    To keep the surface clean we use OTM options on each side (calls above spot, puts below)
    — they're the more liquid, less intrinsic-dominated quotes, so their IVs are the more
    informative ones. The result is ready to pivot into a strike×expiry grid or plot as a
    smile per expiry.
    """
    now = chain["expiry"].min().normalize()
    df = chain.copy()
    # Use the earliest expiry's date as "today" proxy if a real clock isn't passed in.
    df["T"] = (df["expiry"] - now).dt.days.clip(lower=1) / 365.0
    df["moneyness"] = df["strike"] / df["spot"]

    otm = df[
        ((df["type"] == "call") & (df["strike"] >= df["spot"]))
        | ((df["type"] == "put") & (df["strike"] <= df["spot"]))
    ].copy()

    ivs = []
    for row in otm.itertuples():
        ot = OptionType.CALL if row.type == "call" else OptionType.PUT
        ivs.append(implied_volatility(row.mid, row.spot, row.strike, row.T, r, q, ot))
    otm["implied_vol"] = ivs
    return otm.dropna(subset=["implied_vol"]).sort_values(["expiry", "strike"]).reset_index(drop=True)
