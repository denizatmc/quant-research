"""Forecasting forward market returns from a macro feature set.

A deliberately modest, interpretable model: take the macro panel, turn levels into the
*changes* that actually carry information (a 4% 10-year yield isn't a signal; a yield that
jumped 30bp this week is), and regress forward market returns on them with a regularised
linear model. Logistic regression for the *direction* of the move sits alongside it, because
on weekly horizons sign is more forecastable than magnitude.

The honest expectation: macro predicts risk (volatility) far better than it predicts return.
So the headline result to read off this isn't a stellar return R² — it's whether the
direction model beats a coin flip out-of-sample, evaluated walk-forward so there's no
peeking. Usually it's a small, real edge, and saying so plainly is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

from quantlab.models.validation import walk_forward_splits


def build_macro_features(macro: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Turn the raw macro panel into stationary predictors.

    Level series (yields, VIX) -> their `lookback`-day change. Price-like series (HYG, TLT,
    GLD, UUP) -> their `lookback`-day return. Differencing is what makes these usable as
    regression inputs; feeding raw, trending levels into a linear model is a classic way to
    manufacture a spurious fit.
    """
    feats = pd.DataFrame(index=macro.index)
    level_cols = [c for c in ["vix", "tnx_10y", "irx_13w", "curve_slope"] if c in macro.columns]
    ret_cols = [c for c in ["hyg", "tlt", "gld", "uup"] if c in macro.columns]

    for c in level_cols:
        feats[f"{c}_chg"] = macro[c].diff(lookback)
    for c in ret_cols:
        feats[f"{c}_ret"] = macro[c].pct_change(lookback)
    if "vix" in macro.columns:
        feats["vix_level"] = macro["vix"]  # the level itself is a risk-regime marker
    return feats


@dataclass
class MacroForecastResult:
    return_r2_oos: float          # out-of-sample R² of the return regression (expect ~0)
    direction_accuracy: float     # out-of-sample hit rate of the sign model
    n_test: int
    coefficients: pd.Series


def evaluate_macro_forecast(
    macro_features: pd.DataFrame,
    market_prices: pd.Series,
    horizon: int = 5,
    n_splits: int = 5,
) -> MacroForecastResult:
    """Walk-forward evaluation of the return and direction models.

    Both models are refit on each expanding training window and scored on the next block,
    with a purge gap equal to the forecast horizon so overlapping forward returns can't leak
    across the split. We report OOS return R² (almost always tiny — that's the truth about
    return predictability) and direction accuracy vs the 50% baseline.
    """
    fwd_ret = (market_prices.shift(-horizon) / market_prices - 1.0).rename("fwd")
    df = macro_features.join(fwd_ret, how="inner").dropna()
    X = df.drop(columns=["fwd"])
    y = df["fwd"]
    y_dir = (y > 0).astype(int)

    r2s, accs = [], []
    coef_accum = np.zeros(X.shape[1])
    n_fit = 0
    for split in walk_forward_splits(len(df), n_splits=n_splits, purge=horizon):
        tr, te = split.train_idx, split.test_idx
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, y.iloc[tr])
        pred = ridge.predict(Xte)
        ss_res = np.sum((y.iloc[te].values - pred) ** 2)
        ss_tot = np.sum((y.iloc[te].values - y.iloc[tr].mean()) ** 2)
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        coef_accum += ridge.coef_
        n_fit += 1

        # Direction model — only fit if both classes are present in the training block.
        if y_dir.iloc[tr].nunique() == 2:
            clf = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, y_dir.iloc[tr])
            accs.append(float((clf.predict(Xte) == y_dir.iloc[te].values).mean()))

    return MacroForecastResult(
        return_r2_oos=float(np.nanmean(r2s)) if r2s else float("nan"),
        direction_accuracy=float(np.nanmean(accs)) if accs else float("nan"),
        n_test=int(len(df) // (n_splits + 1)),
        coefficients=pd.Series(coef_accum / max(n_fit, 1), index=X.columns).sort_values(key=np.abs, ascending=False),
    )
