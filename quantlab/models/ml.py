"""A cross-sectional ML alpha model with honest walk-forward evaluation.

The task is to predict, each day, the *cross-sectional ranking* of next-period returns from
the feature panel — i.e. which names will outperform their peers. This framing (relative,
not absolute) is what makes ML on noisy market data tractable: you're not asking the model
to predict the market, only to sort the universe.

The model itself is intentionally boring — gradient-boosted trees, lightly regularised.
The interesting, CV-worthy part is the evaluation: features are standardised cross-
sectionally (per day), the target is a forward return joined without lookahead, and scoring
is done by information coefficient on walk-forward folds with a purge gap matching the
label horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from quantlab.models.validation import information_coefficient, walk_forward_splits


def make_forward_target(prices_wide: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Forward `horizon`-day return per (date, symbol), the thing we're predicting.

    The shift(-horizon) is the only place lookahead is *intended* — this is the label, and
    it's aligned so that the features at date t are paired with the return realised over
    (t, t+horizon]. Rows without a full forward window simply have no label.
    """
    fwd = prices_wide.shift(-horizon) / prices_wide - 1.0
    target = fwd.stack()
    target.index = target.index.set_names(["date", "symbol"])
    return target.rename("fwd_return")


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Standardise each feature within each date (rank-relative across the universe).

    Doing this per day removes market-wide level shifts and puts every feature on a
    comparable scale, which is what lets a single model generalise across regimes.
    """
    return panel.groupby(level="date").transform(lambda c: (c - c.mean()) / (c.std() + 1e-9))


@dataclass
class CrossSectionalModelResult:
    fold_ic: list[float] = field(default_factory=list)
    feature_importance: pd.Series | None = None

    @property
    def mean_ic(self) -> float:
        return float(np.nanmean(self.fold_ic)) if self.fold_ic else float("nan")

    @property
    def ic_ir(self) -> float:
        """IC information ratio: mean IC / std IC across folds — stability matters as much
        as level. A high mean IC that swings wildly fold-to-fold isn't tradeable."""
        if len(self.fold_ic) < 2:
            return float("nan")
        return float(np.nanmean(self.fold_ic) / (np.nanstd(self.fold_ic) + 1e-9))


def evaluate_cross_sectional_model(
    features: pd.DataFrame,
    target: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    model=None,
) -> CrossSectionalModelResult:
    """Walk-forward IC evaluation of a cross-sectional return model.

    `features` is the (date, symbol) panel; `target` is the forward return. We align them,
    standardise features per day, then walk forward over *unique dates* (not rows) so a
    whole cross-section moves together between train and test — splitting mid-day would
    leak. The purge gap equals the label horizon so overlapping forward windows can't
    straddle the boundary.
    """
    model = model or GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42
    )

    df = features.join(target, how="inner").dropna()
    feat_cols = [c for c in features.columns]
    X_panel = _cross_sectional_zscore(df[feat_cols])
    y = df["fwd_return"]

    dates = df.index.get_level_values("date").unique().sort_values()
    result = CrossSectionalModelResult()
    importances = np.zeros(len(feat_cols))
    n_fit = 0

    for split in walk_forward_splits(len(dates), n_splits=n_splits, purge=horizon):
        train_dates = dates[split.train_idx]
        test_dates = dates[split.test_idx]
        tr = df.index.get_level_values("date").isin(train_dates)
        te = df.index.get_level_values("date").isin(test_dates)
        if tr.sum() == 0 or te.sum() == 0:
            continue

        model.fit(X_panel[tr].values, y[tr].values)
        preds = model.predict(X_panel[te].values)
        result.fold_ic.append(information_coefficient(preds, y[te].values))
        if hasattr(model, "feature_importances_"):
            importances += model.feature_importances_
            n_fit += 1

    if n_fit:
        result.feature_importance = pd.Series(
            importances / n_fit, index=feat_cols
        ).sort_values(ascending=False)
    return result


def walk_forward_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    model=None,
) -> pd.DataFrame:
    """Return the out-of-sample predictions across all walk-forward test folds.

    Same protocol as `evaluate_cross_sectional_model`, but instead of collapsing each fold
    to a single IC it keeps the per-(date, symbol) predictions paired with the realised
    forward return. That's what you need to go beyond a headline IC — to bucket names into
    predicted quantiles and check that a long-top/short-bottom portfolio actually monotonically
    separates, which is the real test of whether a signal is tradeable rather than just
    correlated.
    """
    model = model or GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42
    )
    df = features.join(target, how="inner").dropna()
    feat_cols = [c for c in features.columns]
    X_panel = _cross_sectional_zscore(df[feat_cols])
    y = df["fwd_return"]

    dates = df.index.get_level_values("date").unique().sort_values()
    out = []
    for split in walk_forward_splits(len(dates), n_splits=n_splits, purge=horizon):
        tr = df.index.get_level_values("date").isin(dates[split.train_idx])
        te = df.index.get_level_values("date").isin(dates[split.test_idx])
        if tr.sum() == 0 or te.sum() == 0:
            continue
        model.fit(X_panel[tr].values, y[tr].values)
        preds = pd.Series(model.predict(X_panel[te].values), index=df.index[te], name="prediction")
        fold = pd.concat([preds, y[te].rename("fwd_return")], axis=1)
        out.append(fold)

    return pd.concat(out) if out else pd.DataFrame(columns=["prediction", "fwd_return"])
