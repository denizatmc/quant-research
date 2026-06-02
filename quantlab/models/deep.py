"""A small LSTM sequence model in PyTorch.

The task here is deliberately one where a recurrent net has a fair shot: forecasting
next-day *realised volatility* from a window of past returns and squared returns. Volatility
has the memory and nonlinearity (clustering, asymmetry) that an LSTM can actually exploit,
whereas next-day *return* prediction is mostly noise and a fancy model just overfits it.

Scope note: this is a faithful, runnable PyTorch implementation — model, training loop with
early stopping, and a walk-forward style train/test split — kept small enough to train on a
CPU in seconds. It demonstrates the framework and the discipline (scaling fit on train only,
no shuffling across the time boundary) rather than chasing a benchmark.

torch is imported lazily and guarded, so importing this module never breaks an environment
that didn't install the `deep` extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for quantlab.models.deep. Install it with "
            "`pip install -e \".[deep]\"` or `pip install torch`."
        )


def make_sequences(
    features: np.ndarray, target: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a (T, F) feature matrix into overlapping (window, F) sequences.

    Sequence i uses features [i, i+window) to predict target[i+window] — so every label
    is strictly in the future of its inputs. No shuffling here; order is the signal.
    """
    xs, ys = [], []
    for i in range(len(features) - window):
        xs.append(features[i : i + window])
        ys.append(target[i + window])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


if _TORCH_AVAILABLE:

    class LSTMForecaster(nn.Module):
        """One LSTM layer + a linear head. Dropout for a little regularisation.

        Small by design: with a few thousand daily observations, a bigger network would
        memorise noise long before it learned anything that generalises."""

        def __init__(self, n_features: int, hidden: int = 32, num_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            # Use the last timestep's hidden state — the summary of the whole window.
            return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class TrainReport:
    train_losses: list[float]
    val_loss: float
    test_rmse: float
    test_corr: float


def train_volatility_lstm(
    returns: pd.Series,
    window: int = 21,
    vol_window: int = 5,
    hidden: int = 32,
    epochs: int = 60,
    lr: float = 1e-3,
    patience: int = 8,
    test_frac: float = 0.2,
    seed: int = 42,
) -> TrainReport:
    """Train the LSTM to forecast next-day realised vol from a window of return features.

    Features per day: the return and its square (a cheap stand-in for instantaneous
    variance). Target: forward `vol_window`-day realised vol. The split is chronological —
    the last `test_frac` of the sample is the untouched test set — and the feature scaler
    is fit on the training portion only, because peeking at test-set moments is a subtle
    but real form of leakage.
    """
    _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    r = returns.dropna()
    # Target: forward realised vol (annualised), shifted so it's genuinely ahead.
    fwd_vol = r.rolling(vol_window).std().shift(-vol_window) * np.sqrt(252)
    feat = np.column_stack([r.values, r.values ** 2])
    tgt = fwd_vol.values

    valid = np.isfinite(tgt)
    feat, tgt = feat[valid], tgt[valid]

    X, y = make_sequences(feat, tgt, window)
    n_test = int(len(X) * test_frac)
    n_val = int(len(X) * 0.15)
    X_train, y_train = X[: -n_test - n_val], y[: -n_test - n_val]
    X_val, y_val = X[-n_test - n_val : -n_test], y[-n_test - n_val : -n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    # Standardise features using train-set statistics only.
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-9
    scale = lambda a: (a - mu) / sd  # noqa: E731 - tiny local closure reads fine here

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr, Xva, Xte = to_t(scale(X_train)), to_t(scale(X_val)), to_t(scale(X_test))
    ytr, yva, yte = to_t(y_train), to_t(y_val), to_t(y_test)

    model = LSTMForecaster(n_features=X.shape[-1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0
    train_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xva), yva).item())
        # Early stopping on the validation block — the standard guard against overfitting.
        if vloss < best_val - 1e-6:
            best_val, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy()
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    corr = float(np.corrcoef(pred, y_test)[0, 1]) if len(y_test) > 1 else float("nan")
    return TrainReport(train_losses=train_losses, val_loss=best_val, test_rmse=rmse, test_corr=corr)
