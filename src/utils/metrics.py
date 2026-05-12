from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    return true, pred


def mae(y_true, y_pred) -> float:
    """Return mean absolute error."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def r2(y_true, y_pred) -> float:
    """Return coefficient of determination."""
    true, pred = _as_arrays(y_true, y_pred)
    denominator = np.sum((true - np.mean(true)) ** 2)
    if denominator == 0:
        return float("nan")
    numerator = np.sum((true - pred) ** 2)
    return float(1 - numerator / denominator)

