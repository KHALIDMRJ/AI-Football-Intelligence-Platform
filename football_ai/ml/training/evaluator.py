"""
Model evaluation utilities.

Computes standard binary classification metrics used throughout the
VAEP literature and sports analytics:

- ROC-AUC
- Brier score (mean squared error of probabilities)
- Log loss
- Precision, Recall, F1 at 0.5 threshold
- Calibration summary (mean predicted vs mean actual)

All metrics are returned as a plain dict[str, float] for easy logging
and serialisation to the model registry metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from football_ai.logger import get_logger

logger = get_logger(__name__)


def evaluate_binary_classifier(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    model_name: str = "model",
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth binary labels (0 or 1).
    y_prob : np.ndarray, shape (n,)
        Predicted probabilities for the positive class.
    model_name : str
        Label used in log output.
    threshold : float
        Probability threshold for converting to hard predictions.

    Returns
    -------
    dict[str, float]
        Keys: roc_auc, brier_score, log_loss, precision, recall, f1,
              mean_pred_prob, mean_actual_prob, n_samples, n_positive.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true_arr)
    n_pos = int(y_true_arr.sum())

    if n_pos == 0 or n_pos == n:
        # Degenerate case — only one class present
        logger.warning(
            "[%s] Cannot compute ROC-AUC: only one class in y_true "
            "(n_pos=%d / n=%d). Returning partial metrics.",
            model_name, n_pos, n,
        )
        try:
            ll = float(log_loss(y_true_arr, y_prob))
        except ValueError:
            ll = float("nan")
        return {
            "roc_auc": float("nan"),
            "brier_score": float(brier_score_loss(y_true_arr, y_prob)),
            "log_loss": ll,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mean_pred_prob": float(y_prob.mean()),
            "mean_actual_prob": float(y_true_arr.mean()),
            "n_samples": n,
            "n_positive": n_pos,
        }

    metrics = {
        "roc_auc":          float(roc_auc_score(y_true_arr, y_prob)),
        "brier_score":      float(brier_score_loss(y_true_arr, y_prob)),
        "log_loss":         float(log_loss(y_true_arr, y_prob)),
        "precision":        float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "mean_pred_prob":   float(y_prob.mean()),
        "mean_actual_prob": float(y_true_arr.mean()),
        "n_samples":        float(n),
        "n_positive":       float(n_pos),
    }

    logger.info(
        "[%s] ROC-AUC=%.4f  Brier=%.4f  LogLoss=%.4f  "
        "Precision=%.4f  Recall=%.4f  F1=%.4f  "
        "MeanPred=%.4f  MeanActual=%.4f  N=%d  Npos=%d",
        model_name,
        metrics["roc_auc"],
        metrics["brier_score"],
        metrics["log_loss"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["mean_pred_prob"],
        metrics["mean_actual_prob"],
        n,
        n_pos,
    )
    return metrics


def evaluate_xg_model(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Evaluate the xG model — same metrics, named for clarity."""
    return evaluate_binary_classifier(y_true, y_prob, model_name="xG")


def evaluate_p_scores_model(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Evaluate the P_scores model."""
    return evaluate_binary_classifier(y_true, y_prob, model_name="P_scores")


def evaluate_p_concedes_model(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Evaluate the P_concedes model."""
    return evaluate_binary_classifier(y_true, y_prob, model_name="P_concedes")


def calibration_summary(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute a calibration table: predicted vs actual probability per bin.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    pd.DataFrame
        Columns: bin_low, bin_high, mean_pred, mean_actual, n_samples.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        rows.append({
            "bin_low":     round(float(lo), 3),
            "bin_high":    round(float(hi), 3),
            "mean_pred":   round(float(y_prob[mask].mean()), 4),
            "mean_actual": round(float(y_true_arr[mask].mean()), 4),
            "n_samples":   n_bin,
        })
    return pd.DataFrame(rows)
