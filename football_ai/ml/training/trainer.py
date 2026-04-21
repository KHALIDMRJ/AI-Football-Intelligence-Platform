"""
Per-model training routines.

Each ``train_*`` function:
1. Prepares training data from the feature DataFrame
2. Splits into train / test
3. Fits the model
4. Evaluates on the test split
5. Saves to the registry
6. Returns the fitted model and metrics dict

These functions are called by the training pipeline
(``football_ai.ml.training.pipeline``) and the CLI script
(``scripts/train_models.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from football_ai.config import settings
from football_ai.logger import get_logger
from football_ai.ml.models.p_concedes_model import PConcedesModel
from football_ai.ml.models.p_scores_model import PScoresModel
from football_ai.ml.models.xg_model import XGModel
from football_ai.ml.models.xt_model import XTModel
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.ml.training.evaluator import (
    evaluate_p_concedes_model,
    evaluate_p_scores_model,
    evaluate_xg_model,
)
from football_ai.utils import timer

logger = get_logger(__name__)

# ── Shared split helper ────────────────────────────────────────────────────────

def _train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with fallback for single-class edge cases."""
    ts  = test_size    or settings.training.test_size
    rs  = random_state or settings.training.random_state
    n_pos = int(y.sum())
    stratify = y if n_pos > 1 and n_pos < len(y) - 1 else None

    return train_test_split(X, y, test_size=ts, random_state=rs, stratify=stratify)


# ── xG ────────────────────────────────────────────────────────────────────────

def train_xg(
    feature_df: pd.DataFrame,
    registry: ModelRegistry,
    C: float = 1.0,
    max_iter: int = 1000,
    calibrate: bool = True,
) -> tuple[XGModel, dict[str, float]]:
    """
    Train the xG logistic regression model on shot events.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Full feature matrix from FeatureAssembler.
    registry : ModelRegistry
        Where the trained model will be saved.
    C : float
        Logistic regression regularisation strength.
    max_iter : int
        Maximum iterations for the solver.
    calibrate : bool
        Wrap with CalibratedClassifierCV.

    Returns
    -------
    (XGModel, metrics) : tuple[XGModel, dict[str, float]]
    """
    with timer("xG prepare data"):
        X, y = XGModel.prepare_training_data(feature_df)

    logger.info("xG training set: %d shots (%d goals)", len(X), int(y.sum()))

    X_tr, X_te, y_tr, y_te = _train_test(X, y)

    model = XGModel(C=C, max_iter=max_iter, calibrate=calibrate)
    with timer("xG fit"):
        model.fit(X_tr, y_tr)

    with timer("xG evaluate"):
        y_prob = model.predict_proba(X_te)
        metrics = evaluate_xg_model(y_te.values, y_prob)

    model.save(registry)
    registry.save("xg", model, metrics=metrics, feature_cols=model.feature_cols)
    return model, metrics


# ── xT ────────────────────────────────────────────────────────────────────────

def train_xt(
    spadl_df: pd.DataFrame,
    registry: ModelRegistry,
    smoothing: float = 1.0,
) -> tuple[XTModel, dict[str, float]]:
    """
    Fit the xT Markov-chain model from the SPADL actions DataFrame.

    Note: xT is not a supervised classification model, so there is no
    train/test split.  We use the full dataset to estimate transition
    probabilities — more data = better estimates.

    Parameters
    ----------
    spadl_df : pd.DataFrame
        SPADL actions (output of PreprocessingPipeline, NOT the feature matrix).
    registry : ModelRegistry

    Returns
    -------
    (XTModel, metrics) : tuple[XTModel, dict[str, float]]
        metrics is a dict of descriptive statistics (not classification metrics).
    """
    model = XTModel(smoothing=smoothing)
    with timer("xT fit"):
        model.fit(spadl_df)

    # Descriptive "metrics" for the registry metadata
    flat = model.xT_flat()
    metrics = {
        "xT_min":  round(float(flat.min()), 6),
        "xT_max":  round(float(flat.max()), 6),
        "xT_mean": round(float(flat.mean()), 6),
        "xT_median": round(float(np.median(flat)), 6),
        "n_zones": float(len(flat)),
    }
    logger.info(
        "xT grid stats: min=%.4f  max=%.4f  mean=%.4f",
        metrics["xT_min"], metrics["xT_max"], metrics["xT_mean"],
    )

    registry.save("xt", model, metrics=metrics, feature_cols=[])
    return model, metrics


# ── P_scores ──────────────────────────────────────────────────────────────────

def train_p_scores(
    feature_df: pd.DataFrame,
    registry: ModelRegistry,
    **xgb_kwargs: object,
) -> tuple[PScoresModel, dict[str, float]]:
    """
    Train the P_scores XGBoost model.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Full feature matrix from FeatureAssembler.
    registry : ModelRegistry
    **xgb_kwargs
        Forwarded to PScoresModel constructor (n_estimators, max_depth, …).

    Returns
    -------
    (PScoresModel, metrics) : tuple[PScoresModel, dict[str, float]]
    """
    with timer("P_scores prepare data"):
        X, y = PScoresModel.prepare_training_data(feature_df)

    X_tr, X_te, y_tr, y_te = _train_test(X, y)

    model = PScoresModel(**xgb_kwargs)  # type: ignore[arg-type]
    with timer("P_scores fit"):
        model.fit(X_tr, y_tr)

    with timer("P_scores evaluate"):
        y_prob = model.predict_proba(X_te)
        metrics = evaluate_p_scores_model(y_te.values, y_prob)

    registry.save("p_scores", model, metrics=metrics, feature_cols=model.feature_cols)
    return model, metrics


# ── P_concedes ────────────────────────────────────────────────────────────────

def train_p_concedes(
    feature_df: pd.DataFrame,
    registry: ModelRegistry,
    **xgb_kwargs: object,
) -> tuple[PConcedesModel, dict[str, float]]:
    """
    Train the P_concedes XGBoost model.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Full feature matrix from FeatureAssembler.
    registry : ModelRegistry
    **xgb_kwargs
        Forwarded to PConcedesModel constructor.

    Returns
    -------
    (PConcedesModel, metrics) : tuple[PConcedesModel, dict[str, float]]
    """
    with timer("P_concedes prepare data"):
        X, y = PConcedesModel.prepare_training_data(feature_df)

    X_tr, X_te, y_tr, y_te = _train_test(X, y)

    model = PConcedesModel(**xgb_kwargs)  # type: ignore[arg-type]
    with timer("P_concedes fit"):
        model.fit(X_tr, y_tr)

    with timer("P_concedes evaluate"):
        y_prob = model.predict_proba(X_te)
        metrics = evaluate_p_concedes_model(y_te.values, y_prob)

    registry.save("p_concedes", model, metrics=metrics, feature_cols=model.feature_cols)
    return model, metrics
