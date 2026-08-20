"""
tuning.py
Hyperparameter tuning for the 4-model benchmark, using TimeSeriesSplit
inside RandomizedSearchCV / GridSearchCV so tuning never leaks future
rows into a validation fold (the same leakage concern that motivated
TimeSeriesSplit in models.py's CV benchmark).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb

# scipy's randint/uniform give RandomizedSearchCV proper distributions to
# sample from; falls back to plain lists if scipy isn't available.
try:
    from scipy.stats import randint, uniform
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _search_cv(n_splits: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def tune_ridge(X_train, y_train, n_splits=5):
    """Small enough to grid search exhaustively."""
    param_grid = {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]}
    search = GridSearchCV(
        Ridge(random_state=42),
        param_grid=param_grid,
        cv=_search_cv(n_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search


def tune_decision_tree(X_train, y_train, n_splits=5):
    param_grid = {
        "max_depth": [4, 6, 8, 10, 14, None],
        "min_samples_leaf": [1, 5, 10, 20],
    }
    search = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=_search_cv(n_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search


def tune_random_forest(X_train, y_train, n_splits=2, n_iter=4):
    """
    RandomForest is by far the most expensive model to refit repeatedly
    (each fit trains n_estimators full trees on up to ~35k rows), so this
    uses fewer CV folds and far fewer random-search draws than the other
    models, and caps n_estimators/max_depth, to keep tuning time
    reasonable on a single machine. Widen n_splits/n_iter/the ranges below
    if you have more time to spare; the defaults are a deliberate
    speed/thoroughness tradeoff, not a claim that this is the optimal
    search budget. max_features excludes None (bootstrap over ALL
    features per split) since that's the slowest and least regularized
    option and is rarely picked over sqrt/0.5-0.7 on tabular data anyway.
    """
    if _HAVE_SCIPY:
        param_dist = {
            "n_estimators": randint(80, 220),
            "max_depth": randint(6, 16),
            "min_samples_leaf": randint(2, 20),
            "max_features": [0.5, 0.7, "sqrt"],
        }
    else:
        param_dist = {
            "n_estimators": [80, 120, 150, 180, 220],
            "max_depth": [6, 8, 10, 12, 14, 16],
            "min_samples_leaf": [2, 5, 10, 15, 20],
            "max_features": [0.5, 0.7, "sqrt"],
        }

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=_search_cv(n_splits),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,  # RF already parallelizes internally (n_jobs=-1 above);
                   # parallelizing the search on top oversubscribes cores.
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search


def tune_xgboost(X_train, y_train, n_splits=4, n_iter=15):
    """
    The model whose 'barely beats Ridge' result the README leans on most,
    so this gets the largest search budget of the four.
    """
    if _HAVE_SCIPY:
        param_dist = {
            "n_estimators": randint(150, 500),
            "max_depth": randint(3, 8),
            "learning_rate": uniform(0.02, 0.13),   # 0.02 - 0.15
            "subsample": uniform(0.6, 0.4),         # 0.6 - 1.0
            "colsample_bytree": uniform(0.6, 0.4),  # 0.6 - 1.0
            "min_child_weight": randint(1, 10),
        }
    else:
        param_dist = {
            "n_estimators": [150, 250, 350, 400, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 8],
        }

    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=_search_cv(n_splits),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,  # same reasoning as RF: XGBoost already uses all cores
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search


TUNERS = {
    "Ridge Regression (Baseline)": tune_ridge,
    "Decision Tree Regressor": tune_decision_tree,
    "Random Forest Regressor": tune_random_forest,
    "XGBoost Regressor (Redo)": tune_xgboost,
}


def tune_all_models(X_train, y_train, verbose=True):
    """
    Runs every tuner on the SAME training portion (never the holdout) and
    returns:
      - fitted_tuned: {name: fitted best_estimator_}
      - best_params: {name: dict of best hyperparameters}
      - cv_rmse: {name: best cross-validated RMSE found during search}
    """
    fitted_tuned, best_params, cv_rmse = {}, {}, {}

    for name, tuner_fn in TUNERS.items():
        if verbose:
            print(f"Tuning {name} ...")
        best_est, params, search = tuner_fn(X_train, y_train)
        fitted_tuned[name] = best_est
        best_params[name] = params
        cv_rmse[name] = -search.best_score_  # scorer is negated RMSE
        if verbose:
            print(f"  best params: {params}")
            print(f"  best CV RMSE: {cv_rmse[name]:.3f}")

    return fitted_tuned, best_params, cv_rmse
