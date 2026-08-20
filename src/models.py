"""
models.py
4-way algorithm benchmark: Ridge (baseline) vs Decision Tree vs Random Forest
vs XGBoost (redo), evaluated with chronological TimeSeriesSplit cross-validation
to avoid the lookahead bias that random k-fold CV would introduce on
time-series data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def get_models():
    return {
        "Ridge Regression (Baseline)": Ridge(alpha=1.0, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
        ),
        "XGBoost Regressor (Redo)": xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


def run_timeseries_cv_benchmark(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Runs every model through the SAME TimeSeriesSplit folds and reports
    mean MAE / RMSE / R2 across folds. This is the real, reproducible
    replacement for the earlier hand-typed benchmark table.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = get_models()
    results = {name: {"MAE": [], "RMSE": [], "R2": []} for name in models}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            results[name]["MAE"].append(mean_absolute_error(y_val, preds))
            results[name]["RMSE"].append(np.sqrt(mean_squared_error(y_val, preds)))
            results[name]["R2"].append(r2_score(y_val, preds))

        print(f"Fold {fold + 1}/{n_splits} complete.")

    summary_rows = []
    for name, metrics in results.items():
        summary_rows.append({
            "Algorithm": name,
            "MAE": np.mean(metrics["MAE"]),
            "RMSE": np.mean(metrics["RMSE"]),
            "R2": np.mean(metrics["R2"]),
        })

    summary_df = pd.DataFrame(summary_rows)

    baseline_r2 = summary_df.loc[
        summary_df["Algorithm"] == "Ridge Regression (Baseline)", "R2"
    ].values[0]
    summary_df["Improvement vs Baseline (R2 pts)"] = summary_df["R2"] - baseline_r2

    return summary_df, models, results


def cv_raw_scores_to_df(results: dict) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        n_folds = len(metrics["R2"])
        for fold in range(n_folds):
            rows.append({
                "Algorithm": name,
                "Fold": fold + 1,
                "MAE": metrics["MAE"][fold],
                "RMSE": metrics["RMSE"][fold],
                "R2": metrics["R2"][fold],
            })
    return pd.DataFrame(rows)


def fit_final_models(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    """
    Chronological 80/20 train-test split (no shuffling) + fit all 4 models
    on the full training history for final evaluation / plotting / SHAP.
    """
    split_idx = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = get_models()
    fitted = {}
    test_metrics = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fitted[name] = model
        test_metrics.append({
            "Algorithm": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        })

    return fitted, pd.DataFrame(test_metrics), (X_train, X_test, y_train, y_test)
