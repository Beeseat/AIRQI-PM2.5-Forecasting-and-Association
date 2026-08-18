"""
main.py - Central pipeline orchestrator.

Redo of: Dieselmarble/Beijing-PM2.5-Forecasting
(original project used Ridge/Lasso Regression and Gaussian Kernels
on the UCI Beijing PM2.5 dataset)

This version adds:
  - Cyclical (sin/cos) temporal encoding
  - Lag + rolling-window features
  - A real 4-way benchmark: Ridge vs Decision Tree vs Random Forest vs XGBoost
  - Chronological TimeSeriesSplit cross-validation (no lookahead bias)
  - SHAP TreeExplainer interpretability on the final model
  - Association rule mining (custom Apriori) over discretized weather/PM2.5
    conditions - a second data-mining paradigm alongside the regression
    benchmark, chosen because hourly PM2.5 barely changes hour-to-hour
    (see lag_1h dominance in SHAP), which makes hour-ahead classification
    a weak exercise but doesn't affect market-basket-style association mining
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_and_prepare, load_for_association
from src.models import run_timeseries_cv_benchmark, fit_final_models, cv_raw_scores_to_df
from src.utils import plot_actual_vs_predicted, plot_shap_summary, plot_top_rules
from src.association_rules import mine_pollution_rules, rules_involving_pm25
from src.tuning import tune_all_models
from src.significance import significance_summary_vs_baseline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "PRSA_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading & preparing real UCI Beijing PM2.5 data")
    print("=" * 60)
    X, y, dt = load_and_prepare(DATA_PATH)
    print(f"Rows after cleaning + feature engineering: {len(X)}")
    print(f"Features: {list(X.columns)}")
    print(f"Date range: {dt.min()} -> {dt.max()}\n")

    print("=" * 60)
    print("STEP 2: TimeSeriesSplit CV benchmark (5 folds, all 4 models)")
    print("=" * 60)
    cv_summary, _, cv_raw = run_timeseries_cv_benchmark(X, y, n_splits=5)
    cv_summary_sorted = cv_summary.sort_values("R2", ascending=False)
    print("\nCross-validated benchmark (mean across folds):")
    print(cv_summary_sorted.to_string(index=False))
    cv_summary_sorted.to_csv(
        os.path.join(OUTPUT_DIR, "benchmark_results_cv.csv"), index=False
    )

    print("\n" + "=" * 60)
    print("STEP 2b: Paired significance tests across CV folds (vs Ridge baseline)")
    print("=" * 60)
    # No new model fits needed - reuses the 5 paired fold scores from STEP 2.
    # CAVEAT: n=5 folds is a small sample; a non-significant p-value here
    # means "not enough evidence to tell these apart," not "they're equal."
    cv_raw_df = cv_raw_scores_to_df(cv_raw)
    sig_results = significance_summary_vs_baseline(
        cv_raw_df, baseline="Ridge Regression (Baseline)", metrics=("R2", "RMSE", "MAE")
    )
    print(sig_results.to_string(index=False))
    sig_results.to_csv(os.path.join(OUTPUT_DIR, "significance_tests_vs_ridge.csv"), index=False)
    print("Saved outputs/significance_tests_vs_ridge.csv")
    print("(n=5 folds -> treat p-values as a sanity check, not proof; see README)")

    print("\n" + "=" * 60)
    print("STEP 3: Final chronological 80/20 fit + test-set metrics")
    print("=" * 60)
    fitted_models, test_metrics, splits = fit_final_models(X, y, test_frac=0.2)
    X_train, X_test, y_train, y_test = splits
    print(test_metrics.sort_values("R2", ascending=False).to_string(index=False))
    test_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "benchmark_results_final_holdout.csv"), index=False
    )

    best_name = test_metrics.sort_values("R2", ascending=False).iloc[0]["Algorithm"]
    best_model = fitted_models[best_name]
    print(f"\nBest model on held-out test set: {best_name}")

    print("\n" + "=" * 60)
    print("STEP 4: Hyperparameter tuning (TimeSeriesSplit search, train set only)")
    print("=" * 60)
    # Tuning is run ONLY on X_train/y_train (never X_test/y_test), so the
    # holdout numbers below stay a genuine out-of-sample estimate. Search
    # itself uses TimeSeriesSplit internally, for the same no-leakage
    # reason the CV benchmark in STEP 2 does.
    fitted_tuned, best_params, tuning_cv_rmse = tune_all_models(X_train, y_train)

    tuned_metrics = []
    for name, model in fitted_tuned.items():
        preds = model.predict(X_test)
        tuned_metrics.append({
            "Algorithm": name + " [Tuned]",
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        })
    tuned_metrics_df = pd.DataFrame(tuned_metrics)

    combined_metrics = pd.concat([test_metrics, tuned_metrics_df], ignore_index=True)
    combined_metrics = combined_metrics.sort_values("R2", ascending=False).reset_index(drop=True)
    print("\nUntuned vs tuned, final holdout (all 8 rows):")
    print(combined_metrics.to_string(index=False))
    combined_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "benchmark_results_holdout_tuned_vs_untuned.csv"), index=False
    )
    print("Saved outputs/benchmark_results_holdout_tuned_vs_untuned.csv")

    best_params_df = pd.DataFrame([
        {"Algorithm": name, "Best Params": params, "Tuning CV RMSE": tuning_cv_rmse[name]}
        for name, params in best_params.items()
    ])
    best_params_df.to_csv(os.path.join(OUTPUT_DIR, "tuning_best_params.csv"), index=False)
    print("Saved outputs/tuning_best_params.csv")

    # Update "best model" / XGBoost references below to the tuned model,
    # since it's the more defensible final estimate.
    best_tuned_name = tuned_metrics_df.sort_values("R2", ascending=False).iloc[0]["Algorithm"]
    print(f"\nBest model after tuning (holdout): {best_tuned_name}")
    if tuned_metrics_df.sort_values("R2", ascending=False).iloc[0]["R2"] > \
       test_metrics.sort_values("R2", ascending=False).iloc[0]["R2"]:
        best_name = best_tuned_name.replace(" [Tuned]", "")
        best_model = fitted_tuned[best_name]
        print("Tuned model beats the best untuned model on holdout R2; using it for the plot/SHAP steps below.")
    else:
        print("Tuning did not beat the best untuned holdout R2; keeping the untuned best model for the plot/SHAP steps below.")

    print("\n" + "=" * 60)
    print("STEP 5: Actual vs Predicted plot (best model)")
    print("=" * 60)
    preds = best_model.predict(X_test)
    plot_actual_vs_predicted(
        y_test, preds, best_name,
        os.path.join(OUTPUT_DIR, "actual_vs_predicted_comparison.png")
    )
    print("Saved outputs/actual_vs_predicted_comparison.png")

    print("\n" + "=" * 60)
    print("STEP 6: SHAP interpretability (tuned XGBoost)")
    print("=" * 60)
    xgb_model = fitted_tuned.get("XGBoost Regressor (Redo)", fitted_models["XGBoost Regressor (Redo)"])
    plot_shap_summary(
        xgb_model, X_test,
        os.path.join(OUTPUT_DIR, "feature_importance_shap.png")
    )
    print("Saved outputs/feature_importance_shap.png")

    print("\n" + "=" * 60)
    print("STEP 7: Association rule mining (Apriori) on weather/PM2.5 conditions")
    print("=" * 60)
    assoc_raw = load_for_association(DATA_PATH)

    # General pass: broad, reasonably common condition co-occurrences
    frequent_general, rules_general = mine_pollution_rules(
        assoc_raw, min_support=0.02, min_confidence=0.4, max_len=3
    )
    pm_rules_general = rules_involving_pm25(rules_general)
    print(f"Frequent itemsets (support>=0.02): {len(frequent_general)}")
    print(f"Rules (confidence>=0.4): {len(rules_general)}  |  with PM2.5 as consequent: {len(pm_rules_general)}")
    pm_rules_general.to_csv(os.path.join(OUTPUT_DIR, "association_rules_pm25.csv"), index=False)
    print("Saved outputs/association_rules_pm25.csv")

    # Targeted pass: lower thresholds + longer itemsets to surface rules for
    # the rarer severe pollution bands (VeryUnhealthy / Hazardous), which the
    # min_support=0.02 general pass mostly misses simply because those bands
    # occur less often in the data (~7% Hazardous, ~14% VeryUnhealthy overall)
    _, rules_severe = mine_pollution_rules(
        assoc_raw, min_support=0.006, min_confidence=0.2, max_len=4
    )
    pm_rules_severe = rules_involving_pm25(rules_severe)
    pm_rules_severe = pm_rules_severe[
        pm_rules_severe["consequent"].isin(["PM25=VeryUnhealthy", "PM25=Hazardous"])
    ].sort_values("lift", ascending=False).reset_index(drop=True)
    pm_rules_severe.to_csv(os.path.join(OUTPUT_DIR, "association_rules_severe_pm25.csv"), index=False)
    print(f"Rules for VeryUnhealthy/Hazardous bands: {len(pm_rules_severe)}")
    print("Saved outputs/association_rules_severe_pm25.csv")

    if len(pm_rules_general):
        print("\nTop 5 general rules by lift (PM2.5 as consequent):")
        print(pm_rules_general.head(5).to_string(index=False))
    if len(pm_rules_severe):
        print("\nTop 5 severe-pollution rules by lift:")
        print(pm_rules_severe.head(5).to_string(index=False))

    plot_top_rules(
        pm_rules_general,
        os.path.join(OUTPUT_DIR, "association_rules_top_lift.png"),
        top_n=15,
        title="Top association rules by lift (PM2.5 as consequent)"
    )
    print("Saved outputs/association_rules_top_lift.png")

    plot_top_rules(
        pm_rules_severe,
        os.path.join(OUTPUT_DIR, "association_rules_severe_top_lift.png"),
        top_n=12,
        title="Top rules for VeryUnhealthy / Hazardous PM2.5"
    )
    print("Saved outputs/association_rules_severe_top_lift.png")

    print("\nPipeline complete. All outputs written to /outputs.")


if __name__ == "__main__":
    main()
