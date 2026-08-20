"""
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
  - Forecast horizon sweep (STEP 8): reruns the CV benchmark at +6h/+24h
    targets (on top of the original 1-hour-ahead setup) to test whether
    the tree-ensemble advantage over Ridge grows once lag_1h is stale
  - Exploratory data analysis (STEP 0): missingness, distribution, seasonality,
    and correlation plots on the raw data before any cleaning or modeling
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_and_prepare, load_for_association, load_raw_data
from src.models import run_timeseries_cv_benchmark, fit_final_models, cv_raw_scores_to_df
from src.utils import (
    plot_actual_vs_predicted, plot_shap_summary, plot_top_rules, plot_horizon_comparison,
    plot_cv_benchmark, plot_holdout_benchmark, plot_seed_robustness, plot_significance,
)
from src.eda import run_eda
from src.association_rules import mine_pollution_rules, rules_involving_pm25
from src.tuning import tune_all_models
from src.significance import significance_summary_vs_baseline
from src.robustness import seed_robustness, format_summary
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "PRSA_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Outputs are organized into numbered subfolders, one per pipeline stage,
# so results can be read in the same order the analysis narrative follows.
DIR_EDA = os.path.join(OUTPUT_DIR, "01_eda")
DIR_BENCHMARK = os.path.join(OUTPUT_DIR, "02_benchmark")
DIR_SIGNIFICANCE = os.path.join(OUTPUT_DIR, "03_significance")
DIR_SEED_ROBUSTNESS = os.path.join(OUTPUT_DIR, "04_seed_robustness")
DIR_TUNING = os.path.join(OUTPUT_DIR, "05_tuning")
DIR_HORIZON = os.path.join(OUTPUT_DIR, "06_forecast_horizon")
DIR_SHAP = os.path.join(OUTPUT_DIR, "07_explainability_shap")
DIR_ASSOC = os.path.join(OUTPUT_DIR, "08_association_rules")


def main():
    for d in (
        OUTPUT_DIR, DIR_EDA, DIR_BENCHMARK, DIR_SIGNIFICANCE, DIR_SEED_ROBUSTNESS,
        DIR_TUNING, DIR_HORIZON, DIR_SHAP, DIR_ASSOC,
    ):
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("STEP 0: Exploratory data analysis (raw data, before cleaning)")
    print("=" * 60)
    raw_for_eda = load_raw_data(DATA_PATH)
    cleaned_for_eda = load_for_association(DATA_PATH)
    run_eda(raw_for_eda, cleaned_for_eda, DIR_EDA)
    print("Saved outputs/01_eda/eda_missingness.png")
    print("Saved outputs/01_eda/eda_pm25_timeseries.png")
    print("Saved outputs/01_eda/eda_pm25_distribution.png")
    print("Saved outputs/01_eda/eda_correlation_heatmap.png")
    print("Saved outputs/01_eda/eda_pm25_by_month.png")
    print("Saved outputs/01_eda/eda_pm25_by_wind_direction.png")

    print("\n" + "=" * 60)
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
        os.path.join(DIR_BENCHMARK, "benchmark_results_cv.csv"), index=False
    )
    plot_cv_benchmark(cv_summary_sorted, os.path.join(DIR_BENCHMARK, "benchmark_cv_chart.png"))
    print("Saved outputs/02_benchmark/benchmark_cv_chart.png")

    print("\n" + "=" * 60)
    print("STEP 2b: Paired significance tests across CV folds (vs Ridge baseline)")
    print("=" * 60)
    cv_raw_df = cv_raw_scores_to_df(cv_raw)
    sig_results = significance_summary_vs_baseline(
        cv_raw_df, baseline="Ridge Regression (Baseline)", metrics=("R2", "RMSE", "MAE")
    )
    print(sig_results.to_string(index=False))
    sig_results.to_csv(os.path.join(DIR_SIGNIFICANCE, "significance_tests_vs_ridge.csv"), index=False)
    print("Saved outputs/03_significance/significance_tests_vs_ridge.csv")
    print("(n=5 folds -> treat p-values as a sanity check, not proof; see README)")
    plot_significance(sig_results, os.path.join(DIR_SIGNIFICANCE, "significance_r2_chart.png"), metric="R2")
    print("Saved outputs/03_significance/significance_r2_chart.png")

    print("\n" + "=" * 60)
    print("STEP 3: Final chronological 80/20 fit + test-set metrics")
    print("=" * 60)
    fitted_models, test_metrics, splits = fit_final_models(X, y, test_frac=0.2)
    X_train, X_test, y_train, y_test = splits
    print(test_metrics.sort_values("R2", ascending=False).to_string(index=False))
    test_metrics.to_csv(
        os.path.join(DIR_BENCHMARK, "benchmark_results_final_holdout.csv"), index=False
    )
    plot_holdout_benchmark(
        test_metrics, os.path.join(DIR_BENCHMARK, "benchmark_holdout_chart.png"),
        title="Chronological 80/20 holdout (untuned)"
    )
    print("Saved outputs/02_benchmark/benchmark_holdout_chart.png")

    best_name = test_metrics.sort_values("R2", ascending=False).iloc[0]["Algorithm"]
    best_model = fitted_models[best_name]
    print(f"\nBest model on held-out test set: {best_name}")

    print("\n" + "=" * 60)
    print("STEP 4: Hyperparameter tuning (TimeSeriesSplit search, train set only)")
    print("=" * 60)
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
        os.path.join(DIR_TUNING, "benchmark_results_holdout_tuned_vs_untuned.csv"), index=False
    )
    print("Saved outputs/05_tuning/benchmark_results_holdout_tuned_vs_untuned.csv")
    plot_holdout_benchmark(
        combined_metrics, os.path.join(DIR_TUNING, "benchmark_tuned_vs_untuned_chart.png"),
        title="Holdout: tuned vs untuned"
    )
    print("Saved outputs/05_tuning/benchmark_tuned_vs_untuned_chart.png")

    best_params_df = pd.DataFrame([
        {"Algorithm": name, "Best Params": params, "Tuning CV RMSE": tuning_cv_rmse[name]}
        for name, params in best_params.items()
    ])
    best_params_df.to_csv(os.path.join(DIR_TUNING, "tuning_best_params.csv"), index=False)
    print("Saved outputs/05_tuning/tuning_best_params.csv")

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
    print("STEP 4b: Seed-robustness check (base models, fixed holdout split)")
    print("=" * 60)
    seed_summary, seed_per_seed = seed_robustness(
        X_train, X_test, y_train, y_test, seeds=(0, 1, 2, 3, 4)
    )
    print(f"\nHoldout metrics across {int(seed_summary['n_seeds'].iloc[0])} seeds (only random_state varies):")
    print(format_summary(seed_summary))
    seed_summary.to_csv(os.path.join(DIR_SEED_ROBUSTNESS, "seed_robustness_summary.csv"), index=False)
    seed_per_seed.to_csv(os.path.join(DIR_SEED_ROBUSTNESS, "seed_robustness_per_seed.csv"), index=False)
    print("Saved outputs/04_seed_robustness/seed_robustness_summary.csv and outputs/04_seed_robustness/seed_robustness_per_seed.csv")
    plot_seed_robustness(seed_summary, os.path.join(DIR_SEED_ROBUSTNESS, "seed_robustness_chart.png"))
    print("Saved outputs/04_seed_robustness/seed_robustness_chart.png")

    print("\n" + "=" * 60)
    print("STEP 5: Actual vs Predicted plot (best model)")
    print("=" * 60)
    preds = best_model.predict(X_test)
    plot_actual_vs_predicted(
        y_test, preds, best_name,
        os.path.join(DIR_BENCHMARK, "actual_vs_predicted_comparison.png")
    )
    print("Saved outputs/02_benchmark/actual_vs_predicted_comparison.png")

    print("\n" + "=" * 60)
    print("STEP 6: SHAP interpretability (tuned XGBoost)")
    print("=" * 60)
    xgb_model = fitted_tuned.get("XGBoost Regressor (Redo)", fitted_models["XGBoost Regressor (Redo)"])
    plot_shap_summary(
        xgb_model, X_test,
        os.path.join(DIR_SHAP, "feature_importance_shap.png")
    )
    print("Saved outputs/07_explainability_shap/feature_importance_shap.png")

    print("\n" + "=" * 60)
    print("STEP 7: Association rule mining (Apriori) on weather/PM2.5 conditions")
    print("=" * 60)
    assoc_raw = load_for_association(DATA_PATH)

    frequent_general, rules_general = mine_pollution_rules(
        assoc_raw, min_support=0.02, min_confidence=0.4, max_len=3
    )
    pm_rules_general = rules_involving_pm25(rules_general)
    print(f"Frequent itemsets (support>=0.02): {len(frequent_general)}")
    print(f"Rules (confidence>=0.4): {len(rules_general)}  |  with PM2.5 as consequent: {len(pm_rules_general)}")
    pm_rules_general.to_csv(os.path.join(DIR_ASSOC, "association_rules_pm25.csv"), index=False)
    print("Saved outputs/08_association_rules/association_rules_pm25.csv")

    _, rules_severe = mine_pollution_rules(
        assoc_raw, min_support=0.006, min_confidence=0.2, max_len=4
    )
    pm_rules_severe = rules_involving_pm25(rules_severe)
    pm_rules_severe = pm_rules_severe[
        pm_rules_severe["consequent"].isin(["PM25=VeryUnhealthy", "PM25=Hazardous"])
    ].sort_values("lift", ascending=False).reset_index(drop=True)
    pm_rules_severe.to_csv(os.path.join(DIR_ASSOC, "association_rules_severe_pm25.csv"), index=False)
    print(f"Rules for VeryUnhealthy/Hazardous bands: {len(pm_rules_severe)}")
    print("Saved outputs/08_association_rules/association_rules_severe_pm25.csv")

    if len(pm_rules_general):
        print("\nTop 5 general rules by lift (PM2.5 as consequent):")
        print(pm_rules_general.head(5).to_string(index=False))
    if len(pm_rules_severe):
        print("\nTop 5 severe-pollution rules by lift:")
        print(pm_rules_severe.head(5).to_string(index=False))

    plot_top_rules(
        pm_rules_general,
        os.path.join(DIR_ASSOC, "association_rules_top_lift.png"),
        top_n=15,
        title="Top association rules by lift (PM2.5 as consequent)"
    )
    print("Saved outputs/08_association_rules/association_rules_top_lift.png")

    plot_top_rules(
        pm_rules_severe,
        os.path.join(DIR_ASSOC, "association_rules_severe_top_lift.png"),
        top_n=12,
        title="Top rules for VeryUnhealthy / Hazardous PM2.5"
    )
    print("Saved outputs/08_association_rules/association_rules_severe_top_lift.png")

    print("\n" + "=" * 60)
    print("STEP 8: Forecast horizon comparison (0 / +6 / +24 extra hours)")
    print("=" * 60)
    HORIZONS = [0, 6, 24]
    horizon_rows = []
    for h in HORIZONS:
        print(f"\n-- horizon extra hours = {h} --")
        X_h, y_h, _ = load_and_prepare(DATA_PATH, horizon=h)
        cv_summary_h, _, _ = run_timeseries_cv_benchmark(X_h, y_h, n_splits=5)
        cv_summary_h["Horizon"] = h
        print(cv_summary_h.sort_values("R2", ascending=False).to_string(index=False))
        horizon_rows.append(cv_summary_h)

    horizon_df = pd.concat(horizon_rows, ignore_index=True)
    horizon_df.to_csv(os.path.join(DIR_HORIZON, "benchmark_results_by_horizon.csv"), index=False)
    print("\nSaved outputs/benchmark_results_by_horizon.csv")

    plot_horizon_comparison(
        horizon_df, os.path.join(DIR_HORIZON, "horizon_comparison_r2.png"), metric="R2"
    )
    print("Saved outputs/06_forecast_horizon/horizon_comparison_r2.png")
    plot_horizon_comparison(
        horizon_df, os.path.join(DIR_HORIZON, "horizon_comparison_rmse.png"), metric="RMSE"
    )
    print("Saved outputs/06_forecast_horizon/horizon_comparison_rmse.png")

    print("\n" + "=" * 60)
    print("STEP 8b: SHAP at +24h horizon (does lag_1h still dominate?)")
    print("=" * 60)
    X_24, y_24, _ = load_and_prepare(DATA_PATH, horizon=24)
    fitted_24, test_metrics_24, splits_24 = fit_final_models(X_24, y_24, test_frac=0.2)
    _, X_test_24, _, _ = splits_24
    plot_shap_summary(
        fitted_24["XGBoost Regressor (Redo)"], X_test_24,
        os.path.join(DIR_SHAP, "feature_importance_shap_horizon24.png")
    )
    print("Saved outputs/07_explainability_shap/feature_importance_shap_horizon24.png")

    print("\nPipeline complete. All outputs written to /outputs.")


if __name__ == "__main__":
    main()
