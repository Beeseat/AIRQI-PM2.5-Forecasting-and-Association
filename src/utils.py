"""
utils.py
Visualization and SHAP interpretability helpers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def plot_actual_vs_predicted(y_test, preds, model_name, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:500], label="Actual", linewidth=1.2)
    plt.plot(preds[:500], label="Predicted", linewidth=1.0, alpha=0.8)
    plt.title(f"Actual vs Predicted PM2.5 - {model_name} (first 500 test hours)")
    plt.xlabel("Time step (hours)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_shap_summary(model, X_test, save_path, sample_size=2000):
    """SHAP TreeExplainer summary plot for the final XGBoost model."""
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return shap_values


def plot_horizon_comparison(horizon_df, save_path, metric="R2"):
    """
    Line chart of `metric` vs. forecast horizon, one line per algorithm.
    horizon_df must have columns: Algorithm, Horizon, <metric>.
    Designed to show whether tree ensembles pull away from Ridge as the
    horizon grows (the project's core forecast-horizon question).
    """
    plt.figure(figsize=(9, 5.5))
    colors = {
        "Ridge Regression (Baseline)": "#4C72B0",
        "Decision Tree Regressor": "#DD8452",
        "Random Forest Regressor": "#55A868",
        "XGBoost Regressor (Redo)": "#C44E52",
    }
    for name, group in horizon_df.groupby("Algorithm"):
        group = group.sort_values("Horizon")
        plt.plot(
            group["Horizon"], group[metric],
            marker="o", linewidth=2, label=name,
            color=colors.get(name),
        )
    plt.xlabel("Extra hours ahead of the original 1-hour-ahead setup")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. forecast horizon (5-fold TimeSeriesSplit CV)")
    plt.xticks(sorted(horizon_df["Horizon"].unique()))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _short_label(name):
    """Trims the repetitive 'Regressor'/'Regression' suffix so rotated
    x-axis labels take less horizontal space, without losing the
    baseline/tuned/redo qualifiers that distinguish otherwise-identical names."""
    return (
        name.replace(" Regressor", "").replace(" Regression", "")
    )


def plot_cv_benchmark(cv_summary, save_path):
    """Grouped bar chart of MAE, RMSE, and R2 across the CV benchmark, one group per model."""
    df = cv_summary.sort_values("R2", ascending=False)
    metrics = ["MAE", "RMSE", "R2"]
    n = len(df)
    fig, axes = plt.subplots(1, 3, figsize=(max(15, 3.2 * n), 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    labels = [_short_label(a) for a in df["Algorithm"]]
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(range(n), df[metric], color=colors[:n])
        ax.set_title(metric)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("5-fold TimeSeriesSplit CV benchmark")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_holdout_benchmark(metrics_df, save_path, title="Chronological 80/20 holdout"):
    """Grouped bar chart of MAE, RMSE, and R2 for a holdout results table."""
    df = metrics_df.sort_values("R2", ascending=False)
    metrics = ["MAE", "RMSE", "R2"]
    n = len(df)
    fig, axes = plt.subplots(1, 3, figsize=(max(15, 3.2 * n), 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    labels = [_short_label(a) for a in df["Algorithm"]]
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(range(n), df[metric], color=colors)
        ax.set_title(metric)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_seed_robustness(seed_summary, save_path):
    """Bar chart of mean holdout R2 per model with error bars for seed-to-seed std."""
    df = seed_summary.sort_values("R2_mean", ascending=False)
    labels = [_short_label(a) for a in df["Algorithm"]]
    plt.figure(figsize=(9, 5.5))
    plt.bar(labels, df["R2_mean"], yerr=df["R2_std"], capsize=6,
            color="#4C72B0", alpha=0.85)
    plt.ylabel("Holdout R2 (mean +/- std across 5 seeds)")
    plt.title("Seed robustness: holdout R2 by model")
    plt.xticks(rotation=20, ha="right")
    ymin = max(0, df["R2_mean"].min() - 0.05)
    plt.ylim(ymin, df["R2_mean"].max() + df["R2_std"].max() + 0.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_significance(sig_df, save_path, metric="R2"):
    """Bar chart of paired t-test / Wilcoxon p-values vs Ridge, with the p=0.05 line marked."""
    df = sig_df[sig_df["Metric"] == metric].copy()
    labels = [_short_label(a) for a in df["Model B"]]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9.5, 5.5))
    plt.bar(x - width / 2, df["Paired t-test p"], width, label="Paired t-test p", color="#4C72B0")
    plt.bar(x + width / 2, df["Wilcoxon p"], width, label="Wilcoxon p", color="#DD8452")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("p-value")
    plt.title(f"Significance vs Ridge baseline ({metric})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_rules(rules_df, save_path, top_n=15, title="Top association rules by lift"):
    """
    Horizontal bar chart of the top-N association rules ranked by lift.
    Each bar is labeled 'antecedent -> consequent'.
    """
    if rules_df is None or len(rules_df) == 0:
        return

    top = rules_df.sort_values("lift", ascending=False).head(top_n).iloc[::-1]
    labels = [f"{row.antecedent}  ->  {row.consequent}" for row in top.itertuples()]

    plt.figure(figsize=(11, max(4, 0.4 * len(top))))
    bars = plt.barh(labels, top["lift"], color="#4C72B0")
    plt.xlabel("Lift")
    plt.title(title)
    plt.axvline(1.0, color="grey", linestyle="--", linewidth=1, label="Lift = 1 (no association)")
    plt.legend(loc="lower right")
    plt.xlim(0, top["lift"].max() * 1.18)  # headroom so confidence labels aren't clipped

    for bar, conf in zip(bars, top["confidence"]):
        plt.text(bar.get_width() + top["lift"].max() * 0.015, bar.get_y() + bar.get_height() / 2,
                  f"conf={conf:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
