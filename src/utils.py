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
