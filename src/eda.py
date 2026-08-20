"""
eda.py
Exploratory data analysis on the raw Beijing PM2.5 dataset, run before any
cleaning or feature engineering. Produces the plots referenced in the
README's EDA section: missing-data pattern, the full 5-year PM2.5 series,
its distribution against AQI-style severity bands, a correlation heatmap
of the numeric weather variables, PM2.5 seasonality by month, and PM2.5 by
wind direction (the same variable the association rules later flag as the
strongest driver).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AQI_BANDS = [
    (0, 35, "Good", "#66c2a5"),
    (35, 75, "Moderate", "#ffd92f"),
    (75, 150, "Unhealthy", "#fc8d62"),
    (150, 250, "VeryUnhealthy", "#e34a33"),
    (250, 600, "Hazardous", "#7f0000"),
]


def plot_missingness(raw_df: pd.DataFrame, save_path):
    """Share of missing pm2.5 readings per month, across the full date range.
    The raw data's pm2.5 column is NaN for stretches of the dataset (most
    heavily at the very start); this shows where those gaps fall before
    they get interpolated away in data_loader.clean_data.
    """
    df = raw_df.copy()
    df["year_month"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    monthly = df.groupby("year_month")["pm2.5"].apply(lambda s: s.isna().mean() * 100)

    plt.figure(figsize=(12, 4.5))
    plt.bar(monthly.index, monthly.values, width=20, color="#C44E52")
    plt.ylabel("Missing pm2.5 readings (%)")
    plt.title("Missing PM2.5 data by month, before interpolation")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pm25_timeseries(cleaned_df: pd.DataFrame, save_path):
    """Full hourly PM2.5 series across the whole date range, interpolated."""
    plt.figure(figsize=(13, 4.5))
    plt.plot(cleaned_df["datetime"], cleaned_df["pm2.5"], linewidth=0.4, color="#4C72B0")
    plt.ylabel("PM2.5 (ug/m3)")
    plt.xlabel("Date")
    plt.title("Beijing PM2.5, hourly, 2010-2014 (missing values interpolated)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pm25_distribution(cleaned_df: pd.DataFrame, save_path):
    """Histogram of PM2.5 readings with AQI-style severity bands shaded,
    the same bins src/association_rules.py uses to discretize PM2.5."""
    plt.figure(figsize=(10, 5))
    plt.hist(cleaned_df["pm2.5"], bins=120, color="#4C72B0", edgecolor="none")
    ymax = plt.gca().get_ylim()[1]
    for lo, hi, label, color in AQI_BANDS:
        plt.axvspan(lo, min(hi, cleaned_df["pm2.5"].max()), color=color, alpha=0.15)
        plt.text(min(lo + 5, cleaned_df["pm2.5"].max()), ymax * 0.95, label,
                  rotation=90, va="top", fontsize=8, color=color)
    plt.xlabel("PM2.5 (ug/m3)")
    plt.ylabel("Hourly readings")
    plt.title("PM2.5 distribution with AQI severity bands")
    plt.xlim(0, cleaned_df["pm2.5"].quantile(0.995))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(cleaned_df: pd.DataFrame, save_path):
    """Correlation matrix of PM2.5 and the raw numeric weather variables."""
    cols = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    corr = cleaned_df[cols].corr()

    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                      fontsize=8, color="black")
    plt.title("Correlation: PM2.5 and weather variables")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_seasonal_boxplot(cleaned_df: pd.DataFrame, save_path):
    """PM2.5 distribution by calendar month, to show the winter-heating
    seasonality the association rules later quantify."""
    df = cleaned_df.copy()
    df["month"] = df["datetime"].dt.month
    data = [df.loc[df["month"] == m, "pm2.5"].values for m in range(1, 13)]
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    plt.figure(figsize=(10, 5))
    bp = plt.boxplot(data, showfliers=False, patch_artist=True)
    plt.xticks(range(1, 13), month_names)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    plt.ylabel("PM2.5 (ug/m3)")
    plt.title("PM2.5 by month (outliers hidden for readability)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pm25_by_wind_direction(cleaned_df: pd.DataFrame, save_path):
    """Mean PM2.5 by wind direction, the strongest single driver the
    association rules later identify (see outputs/association_rules_*.csv)."""
    df = cleaned_df.copy()
    means = df.groupby("cbwd")["pm2.5"].mean().sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(means.index, means.values, color="#55A868")
    for bar, val in zip(bars, means.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Mean PM2.5 (ug/m3)")
    plt.xlabel("Wind direction (cbwd)")
    plt.title("Mean PM2.5 by wind direction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_eda(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, output_dir: str):
    """Runs the full EDA suite and writes every plot to output_dir with an
    eda_ prefix. raw_df is the unmodified load (for the missingness plot);
    cleaned_df has pm2.5 interpolated and a datetime column (the output of
    data_loader.clean_data on data_loader.load_raw_data)."""
    import os
    plot_missingness(raw_df, os.path.join(output_dir, "eda_missingness.png"))
    plot_pm25_timeseries(cleaned_df, os.path.join(output_dir, "eda_pm25_timeseries.png"))
    plot_pm25_distribution(cleaned_df, os.path.join(output_dir, "eda_pm25_distribution.png"))
    plot_correlation_heatmap(cleaned_df, os.path.join(output_dir, "eda_correlation_heatmap.png"))
    plot_seasonal_boxplot(cleaned_df, os.path.join(output_dir, "eda_pm25_by_month.png"))
    plot_pm25_by_wind_direction(cleaned_df, os.path.join(output_dir, "eda_pm25_by_wind_direction.png"))
