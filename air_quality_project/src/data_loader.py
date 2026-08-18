"""
data_loader.py
Loads the real UCI Beijing PM2.5 dataset (PRSA_data.csv), cleans it,
and engineers cyclical + lag/rolling features for time-series modeling.

Dataset source: UCI Machine Learning Repository - "Beijing PM2.5 Data"
(Chen, S. 2015. https://doi.org/10.24432/C5JS49)
43,824 hourly observations, Jan 1 2010 - Dec 31 2014.
"""

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and build a proper datetime index."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing pm2.5 values (the raw data marks them as NA).
    The first ~24 hours of the dataset have no pm2.5 reading at all,
    so those rows are dropped; remaining gaps are time-interpolated.
    """
    df = df.copy()
    df["pm2.5"] = df["pm2.5"].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=["pm2.5"]).reset_index(drop=True)

    # one-hot encode wind direction (categorical)
    df = pd.get_dummies(df, columns=["cbwd"], prefix="wind")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical encoding of periodic time features + lag/rolling features
    of the target variable, as described in the project README.
    """
    df = df.copy()

    df["dayofweek"] = df["datetime"].dt.dayofweek

    # --- Cyclical encoding ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # --- Lag & rolling window features (of the target itself) ---
    df["lag_1h"] = df["pm2.5"].shift(1)
    df["lag_2h"] = df["pm2.5"].shift(2)
    df["roll_mean_6h"] = df["pm2.5"].shift(1).rolling(window=6).mean()
    df["roll_mean_24h"] = df["pm2.5"].shift(1).rolling(window=24).mean()

    # drop raw columns superseded by cyclical encodings / no longer needed
    df = df.drop(columns=["No", "year", "month", "day", "hour", "dayofweek"])
    df = df.dropna().reset_index(drop=True)

    return df


def load_and_prepare(path: str):
    """Full pipeline: raw -> cleaned -> feature engineered -> X, y, datetime."""
    raw = load_raw_data(path)
    cleaned = clean_data(raw)
    feats = engineer_features(cleaned)

    datetime_col = feats["datetime"]
    y = feats["pm2.5"]
    X = feats.drop(columns=["datetime", "pm2.5"])

    return X, y, datetime_col


def load_for_association(path: str) -> pd.DataFrame:
    """
    Lightly-cleaned version of the raw data for association rule mining.
    Unlike clean_data(), this keeps 'cbwd' as a plain categorical column
    (not one-hot encoded) since the association module bins it into
    transaction items itself.
    """
    df = load_raw_data(path)
    df["pm2.5"] = df["pm2.5"].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=["pm2.5"]).reset_index(drop=True)
    return df
