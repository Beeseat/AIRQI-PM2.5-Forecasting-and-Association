import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models import get_models

DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _build_models_for_seed(seed: int, rf_n_estimators: int | None = None):
    models = get_models()
    for name, model in models.items():
        try:
            model.set_params(random_state=seed)
        except (ValueError, TypeError):
            pass  # model has no random_state (none currently, but be safe)
    if rf_n_estimators is not None and "Random Forest Regressor" in models:
        models["Random Forest Regressor"].set_params(n_estimators=rf_n_estimators)
    return models


def seed_robustness(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seeds=DEFAULT_SEEDS,
    rf_n_estimators: int | None = None,
):
    """
    Refit every base model once per seed on the fixed (X_train -> X_test)
    holdout and collect per-seed holdout metrics.
    """
    seeds = list(seeds)
    rows = []
    for seed in seeds:
        models = _build_models_for_seed(seed, rf_n_estimators=rf_n_estimators)
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rows.append({
                "Algorithm": name,
                "Seed": seed,
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                "R2": r2_score(y_test, preds),
            })

    per_seed = pd.DataFrame(rows)
    summary = _summarize(per_seed, n_seeds=len(seeds))
    return summary, per_seed


def _summarize(per_seed: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """Collapse the long per-seed frame into per-model mean/std/min/max."""
    metrics = ("R2", "RMSE", "MAE")
    # population std (ddof=0): we're describing the spread of the seeds we ran,
    # not inferring a wider population, so ddof=0 is the honest choice here.
    agg = per_seed.groupby("Algorithm")[list(metrics)].agg(
        ["mean", "std", "min", "max"]
    )
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[f"{m}_mean"] = agg[(m, "mean")]
        out[f"{m}_std"] = per_seed.groupby("Algorithm")[m].std(ddof=0)
        out[f"{m}_min"] = agg[(m, "min")]
        out[f"{m}_max"] = agg[(m, "max")]
        out[f"{m}_range"] = out[f"{m}_max"] - out[f"{m}_min"]
    out["n_seeds"] = n_seeds
    out = out.reset_index().sort_values("R2_mean", ascending=False).reset_index(drop=True)
    return out


def format_summary(summary: pd.DataFrame) -> str:
    lines = []
    for _, r in summary.iterrows():
        lines.append(
            f"  {r['Algorithm']:<32} "
            f"R2 {r['R2_mean']:.5f} +/- {r['R2_std']:.5f} "
            f"(range {r['R2_range']:.5f})"
        )
    return "\n".join(lines)
