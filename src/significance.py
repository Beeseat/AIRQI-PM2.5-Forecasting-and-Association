
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats


def paired_tests(cv_raw_df: pd.DataFrame, metric: str = "R2", baseline: str | None = None):
    """
    Runs paired t-test + Wilcoxon signed-rank on every pair of models
    """
    wide = cv_raw_df.pivot(index="Fold", columns="Algorithm", values=metric)
    algos = list(wide.columns)

    if baseline is not None:
        if baseline not in algos:
            raise ValueError(f"baseline '{baseline}' not found in {algos}")
        pairs = [(baseline, other) for other in algos if other != baseline]
    else:
        pairs = list(itertools.combinations(algos, 2))

    rows = []
    for a, b in pairs:
        diffs = (wide[a] - wide[b]).values  # positive => a scored higher

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, t_p = stats.ttest_rel(wide[a], wide[b])
            try:
                w_stat, w_p = stats.wilcoxon(wide[a], wide[b])
            except ValueError:
                # all differences identical / all zero -> wilcoxon undefined
                w_stat, w_p = np.nan, np.nan

        rows.append({
            "Model A": a,
            "Model B": b,
            "Metric": metric,
            "Mean(A) - Mean(B)": diffs.mean(),
            "Higher mean": a if diffs.mean() > 0 else b,
            "Paired t-test p": t_p,
            "Wilcoxon p": w_p,
            "Significant at 0.05 (t-test)": bool(t_p < 0.05) if not np.isnan(t_p) else None,
            "Significant at 0.05 (Wilcoxon)": bool(w_p < 0.05) if not np.isnan(w_p) else None,
        })

    return pd.DataFrame(rows)


def significance_summary_vs_baseline(cv_raw_df: pd.DataFrame, baseline: str, metrics=("R2", "RMSE", "MAE")):
    frames = [paired_tests(cv_raw_df, metric=m, baseline=baseline) for m in metrics]
    return pd.concat(frames, ignore_index=True)
