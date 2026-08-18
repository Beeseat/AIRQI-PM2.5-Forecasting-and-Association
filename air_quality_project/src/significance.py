"""
significance.py
Paired significance testing across the 5 TimeSeriesSplit CV folds already
computed in models.run_timeseries_cv_benchmark. "Paired" here means: fold 1
used the exact same train/validation split for every model, fold 2 used the
same split for every model, etc. - so for any two models we have 5 matched
observations (one per fold), which is exactly the setup a paired test wants.

No extra model fits are needed; this just adds statistics on top of numbers
the benchmark step was already computing and throwing away.

Two tests are reported side by side:
  - Paired t-test (parametric; assumes the per-fold differences are
    approximately normal - a shaky assumption with only 5 folds, but the
    standard default).
  - Wilcoxon signed-rank test (nonparametric; doesn't assume normality, but
    needs decent sample size to have any power - with n=5 it will often be
    unable to reject even a real, meaningful difference).

CAVEAT (report this, don't hide it): n=5 folds is a small sample for either
test. A non-significant result here means "not enough evidence to
distinguish these models at this sample size," not "these models perform
identically." Treat p-values as a sanity check on the "these are basically
tied" narrative, not as proof of it.
"""

import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats


def paired_tests(cv_raw_df: pd.DataFrame, metric: str = "R2", baseline: str | None = None):
    """
    Runs paired t-test + Wilcoxon signed-rank on every pair of models (or,
    if `baseline` is given, only baseline-vs-each-other-model) using the
    long-format DataFrame from models.cv_raw_scores_to_df().

    Returns a DataFrame with one row per comparison: mean difference,
    t-test p-value, Wilcoxon p-value, and which model "wins" the mean.
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
    """Convenience wrapper: baseline-vs-everyone, across multiple metrics, concatenated."""
    frames = [paired_tests(cv_raw_df, metric=m, baseline=baseline) for m in metrics]
    return pd.concat(frames, ignore_index=True)
