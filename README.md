# Beijing PM2.5 - Forecasting and Pollution-Driver Analysis

## Motivation
Beijing PM2.5 readings regularly spike into the hazardous range, and residents, schools, and public health agencies rely on short-notice air quality information to make decisions. The original project this repo redoes ([Dieselmarble/Beijing-PM2.5-Forecasting](https://github.com/Dieselmarble/Beijing-PM2.5-Forecasting)) treats this as a regression problem: fit Ridge/Lasso and Gaussian-kernel models on the UCI "Beijing PM2.5 Data" set (Chen, S. 2015, UCI ML Repository, https://doi.org/10.24432/C5JS49) and report error metrics. That measures how close the model gets to the right number, but it does not identify what actually drives a pollution episode.

This redo keeps the regression forecasting task and rebuilds it with a proper benchmark, correct time-series validation, real feature engineering, and real explainability. It then adds a second analysis, association rule mining, aimed at the driver question the original project does not address.

## Summary of results
- A correctly validated 4-model benchmark shows tree ensembles barely outperform a plain Ridge regression at 1-hour-ahead forecasting (R2 0.951 vs 0.950 on holdout, untuned).
- **Paired significance testing across the 5 CV folds shows none of these differences are statistically distinguishable from noise** (paired t-test and Wilcoxon signed-rank, both p > 0.05 for every model vs. Ridge on R2 and RMSE - see [Statistical significance](#statistical-significance)). With only 5 folds this is a low-power test, so it isn't proof the models are equal, but it does mean the "XGBoost/RF barely beat Ridge" framing should be read as "indistinguishable at this sample size," not as a confirmed small effect.
- SHAP analysis shows why: the previous hour's PM2.5 reading dominates every other feature. Pollution behaves close to a slowly drifting process at this timescale, which leaves little room for nonlinear modeling to add value at a 1-hour horizon.
- Hyperparameter tuning (TimeSeriesSplit-based random/grid search per model, see [Hyperparameter tuning](#hyperparameter-tuning)) does not change this conclusion. All four tuned models still land within R2 0.945-0.953 on the holdout - a tuned Random Forest edges out everything else, but the spread across all 8 tuned/untuned models is under 0.01 R2, well inside what the significance testing above says is noise-level for this dataset.
- This result motivated the second half of the project. Since hour-to-hour prediction has limited headroom, association rule mining was used to identify which weather and time conditions accompany good versus hazardous air. It found that calm winter conditions are associated with hazardous PM2.5 at roughly 8 times the base rate, a pattern the regression analysis does not surface directly.

## Dataset
UCI Beijing PM2.5 dataset, 43,824 hourly rows (Jan 2010 to Dec 2014): US Embassy PM2.5 readings plus Beijing Capital Airport meteorology (dew point, temperature, pressure, wind direction, cumulated wind speed, snow and rain hours). Missing PM2.5 values are linearly interpolated.

## Approach

### 1. Forecasting (regression)
- **Cyclical temporal encoding**: hour, day-of-week, and month are transformed to sin/cos pairs so that, for example, hour 23 and hour 0 are treated as adjacent.
- **Lag and rolling-window features**: lag_1h, lag_2h, and 6h/24h trailing means of PM2.5.
- **4-way benchmark**: Ridge (baseline), Decision Tree, Random Forest, and XGBoost.
- **TimeSeriesSplit cross-validation**: chronological folds only, avoiding the random-shuffle leakage a naive redo would introduce on time-series data.
- **Paired significance testing**: paired t-test + Wilcoxon signed-rank across the 5 CV folds, comparing each model to the Ridge baseline. See [Statistical significance](#statistical-significance).
- **Hyperparameter tuning**: `RandomizedSearchCV`/`GridSearchCV` per model, using `TimeSeriesSplit` internally so the search itself can't leak future rows into a validation fold. See [Hyperparameter tuning](#hyperparameter-tuning) below.
- **SHAP TreeExplainer**: feature-attribution analysis on the final (tuned) XGBoost model, in place of simple Gini/gain importances.

### 2. Pollution-driver analysis (association rule mining)
Hourly PM2.5 changes little between consecutive readings, which is consistent with the SHAP result above. Framed as classification, predicting next hour's AQI band would largely reduce to a persistence problem: predicting the same band as the previous hour. Association rule mining avoids this issue, since it does not forecast anything. It identifies which conditions co-occur, so hourly resolution is not a limitation here the way it is for classification.

- PM2.5 is binned into 5 severity bands (Good, Moderate, Unhealthy, VeryUnhealthy, Hazardous). Wind direction, wind speed, temperature, pressure, dew point, season, time of day, and snow/rain are each discretized into categorical items (`src/association_rules.py:discretize_for_transactions`).
- No frequent-itemset library (e.g. mlxtend) was available in this environment, so Apriori candidate generation, support pruning, and rule extraction (support, confidence, lift) are implemented directly, following the level-wise algorithm described in Agrawal and Srikant (1994). See `apriori()` and `generate_rules()`.
- Two passes are run: a general pass (`min_support=0.02, min_confidence=0.4`) covering common conditions, and a lower-threshold pass (`min_support=0.006, min_confidence=0.2, max_len=4`) to surface rules for the rarer VeryUnhealthy and Hazardous bands (roughly 14% and 7% of rows respectively), which the general threshold mostly misses because those bands occur less often.

## Hyperparameter tuning
All four models use fixed default-ish settings in the main benchmark above. To check whether that understates any model's ceiling, `src/tuning.py` runs a search per model, restricted to `X_train`/`y_train` only (the final holdout is never touched during search), and refits each on the full training set before a single evaluation on the holdout:
- **Ridge**: exhaustive grid over `alpha`.
- **Decision Tree**: exhaustive grid over `max_depth`, `min_samples_leaf`.
- **Random Forest**: `RandomizedSearchCV` (2 TimeSeriesSplit folds, 4 draws) over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`. Kept small on purpose - refitting a few hundred trees repeatedly on ~35k rows is by far the most expensive search here.
- **XGBoost**: `RandomizedSearchCV` (4 TimeSeriesSplit folds, 15 draws) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight` - the largest budget of the four, since it's the model the "barely beats Ridge" headline claim rests on.

All searches optimize mean CV RMSE (not R2 directly, though the two agree on ranking here).

**Result**: tuning does not overturn the "limited headroom" conclusion - if anything it reinforces it. Tuned Random Forest becomes the best single model on the holdout (R2 0.953, beating untuned XGBoost's 0.951), but the gap over tuned Ridge (0.950) is still under 0.005 R2. Notably, tuned XGBoost (0.950) is very slightly *worse* on the holdout than the untuned default (0.951) - its search picked the params with the best mean CV RMSE across the training folds, which isn't guaranteed to be the best fit for this one specific holdout split. That's expected variance at this scale, not a bug, and it's itself a small illustration of how little separates these models once lag_1h is doing most of the work. Only Decision Tree improves meaningfully with tuning (0.935 -> 0.945), consistent with it being the most under-regularized model at default settings (`max_depth=10` untuned vs. a much shallower tuned optimum).

## Statistical significance
The CV benchmark's 5 `TimeSeriesSplit` folds use the identical train/validation split for every model, so each model's fold-1 score is naturally *paired* with every other model's fold-1 score (same for folds 2-5). `src/significance.py` runs a paired t-test and a Wilcoxon signed-rank test on those 5 paired scores, per metric, for each model against the Ridge baseline. This adds zero extra model fits - it's pure statistics on numbers the benchmark step was already computing and discarding.

**Result**: no model's difference from Ridge is significant at p<0.05 on R2 or RMSE, by either test - including Decision Tree, despite it having by far the largest raw R2 gap (0.097). The one exception is Decision Tree vs. Ridge on MAE (paired t-test p=0.046, barely under 0.05; Wilcoxon does not agree). Full table: `outputs/significance_tests_vs_ridge.csv`.

**Important caveat - don't over-read this**: n=5 folds is a small sample for either test, and Wilcoxon in particular has a hard floor on how low its p-value can go with only 5 pairs (its minimum possible p-value at n=5 is 0.0625, so it can *never* report p<0.05 here regardless of effect size - this is a property of the test, not evidence of "no effect"). A non-significant result means "not enough evidence, at this sample size, to distinguish these models," not "these models perform identically." Read it as a sanity check on the "these models are all close" narrative from the main results table, not as formal proof of it. A larger number of folds (at the cost of smaller, noisier individual folds) or repeated CV with different fold boundaries would give the tests more power if a future version of this project wanted a firmer answer.

## Results

**5-fold TimeSeriesSplit CV (mean across folds):**
| Algorithm | MAE | RMSE | R2 |
|---|---|---|---|
| Ridge (Baseline) | 12.24 | 21.99 | 0.943 |
| XGBoost (Redo) | 12.59 | 23.76 | 0.934 |
| Random Forest | 12.60 | 24.49 | 0.929 |
| Decision Tree | 14.75 | 34.33 | 0.846 |

**Chronological 80/20 holdout (final fit, untuned defaults):**
| Algorithm | MAE | RMSE | R2 |
|---|---|---|---|
| XGBoost (Redo) | 11.46 | 20.72 | 0.951 |
| Random Forest | 11.47 | 20.93 | 0.950 |
| Ridge (Baseline) | 11.61 | 21.05 | 0.950 |
| Decision Tree | 12.69 | 23.99 | 0.935 |

**Chronological 80/20 holdout, tuned vs untuned (see [Hyperparameter tuning](#hyperparameter-tuning)):**
| Algorithm | MAE | RMSE | R2 |
|---|---|---|---|
| Random Forest [Tuned] | 11.22 | 20.34 | 0.953 |
| XGBoost (Redo) | 11.46 | 20.72 | 0.951 |
| Random Forest | 11.47 | 20.93 | 0.950 |
| XGBoost (Redo) [Tuned] | 11.50 | 20.98 | 0.950 |
| Ridge (Baseline) | 11.61 | 21.05 | 0.950 |
| Ridge (Baseline) [Tuned] | 11.60 | 21.05 | 0.950 |
| Decision Tree [Tuned] | 12.27 | 22.05 | 0.945 |
| Decision Tree | 12.69 | 23.99 | 0.935 |

Full table: `outputs/benchmark_results_holdout_tuned_vs_untuned.csv`. Best params per model: `outputs/tuning_best_params.csv`.

**Top association rules found:**
| Rule | Lift | Confidence |
|---|---|---|
| `WIND=NW, WIND_SPD=High -> PM25=Good` | 2.64 | 0.75 |
| `DEWP=High, WIND=SE -> PM25=Unhealthy` | 1.56 | 0.43 |
| `DEWP=Med, SEASON=Winter, WIND_SPD=Low -> PM25=Hazardous` | 8.20 | 0.58 |

Full rule tables: `outputs/association_rules_pm25.csv` and `outputs/association_rules_severe_pm25.csv`. Charts: `outputs/association_rules_top_lift.png` and `outputs/association_rules_severe_top_lift.png`.

## Discussion
The regression analysis and the association analysis were run independently but agree on the underlying physics, which serves as a useful cross-check.

SHAP ranks lag features highest for the regression model, but among the non-lag weather features, wind and dew point/temperature also rank highly. Association mining, using a completely different mechanism (co-occurrence counting rather than gradient-boosted trees), independently identifies the same variables, wind direction, wind speed, dew point, and season, as the strongest drivers of pollution category.

The strongest rule found, `DEWP=Med, SEASON=Winter, WIND_SPD=Low -> PM25=Hazardous` (lift 8.2), matches Beijing's documented winter-heating smog pattern: cold, calm, humid conditions trap emissions near ground level. The association rules make this pattern explicit in a way the regression model's feature weights do not.

XGBoost's improvement over the Ridge baseline is small, which follows directly from lag_1h's dominance in the SHAP analysis. Tuning (above) doesn't change this - it's not an artifact of unfavorable defaults. It is also the reason the project moved to association rules for the driver-analysis component.

## Limitations and future work
- **Forecast horizon**: all results above are for a 1-hour-ahead forecast, where lag_1h dominates. Testing t+6 and t+24 horizons would likely show a larger tree-ensemble advantage, since the persistence shortcut weakens as the horizon increases.
- **Tuning budget**: the Random Forest and XGBoost searches use a small number of random-search draws and (for Random Forest) only 2 CV folds, traded off deliberately against runtime - see [Hyperparameter tuning](#hyperparameter-tuning). A larger budget (more draws, more folds, wider ranges) might shift the tuned numbers slightly, though the overall "tight clustering around R2~0.95" pattern is unlikely to change given how dominant lag_1h is.
- **Statistical testing**: paired significance testing was added (see [Statistical significance](#statistical-significance)) and found no model significantly different from Ridge - but with only 5 CV folds, this test has low power and can't fully resolve whether the small observed differences are real. Repeated/nested CV with more folds would give a firmer answer.
- **Seed sensitivity**: results above use a single `random_state=42` throughout. A quick check across 7 seeds (untuned models, same holdout split) shows Ridge is exactly deterministic (no stochastic component), Random Forest is very stable (R2 range 0.0005), but XGBoost (range 0.0023) and Decision Tree (range 0.0039) have real seed-to-seed variance - in XGBoost's case, larger than some of the model-vs-model gaps reported in the results table. Reporting mean ± std R2 across several seeds, rather than a single point estimate, would be a natural next step.
- **Association rule redundancy**: some mined rules are close variants of each other, differing by one additional weather condition. A maximal or closed-itemset filter would reduce this redundancy.

## How to run
```bash
pip install -r requirements.txt
python main.py
```
Takes a few minutes end-to-end, mostly the hyperparameter tuning step (Random Forest search in particular). Outputs (CV benchmark CSV, significance-test CSV, holdout benchmark CSV, tuned-vs-untuned holdout CSV, tuning best-params CSV, actual-vs-predicted plot, SHAP summary plot, association rule CSVs and charts) are written to `outputs/`.