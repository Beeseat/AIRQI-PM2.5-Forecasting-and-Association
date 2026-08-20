# Beijing PM2.5: Forecasting and Pollution-Driver Analysis

## Motivation
The original project this repository redoes ([Dieselmarble/Beijing-PM2.5-Forecasting](https://github.com/Dieselmarble/Beijing-PM2.5-Forecasting)) frames Beijing air quality as a regression problem: fit Ridge/Lasso and Gaussian-kernel models on the UCI "Beijing PM2.5 Data" set (Chen, S. 2015, UCI ML Repository, https://doi.org/10.24432/C5JS49) and report error metrics. That approach quantifies forecast accuracy but does not identify what drives a pollution episode.

This redo retains the regression forecasting task and rebuilds it with a proper multi-model benchmark, correct time-series validation, engineered features, and model explainability. A second analysis, association rule mining, is added to address the driver question directly.

## Summary of results
- A 4-model benchmark (Ridge, Decision Tree, Random Forest, XGBoost) at 1-hour-ahead forecasting shows tree ensembles do not outperform Ridge regression once validated properly. Under 5-fold TimeSeriesSplit CV, Ridge (R2 0.943) outperforms both XGBoost (0.934) and Random Forest (0.929); a small XGBoost/RF edge appears only on the single chronological holdout split (R2 0.951 vs. 0.950) and is not stable across seeds or statistically distinguishable from Ridge (see [Statistical significance](#statistical-significance), [Seed robustness](#seed-robustness)).
- SHAP analysis explains why: the previous hour's PM2.5 reading dominates every other feature. At this timescale pollution behaves close to a persistence process, leaving little room for nonlinear modeling to add value.
- The forecast-horizon extension (+6h, +24h) tests whether this changes as the persistence signal weakens. It does not favor the tree ensembles: Ridge remains the top performer at +24h, and Decision Tree's R2 goes negative. See [Forecast horizon](#forecast-horizon) and [Discussion](#discussion).
- Association rule mining was used to identify weather/time conditions associated with pollution severity, independent of the regression task. Calm, humid winter conditions are associated with hazardous PM2.5 at roughly 8x the base rate.

## Dataset
UCI Beijing PM2.5 dataset, 43,824 hourly rows (Jan 2010–Dec 2014): US Embassy PM2.5 readings plus Beijing Capital Airport meteorology (dew point, temperature, pressure, wind direction, cumulated wind speed, snow and rain hours). Missing PM2.5 values are linearly interpolated.

## Exploratory data analysis
Run before any cleaning or modeling (pipeline STEP 0, `src/eda.py`), on the raw data as loaded from `PRSA_data.csv`.

- **Missingness**: PM2.5 is missing for 2,067 of 43,824 hours (4.7%), concentrated in the first two years of the dataset rather than spread evenly; some months in 2010–2011 are missing over 25% of readings, while most months from 2012 onward are missing under 2%. This is why `data_loader.clean_data` interpolates rather than drops rows outright, dropping only the small number of rows before the first valid reading. Chart: `outputs/01_eda/eda_missingness.png`.
- **PM2.5 over time**: the full hourly series shows short, sharp spikes reaching 600–1000 ug/m3 against a baseline that mostly sits under 150 ug/m3, with the spikes visibly denser in winter months. This is the pattern the association rule mining later quantifies directly. Chart: `outputs/01_eda/eda_pm25_timeseries.png`.
- **PM2.5 distribution**: heavily right-skewed, with most hours falling in the Good/Moderate AQI range and a long tail into Unhealthy and Hazardous territory. The skew is why the association rule mining's targeted pass (`min_support=0.006`) exists: a uniform support threshold tuned for the bulk of the distribution would miss the rare severe-pollution rows almost entirely. Chart: `outputs/01_eda/eda_pm25_distribution.png`.
- **Correlation with raw weather variables**: PM2.5's linear correlation with any single raw weather variable is weak (|r| under 0.25 for all of DEWP, TEMP, PRES, Iws, Is, Ir). This matters for reading the modeling results correctly: it means the strong performance of lag_1h (see [Discussion](#discussion)) is not because the raw weather features are individually uninformative substitutes for it, they are genuinely weak predictors on their own, and PM2.5's own recent history is doing essentially all of the work. Chart: `outputs/01_eda/eda_correlation_heatmap.png`.
- **Seasonality**: median and upper-quartile PM2.5 are visibly higher in winter months (Oct–Feb) than in summer, consistent with Beijing's winter-heating smog pattern and with the `DEWP=Med, SEASON=Winter, WIND_SPD=Low -> PM25=Hazardous` rule found later. Chart: `outputs/01_eda/eda_pm25_by_month.png`.
- **Wind direction**: mean PM2.5 varies close to 2x across the four wind-direction categories (calm/variable and southeasterly winds run highest, northwesterly winds run lowest), foreshadowing wind direction's role as the strongest single lift driver in the association rules. Chart: `outputs/01_eda/eda_pm25_by_wind_direction.png`.

## Approach

### 1. Forecasting (regression)
- **Cyclical temporal encoding**: hour, day-of-week, and month transformed to sin/cos pairs.
- **Lag and rolling-window features**: lag_1h, lag_2h, and 6h/24h trailing means of PM2.5.
- **4-way benchmark**: Ridge (baseline), Decision Tree, Random Forest, XGBoost.
- **TimeSeriesSplit cross-validation**: chronological folds, avoiding the lookahead bias a random-shuffle split would introduce.
- **Paired significance testing**: paired t-test and Wilcoxon signed-rank across the 5 CV folds, each model vs. Ridge. See [Statistical significance](#statistical-significance).
- **Seed-robustness check**: each base model refit across 5 seeds on the fixed holdout split. See [Seed robustness](#seed-robustness).
- **Hyperparameter tuning**: `RandomizedSearchCV`/`GridSearchCV` per model, using `TimeSeriesSplit` internally. See [Hyperparameter tuning](#hyperparameter-tuning).
- **Forecast horizon**: target shifted +6h/+24h beyond the original setup, re-run through the same CV benchmark. See [Forecast horizon](#forecast-horizon).
- **SHAP TreeExplainer**: feature-attribution analysis on the tuned XGBoost model.

### 2. Pollution-driver analysis (association rule mining)
Hourly PM2.5 changes little between consecutive readings, consistent with the SHAP result above. Framed as classification, predicting the next hour's AQI band would largely reduce to predicting the previous band. Association rule mining sidesteps this: it identifies co-occurring conditions rather than forecasting a value, so persistence is not a confound.

- PM2.5 is binned into 5 severity bands (Good, Moderate, Unhealthy, VeryUnhealthy, Hazardous). Wind direction, wind speed, temperature, pressure, dew point, season, time of day, and snow/rain are discretized into categorical items (`src/association_rules.py:discretize_for_transactions`).
- Apriori candidate generation, support pruning, and rule extraction (support, confidence, lift) are implemented directly, following the level-wise algorithm in Agrawal and Srikant (1994); no frequent-itemset library was available in this environment. See `apriori()` and `generate_rules()`.
- Two passes are run: a general pass (`min_support=0.02, min_confidence=0.4`) covering common conditions, and a lower-threshold pass (`min_support=0.006, min_confidence=0.2, max_len=4`) to surface rules for the rarer VeryUnhealthy and Hazardous bands (~14% and ~7% of rows respectively).

## Hyperparameter tuning
`src/tuning.py` runs a search per model, restricted to `X_train`/`y_train` (the holdout is never touched during search), and refits each on the full training set before a single holdout evaluation:
- **Ridge**: exhaustive grid over `alpha`.
- **Decision Tree**: exhaustive grid over `max_depth`, `min_samples_leaf`.
- **Random Forest**: `RandomizedSearchCV` (2 TimeSeriesSplit folds, 4 draws) over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
- **XGBoost**: `RandomizedSearchCV` (4 TimeSeriesSplit folds, 15 draws) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`.

All searches optimize mean CV RMSE.

**Result**: tuning does not change the overall conclusion. Tuned Random Forest is the best single model on the holdout (R2 0.953), ahead of untuned XGBoost (0.951), but the gap over tuned Ridge (0.950) remains under 0.005 R2. Tuned XGBoost (0.950) scores marginally below its untuned default (0.951): its search selected the parameters with the best mean CV RMSE across training folds, which is not guaranteed to be optimal for this specific holdout split. Decision Tree is the only model to improve meaningfully with tuning (0.935 → 0.945), consistent with the untuned default (`max_depth=10`) being comparatively unregularized.

## Statistical significance
The CV benchmark's 5 `TimeSeriesSplit` folds use an identical train/validation split across all models, so each model's fold scores are paired with every other model's. `src/significance.py` runs a paired t-test and Wilcoxon signed-rank test on these paired scores, per metric, for each model against Ridge.

**Result**: no model's difference from Ridge is significant at p<0.05 on R2 or RMSE, by either test, including Decision Tree despite its largest raw R2 gap (0.097). The one exception is Decision Tree vs. Ridge on MAE (paired t-test p=0.046; Wilcoxon does not agree). Full table: `outputs/03_significance/significance_tests_vs_ridge.csv`.

n=5 folds is a small sample for both tests; Wilcoxon's minimum attainable p-value at n=5 is 0.0625, so it cannot report p<0.05 here regardless of effect size. A non-significant result indicates insufficient evidence to distinguish the models at this sample size, not equivalence. A larger fold count or repeated CV with varied fold boundaries would increase test power.

## Seed robustness
All reported results pin `random_state=42`. With the top three models on the holdout separated by under 0.0016 R2, this ordering could reflect a stable property of each model or a favorable seed. `src/robustness.py` (pipeline STEP 4b) refits each base model across 5 seeds on the same holdout split: identical rows, split, and hyperparameters, only `random_state` varies.

**Holdout R2 across 5 seeds (0–4):**
| Algorithm | mean R2 | std | range (max-min) |
|---|---|---|---|
| Random Forest | 0.9503 | 0.0002 | 0.0005 |
| Ridge (Baseline) | 0.9497 | 0.0000 | 0.0000 |
| XGBoost (Redo) | 0.9496 | 0.0003 | 0.0009 |
| Decision Tree | 0.9317 | 0.0010 | 0.0023 |

Ridge's std is exactly 0 under the default solver, confirming the harness introduces no variance of its own. The seed-to-seed spread for the other models (0.0005–0.0009) is the same order of magnitude as the gaps between models in the single-seed holdout table, so those gaps do not constitute a reliable ranking. Averaged over seeds, the order becomes Random Forest > Ridge > XGBoost: the single-seed ordering (XGBoost > RF > Ridge) does not survive seed-averaging. Full numbers: `outputs/04_seed_robustness/seed_robustness_summary.csv`, `outputs/04_seed_robustness/seed_robustness_per_seed.csv`.

This is consistent with the significance testing above, obtained by an independent method: `significance.py` varies the split at fixed seed and finds the models statistically indistinguishable; this varies the seed at a fixed split and finds the single-seed ranking unstable. This measures variance across seeds on one fixed holdout split, not a full nested-CV estimate that varies split and seed jointly.

## Results

**5-fold TimeSeriesSplit CV (mean across folds):**
| Algorithm | MAE | RMSE | R2 |
|---|---|---|---|
| Ridge (Baseline) | 12.24 | 21.99 | 0.943 |
| XGBoost (Redo) | 12.59 | 23.76 | 0.934 |
| Random Forest | 12.60 | 24.49 | 0.929 |
| Decision Tree | 14.75 | 34.33 | 0.846 |

Full table: `outputs/02_benchmark/benchmark_results_cv.csv`. Chart: `outputs/02_benchmark/benchmark_cv_chart.png`.

**Paired significance vs. Ridge (R2), 5 CV folds:**

Chart: `outputs/03_significance/significance_r2_chart.png`. Full table: `outputs/03_significance/significance_tests_vs_ridge.csv`.

**Chronological 80/20 holdout (final fit, untuned defaults):**
| Algorithm | MAE | RMSE | R2 |
|---|---|---|---|
| XGBoost (Redo) | 11.46 | 20.72 | 0.951 |
| Random Forest | 11.47 | 20.93 | 0.950 |
| Ridge (Baseline) | 11.61 | 21.05 | 0.950 |
| Decision Tree | 12.69 | 23.99 | 0.935 |

Full table: `outputs/02_benchmark/benchmark_results_final_holdout.csv`. Chart: `outputs/02_benchmark/benchmark_holdout_chart.png`.

The top three models sit within 0.0016 R2 of each other. [Seed robustness](#seed-robustness) shows this ordering is within seed noise and reverses under seed-averaging; it should not be read as a reliable ranking.

**Seed robustness, holdout R2 (mean +/- std across 5 seeds):**

Chart: `outputs/04_seed_robustness/seed_robustness_chart.png`. Full tables: `outputs/04_seed_robustness/seed_robustness_summary.csv`, `outputs/04_seed_robustness/seed_robustness_per_seed.csv`.

**Chronological 80/20 holdout, tuned vs. untuned (see [Hyperparameter tuning](#hyperparameter-tuning)):**
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

Full table: `outputs/05_tuning/benchmark_results_holdout_tuned_vs_untuned.csv`. Chart: `outputs/05_tuning/benchmark_tuned_vs_untuned_chart.png`. Best params per model: `outputs/05_tuning/tuning_best_params.csv`.

**Top association rules:**
| Rule | Lift | Confidence |
|---|---|---|
| `WIND=NW, WIND_SPD=High -> PM25=Good` | 2.64 | 0.75 |
| `DEWP=High, WIND=SE -> PM25=Unhealthy` | 1.56 | 0.43 |
| `DEWP=Med, SEASON=Winter, WIND_SPD=Low -> PM25=Hazardous` | 8.20 | 0.58 |

Full rule tables: `outputs/08_association_rules/association_rules_pm25.csv`, `outputs/08_association_rules/association_rules_severe_pm25.csv`. Charts: `outputs/08_association_rules/association_rules_top_lift.png`, `outputs/08_association_rules/association_rules_severe_top_lift.png`.

## Forecast horizon
Results above are for the original 1-hour-ahead setup (`horizon=0` in `src/data_loader.py`, the default). `engineer_features` accepts a `horizon` parameter that shifts the target forward by that many extra hours; lag/rolling/weather features remain computed strictly from information at or before row t, so the comparison across horizons isolates the effect of forecast distance. Pipeline STEP 8 sweeps `horizon in (0, 6, 24)` through the same 5-fold CV benchmark used in STEP 2.

**5-fold TimeSeriesSplit CV, by horizon (R2):**
| Algorithm | +0h | +6h | +24h |
|---|---|---|---|
| Ridge (Baseline) | 0.943 | 0.573 | 0.180 |
| XGBoost (Redo) | 0.933 | 0.571 | 0.093 |
| Random Forest | 0.929 | 0.575 | 0.097 |
| Decision Tree | 0.846 | 0.376 | −0.188 |

Full table: `outputs/06_forecast_horizon/benchmark_results_by_horizon.csv`. Charts: `outputs/06_forecast_horizon/horizon_comparison_r2.png`, `outputs/06_forecast_horizon/horizon_comparison_rmse.png`.

The tree ensembles do not gain ground on Ridge as the horizon extends; the reverse occurs. At +6h all three models are within statistical noise of each other. At +24h Ridge is the strongest model, Random Forest and XGBoost trail by roughly 0.08–0.09 R2, and Decision Tree's R2 is negative (worse than predicting the mean). STEP 8b reruns SHAP on XGBoost at +24h (`outputs/07_explainability_shap/feature_importance_shap_horizon24.png`) for comparison against the +0h SHAP plot.

## Discussion
The regression and association analyses were run independently and agree on the underlying physics, which serves as a cross-check. SHAP ranks lag features highest for the regression model; among the remaining features, wind and dew point/temperature rank highest. Association mining, using a different mechanism (co-occurrence counting rather than gradient-boosted trees), independently identifies the same variables (wind direction, wind speed, dew point, season) as the strongest non-persistence drivers of pollution category. The strongest rule found, `DEWP=Med, SEASON=Winter, WIND_SPD=Low -> PM25=Hazardous` (lift 8.2), matches Beijing's documented winter-heating smog pattern: cold, calm, humid conditions trap emissions near ground level.

### Why linear regression is competitive with tree ensembles here
The result holds consistently across the CV benchmark, significance testing, seed robustness, and the horizon sweep, so it merits explanation rather than being treated as an artifact.

The predictive signal in this dataset is concentrated almost entirely in lag_1h, and the relationship between consecutive PM2.5 readings is close to linear: the series behaves as a slowly drifting, strongly autocorrelated process rather than one governed by threshold effects or feature interactions. Ridge is well matched to that structure. Tree ensembles approximate a linear relationship with piecewise-constant splits, which is a less efficient use of model capacity than fitting a slope directly, and their added flexibility increases variance without a corresponding reduction in bias when the true relationship is close to linear. This is visible directly in the seed-robustness results: Ridge's holdout score is invariant to the random seed, while XGBoost's ranking relative to Ridge changes across seeds.

The horizon sweep is consistent with this account. As the horizon extends, the autocorrelation-driven signal weakens and the prediction problem becomes noisier. A regularized linear model degrades gradually under these conditions; the tree ensembles degrade faster, and Decision Tree degrades enough to underperform a constant predictor. Trees cannot extrapolate outside the range of target values observed during training, and with less signal to constrain their splits, they are more prone to fitting noise in the training folds.

This pattern is not unique to this dataset. Zeng et al. (2023, AAAI), in "Are Transformers Effective for Time Series Forecasting?", report that a single linear layer matches or outperforms considerably more complex Transformer-based architectures across nine standard long-term forecasting benchmarks, and attribute this to the trend/periodicity structure such series often have, structure a linear model can capture directly. The present result is a smaller-scale instance of the same phenomenon: added model complexity does not help, and can hurt, when the underlying relationship is well approximated by a linear one.

## How to run
```bash
pip install -r requirements.txt
python main.py
```
Takes a few minutes end-to-end, mostly the hyperparameter tuning step. Every CSV result table in `outputs/` has a matching PNG chart generated in the same pipeline step, so the numbers can be read visually instead of scanning raw CSVs. Results are written into eight numbered subfolders under `outputs/`, one per pipeline stage, in the order the analysis narrative above follows (`01_eda/` through `08_association_rules/`). See `outputs/INDEX.md` for the full folder-by-folder map, and `FEATURES_AND_TECHNIQUES.md` (project root) for a plain-language glossary of every technique used.
