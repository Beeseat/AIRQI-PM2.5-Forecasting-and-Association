# Beijing PM2.5 - Forecasting and Pollution-Driver Analysis

Redo of [Dieselmarble/Beijing-PM2.5-Forecasting](https://github.com/Dieselmarble/Beijing-PM2.5-Forecasting), built for a Data Mining course project. The original project forecasts Beijing PM2.5 levels with Ridge/Lasso regression. This version keeps the forecasting task but adds a proper 4-model benchmark, hyperparameter tuning, SHAP explainability, and a second analysis using association rule mining to find out *what actually drives* a pollution spike.

## Dataset

UCI "Beijing PM2.5 Data" - about 43,800 hourly readings (2010-2014) combining PM2.5 levels with weather data (temperature, dew point, pressure, wind, rain/snow).

Chen, S. (2015). Beijing PM2.5 Data. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49

![PM2.5 over time](outputs/01_eda/eda_pm25_timeseries.png)

## Part 1: Forecasting

Four models were benchmarked on 1-hour-ahead PM2.5 forecasts: Ridge as the baseline, Decision Tree, Random Forest, and XGBoost. Validation uses `TimeSeriesSplit` instead of a random shuffle, since shuffling would leak future rows into training on time-series data.

![Model comparison (5-fold CV)](outputs/02_benchmark/benchmark_cv_chart.png)

On the 5-fold CV benchmark, Ridge comes out ahead of every tree model on R2, RMSE, and MAE - the more expressive tree ensembles don't actually help here. A single chronological holdout split tells a slightly different story, with Random Forest and XGBoost narrowly ahead of Ridge instead:

![Model comparison (holdout)](outputs/02_benchmark/benchmark_holdout_chart.png)

Either way the gaps are small. Paired significance tests (t-test and Wilcoxon) across CV folds confirm none of these differences are distinguishable from noise. Hyperparameter tuning doesn't change that.

SHAP shows why: the previous hour's PM2.5 reading dominates every other feature. Pollution barely moves hour to hour, so there's not much signal left for a more expressive model to use.

![SHAP feature importance](outputs/07_explainability_shap/feature_importance_shap.png)

Forecasts were also run at 6-hour and 24-hour horizons, where the previous reading is less useful and the models have more room to separate.

![Forecast horizon comparison](outputs/06_forecast_horizon/horizon_comparison_r2.png)

The gap between models does widen at longer horizons, once the "copy last hour" shortcut stops working.

This isn't unique to this dataset - Zeng et al. (2023, AAAI) found a single linear layer matches or beats far more complex Transformer architectures across nine long-term forecasting benchmarks, since a linear model can capture the trend/periodicity structure many series already have. Same story here, just smaller scale: extra model complexity doesn't help once the underlying relationship is close to linear.

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317

## Part 2: What Drives Pollution Spikes?

Since hour-to-hour prediction plateaus fast, the second half asks a different question: which weather and time conditions co-occur with bad air? This uses association rule mining (Apriori, implemented from scratch) over PM2.5 severity bands and discretized weather features, ranked by **lift** - how much more likely an outcome is given a condition, versus its baseline rate.

![Top association rules](outputs/08_association_rules/association_rules_top_lift.png)

Strong wind is the top predictor of good air across the board. Good-air rules dominate the overall top-lift ranking, so hazardous readings - a rarer band - need their own pass with a lower support threshold to show up.

![Rules for severe pollution](outputs/08_association_rules/association_rules_severe_top_lift.png)

Calm, humid winter conditions are linked to hazardous PM2.5 at roughly **8x** the base rate - Beijing's winter smog pattern: cold, windless air trapping emissions close to the ground.

## Limitations

- **Statistical testing** - the paired significance tests found no model significantly different from Ridge, but with only 5 CV folds this test has low power and can't fully resolve whether the small observed differences are real. Repeated or nested CV with more folds would give a firmer answer.
- **Tuning budget** - the Random Forest and XGBoost searches use a small number of random-search draws (Random Forest only 2 CV folds), traded off deliberately against runtime. A larger budget might shift the tuned numbers slightly, though the "clustered near R2~0.95" pattern is unlikely to change given how dominant lag_1h is.
- **Seed sensitivity** - results above use `random_state=42` throughout. Checking 5 seeds (base models, same holdout split) shows Ridge is exactly deterministic (R2 range 0, no stochastic component), Random Forest is very stable (range 0.0005), XGBoost has more spread (range 0.0009), and Decision Tree the most (range 0.0023) - in Decision Tree's case, comparable to some of the model-vs-model gaps in the results table. Full breakdown in `outputs/04_seed_robustness/`.
- **Association rule redundancy** - some mined rules are close variants of each other, differing by one additional weather condition. A maximal or closed-itemset filter would reduce this.

## Project Structure

```
data/            raw dataset
src/             all analysis code, split by stage
outputs/         charts and result tables, one folder per stage
main.py          runs the full pipeline end to end
```

## Running it

```
pip install -r requirements.txt
python main.py
```

Takes a few minutes, mostly for hyperparameter tuning. All charts and tables get written to `outputs/`.
