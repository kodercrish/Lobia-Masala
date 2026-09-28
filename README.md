# Lobia Masala: Hotel Property Value Prediction

> Team **Lobia Masala**'s entry for a Kaggle-style regression challenge: predicting hotel property values from 79 structural, location and quality features.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189FDD)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9ACD32)

## 📌 Overview

| | |
|---|---|
| **Task** | Supervised regression: predict `HotelValue` for each property |
| **Data** | `train.csv`: 1,200 rows × 81 columns (Id + 79 features + target). `test.csv`: 260 rows. `sample_submission.csv`: `Id,HotelValue` |
| **Features** | 43 categorical and 36 numeric: zoning, district, land and plot shape, construction and renovation year, quality and condition ratings, floor, basement, parking and porch areas, pool, sale type, and more |
| **Target** | `HotelValue`: mean ≈ 181.7k, median 165k, range 34.9k to 745k (right-skewed) |
| **Metric** | RMSE on the original price scale (also tracked: R²) |

## 🧪 Approach

- **Target transform:** `log1p(HotelValue)` for training, with `expm1` applied back for predictions.
- **Outlier removal:** trim the target's 0.1% and 99.9% quantiles, drop `UsableArea ≥ 4000`, and drop low-quality/large-area anomalies. This removes 6 rows.
- **Sparse columns dropped:** `PoolQuality` (≈100% missing), `ExtraFacility` (96%), `ServiceLaneType` (94%), `BoundaryFence` (80%), plus low-signal columns such as `LowQualityArea` and `BasementHalfBaths`.
- **Feature engineering:**
  - `HouseAge`, `YearsSinceModification`, and the interaction `QualityArea = OverallQuality × UsableArea`
  - `TotalBasementScore` / `BasementAvgQuality` (finish-type-weighted basement area), `TotalPoolScore`, `TotalPorchArea`
  - Ordinal encodings for parking quality, condition and finish, and for property functionality
  - `log1p` on skewed areas (`LandArea`, `BasementTotalSF`, `ParkingArea`, …)
  - Multicollinearity pruning (e.g. keep `ParkingCapacity`, drop `ParkingArea`)
- **Pipelines:** `ColumnTransformer` with median/most-frequent imputation, `StandardScaler` and `OneHotEncoder`.
- **Tuning:** `BayesSearchCV` (scikit-optimize) and `GridSearchCV` with K-Fold, 75/25, or 85/15 hold-out validation.
- **EDA:** correlation heatmap of the top features, plus scatter and box plots for every feature after preprocessing.

## 📊 Results

The notebook ([`Untitled.ipynb`](Untitled.ipynb)) printed the numbers below. Most are **training-set** (in-sample) scores, so they flatter the more flexible models. Kaggle leaderboard scores were **not recorded** in the repo.

| Model | Submission file | RMSE | R² | Eval set |
|---|---|---:|---:|---|
| Lasso + polynomial (raw data) | `lasso_poly_submission.csv` | 22,876.82 | 0.9131 | train |
| Lasso (cleaned data) | `lasso_submission_cleaned.csv` | 17,477.56 | 0.9453 | train |
| ElasticNetCV (converged to L1 ratio 1.0) | `elasticnet_submission_cleaned.csv` | 17,477.56 | 0.9453 | train |
| Bayesian Ridge | `bayesianridge_submission_cleaned.csv` | 16,770.82 | 0.9497 | train |
| Lasso + polynomial (engineered features) | `lasso_poly_submission_final.csv` | 16,886.31 | not recorded | train |
| **Linear Regression (marked as the team's best in the notebook)** | `linear_submission_cleaned.csv` | **15,318.01** | **0.9580** | train |
| Ridge, Bayesian-optimised | same run | 16,685.94 | 0.9502 | train |
| Ridge + weighted quality index | `bayesian_weighted_quality_test1.csv` | 17,424.57 | not recorded | train |
| Ridge + targeted imputation | `bayesian_targeted_imputation_test2.csv` | 16,704.59 | not recorded | train |
| Ridge + multicollinearity pruning | `bayesian_pruning_final.csv` / `bayesian_6_fold_final.csv` | 17,151.80 | 0.9473 | train |
| Linear, parking-capacity variant | `linear_selfMod2Parking_submission_cleaned.csv` | 16,246.52 | 0.9528 | train |
| Linear, baseline | same run as below | 22,724.25 | 0.9263 | 15% validation |
| **Ridge, Bayesian-optimised** | `linear+bayesian/linear&Bayesian.csv` | **20,819.77** | **0.9381** | 15% validation |
| XGBoost | `xgboost_submission_final.csv` | 11,267.07 | 0.9773 | train |
| XGBoost (regularised) | `xgboost_submission_regularized.csv` | 18,511.73 | 0.9387 | train |
| Random Forest (tuned) | `random_forest_optimized_test3.csv` | 9,477.91 | not recorded | train |
| Random Forest (baseline) | `random_forest_submission.csv` | not recorded | not recorded | n/a |
| AdaBoost, KNN + PCA, LightGBM, Bayesian Ridge (conjugate priors), RF/XGBoost + Bayesian search | see scripts | not recorded | not recorded | n/a |

The only held-out score is the 15% validation run: Ridge with Bayesian search beat plain linear regression, with RMSE 20.8k against 22.7k.

## 🗂️ Repository Structure

```
.
├── Hotel-Property-Value-Dataset/   # train.csv, test.csv, sample_submission.csv
├── Untitled.ipynb                  # Main experiment notebook (all models above, with outputs)
├── Correlation/                    # Correlation analysis notebook/script + top-feature heatmap
├── OutlierDetection/               # Outlier visualisation script + per-feature scatter/box plots
├── linear+bayesian/                # Linear/Ridge + BayesSearchCV variants (log, no-log, LOOCV)
├── Linear+Grid_allTrain/           # Linear/Ridge + GridSearchCV
├── Bayesian+ConjugatePriors/       # BayesianRidge (conjugate-prior) model
├── laaso+bayesian/                 # LightGBM + Bayesian search
├── RF+GridSearch/, RF+bayesian/    # Random Forest tuned by grid / Bayesian search
├── XGBoost+Bayesian/               # XGBoost + Bayesian search (full-train and split variants)
├── adaboost+bayesian/              # AdaBoost (decision-tree base) + tuning
├── KNN+PCA/                        # PCA → KNN regressor + grid search
└── *.csv                           # Kaggle submission files from each experiment
```

## 🚀 Getting Started

```bash
git clone https://github.com/kodercrish/Lobia-Masala.git
cd Lobia-Masala

# Python 3.12 recommended (scikit-learn >= 1.4 is needed for root_mean_squared_error)
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy "scikit-learn>=1.4" scikit-optimize xgboost lightgbm matplotlib seaborn jupyter

jupyter notebook Untitled.ipynb           # main experiments
python "XGBoost+Bayesian/XGBoost+Bayesian(Split).py"   # or run any model script
```

Run scripts from the repo root, because they load data from `Hotel-Property-Value-Dataset/`. The `Correlation` and `OutlierDetection` scripts prompt for a folder to save plots in. `CorrelationMatrix.ipynb` was written for Google Colab.

## 👥 Contributors

- **Krish Patel**: [@kodercrish](https://github.com/kodercrish)
- [@sidnovates](https://github.com/sidnovates)
