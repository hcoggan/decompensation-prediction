# Decompensation Prediction

Code for predicting clinical decompensation in initially low-acuity pediatric emergency department (ED) visits, and for simulating the downstream effects of earlier detection on ED length of stay (LOS) and crowding.

## Overview

Patients triaged as low-acuity (triage acuity 3–5) who go on to deteriorate — developing new hypoxia or hypotension, requiring intubation or vasopressors, being transferred to the ICU, or dying — are a small but clinically important subgroup. This project:

1. Builds a cohort of pediatric ED visits that were initially stable at triage.
2. Trains and compares classifiers (Random Forest, LightGBM, XGBoost, CatBoost) to predict subsequent decompensation from information available at or shortly after triage.
3. Evaluates the selected model (XGBoost, AUROC ≈ 0.8) using ROC/PR curves, threshold–sensitivity tradeoffs, SHAP values, and partial dependence plots.
4. Simulates how using the model to "uptriage" at-risk patients would affect ED length of stay and crowding, at a range of possible sensitivity thresholds.

## Repository structure

| File | Description |
|---|---|
| `initial-preprocessing.r` | Loads raw EHR extracts, links visit-level demographic, disposition, and triage data by MRN/DOB/arrival time, and performs initial cleaning and identifier resolution. |
| `final-preprocessing.ipynb` | Links continuous covariates (age, triage vitals, weight, etc.) onto the epidemiologically preprocessed visit-level dataset. |
| `predict-decompensation.ipynb` | Defines the decompensation outcome and initially-stable cohort, engineers predictors, trains/tunes the candidate classifiers, and evaluates the selected model (ROC/PR curves, threshold analysis, SHAP, partial dependence). |
| `model-deployment.ipynb` | Models ED length of stay as a function of crowding and visit type, and simulates the effect of using the trained model (at various sensitivity thresholds) on ED LOS and crowding. |
| `models-and-results.joblib` | Saved fitted models and evaluation results from `predict-decompensation.ipynb`. |
| `catboost_info/` | Training logs and diagnostics automatically written out by CatBoost during model fitting. |
| `figures/` | Output figures (ROC/PR curves, SHAP beeswarm plot, partial dependence plots, crowding/switching effect heatmaps, LOS simulation plots). |
| `LICENSE` | Apache License 2.0. |

## Requirements

**Python** (see notebook imports for exact packages used): `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `joblib`, `statsmodels`, `matplotlib`, `seaborn`, `plotnine`.

**R**: `lubridate`, `dplyr`, `data.table`, `tidyverse`-adjacent packages (`tibble`, `stringr`, `forcats`, `broom`), `survival`, `ggsurvfit`, `tidycmprsk`, `MatchIt`, `cobalt`, `exact2x2`, `xgboost`, `glmnet`, `pROC`, `yardstick`, `comorbidity`, `zipcodeR`, `geosphere`, `janitor`, `patchwork`, `cowplot`, `ggridges`, `ggsignif`, `gtsummary`, `knitr`, `testit`, `httpgd`.

## Data

The underlying EHR extracts are not included in this repository due to patient privacy restrictions. Scripts and notebooks expect these raw and intermediate files at paths defined near the top of each file (`loadpath`/`load_filepath`, `savepath`/`save_filepath`, `raw_filepath`) — update these to point at your own data before running.

## Usage

Run in the following order, updating file paths as needed at each step:

1. `initial-preprocessing.r`
2. `final-preprocessing.ipynb`
3. `predict-decompensation.ipynb`
4. `model-deployment.ipynb`

## License

Apache License 2.0 (see `LICENSE`).
