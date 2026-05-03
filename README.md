## diabetes-prediction-lr-rf
Comparing Logistic Regression and Random Forest for diabetes prediction using the Pima Indians Diabetes Dataset.
## Diabetes Prediction: Logistic Regression vs Random Forest

This repository contains the Python implementation supporting the final-year essay:

## Diabetes Prediction: Logistic Regression vs Random Forest

The written essay investigates diabetes outcome classification using the Pima Indians Diabetes Dataset. The main aim is to compare Logistic Regression and Random Forest for binary diabetes classification, while also examining whether variables with stronger pre-modelling distributional separation are assigned greater importance by the fitted classification models.

This repository is provided as supporting material for the essay. It contains the code used to generate the preprocessing summaries, exploratory analysis, model results, figures, and result tables discussed in the report.

---

## Relation to the Essay

The code in this repository corresponds to the main analytical stages of the essay.

---

## 1. Data Preprocessing

Invalid zero values in clinical variables are identified and treated as missing values.

The clinical variables treated for invalid zero values are:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

These zero values are treated as invalid because a value of zero is not clinically plausible for these measurements. Zero values in `Pregnancies` are retained because zero pregnancies is a meaningful value.

The script produces summary tables for:

- dataset structure;
- class distribution;
- invalid zero counts;
- invalid-zero rates by outcome group;
- median replacement values for exploratory summaries.

For exploratory summaries, invalid zero values are recoded and replaced using median values.

For predictive modelling, invalid zero values are first recoded as missing values, and median imputation is then performed inside the modelling pipelines. The imputation values are fitted on the training data only and then applied to the test data to reduce the risk of data leakage.

---

## 2. Exploratory Data Analysis

The exploratory analysis examines the structure of the dataset before model fitting.

This includes:

- class balance between diabetic and non-diabetic cases;
- class-conditional feature distributions;
- missingness indicator analysis for invalid zero values;
- correlation patterns between variables;
- comparison of stronger and weaker predictors.

The purpose of this stage is to understand whether the raw data already suggests which variables may be useful for classification.

The script generates:

- class-conditional histograms for Glucose and BloodPressure;
- a correlation heatmap;
- a class distribution table saved as a CSV file.

The class distribution is reported as a CSV table rather than as a separate figure.

---

## 3. Distributional Divergence Analysis

The essay investigates whether variables that separate diabetic and non-diabetic patients more clearly before modelling are also more important in the fitted models.

To support this analysis, the code calculates:

- Jensen-Shannon divergence;
- Kolmogorov-Smirnov statistic.

These measures are used to compare the class-conditional distributions of each predictor. The resulting tables are then used to compare pre-modelling distributional evidence with model-based importance values.

---

## 4. Model Fitting

Two supervised classification models are used:

- Logistic Regression;
- Random Forest.

---

### 4.1 Logistic Regression

Logistic Regression is used as an interpretable probabilistic baseline. Predictors are standardised, and balanced class weights are used to account for class imbalance.

The Logistic Regression pipeline uses:

- median imputation;
- feature standardisation;
- balanced class weights;
- increased maximum optimisation iterations.

The script outputs:

- Logistic Regression performance metrics;
- Logistic Regression odds ratios.

Because the Logistic Regression predictors are standardised before model fitting, the reported odds ratios correspond to a one-standard-deviation increase in each predictor, holding the other predictors constant.

---

### 4.2 Random Forest

Random Forest is used as a flexible nonlinear model.

The Random Forest model uses:

- 500 trees;
- maximum tree depth of 8;
- square-root feature sampling;
- minimum split size of 5;
- balanced subsampling through `class_weight="balanced_subsample"`;
- feature importance analysis.

Random Forest is trained on imputed but unstandardised predictors, because tree-based models do not require feature scaling in the same way as coefficient-based models.

The script outputs:

- Random Forest performance metrics;
- Gini feature importance;
- permutation importance.

---

## 5. Model Evaluation

The models are evaluated using:

- confusion matrix counts;
- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC;
- Brier score.

These metrics are used because accuracy alone is not sufficient for evaluating classification performance in a moderately imbalanced medical dataset.

The script generates:

- ROC curve figure;
- Precision-Recall curve figure;
- confusion matrix count table;
- Brier score table.

Confusion matrix counts and Brier scores are reported as CSV tables.

---

## 6. Cross-Validation Robustness Check

In addition to the held-out train-test split, repeated stratified cross-validation is included as a robustness check.

The purpose of this analysis is to examine whether the relative performance of Logistic Regression and Random Forest is stable across different data partitions, rather than being an artefact of one particular train-test split.

The cross-validation procedure preserves class proportions across folds and fits preprocessing steps within each training fold only. This means that median imputation and Logistic Regression standardisation are fitted inside the cross-validation pipeline, reducing the risk of validation-set leakage.

The script reports cross-validation results as mean values with standard deviations for:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC.

---

## 7. Model Interpretation

The code compares pre-modelling distributional evidence with model-based importance measures.

The interpretation stage includes:

- Logistic Regression odds ratios;
- Random Forest Gini importance;
- Random Forest permutation importance;
- comparison between divergence measures and model-based importance values.

This supports the essay's main analytical question: whether the structure of the raw data helps explain later model behaviour.

---

## 8. Additional Analysis

Additional checks include:

- threshold sensitivity analysis;
- probability accuracy assessment using the Brier score;
- reduced feature model comparison.

The reduced feature analysis focuses on the most informative variables identified in the exploratory and model-based analysis:

- Glucose;
- BMI;
- Age.

This analysis is used as a robustness check to examine whether the main predictive signal is concentrated in a smaller subset of predictors.

---

## Main Script

The main analysis script is:

```text
diabetes_essay_analysis.py
```

This script performs the main data preprocessing, exploratory analysis, model fitting, model evaluation, cross-validation robustness check, interpretation, and output generation used in the essay.

Running this script from start to finish reproduces the main tables and figures used to support the essay.

---

## How to Run

Place `diabetes.csv` in the same folder as the script, or inside a `data/` folder.

Then run:

```bash
python diabetes_essay_analysis.py
```

The output files will be saved in:

```text
outputs/tables/
outputs/figures/
```

The script also saves copies of the main figure files in the script folder so that the LaTeX file can locate them without changing image paths.

---

## Output Tables

The script generates the following CSV tables:

```text
dataset_summary.csv
class_distribution.csv
invalid_zero_counts.csv
missingness_indicator_summary.csv
cleaning_summary.csv
correlation_matrix.csv
js_ks_divergence.csv
js_divergence_rank.csv
train_test_split_summary.csv
lr_performance.csv
lr_odds_ratios.csv
rf_performance.csv
rf_importance.csv
model_comparison.csv
confusion_matrix_counts.csv
cross_validation_performance.csv
brier_scores.csv
threshold_sensitivity.csv
rf_permutation_importance.csv
divergence_importance_comparison.csv
reduced_feature_comparison.csv
reduced_feature_table_for_essay.csv
```

These tables correspond to the numerical results, diagnostic checks, model comparisons, and summary statistics reported in the essay.

---

## Output Figures

The script generates the following figures:

```text
glucose_distribution_by_outcome.png
bloodpressure_distribution_by_outcome.png
correlation_heatmap.png
rf_importance.png
roc_curves.png
pr_curves.png
```

These figures correspond to the main visual results shown in the essay.

---

## Reproducibility

A fixed random seed is used for:

- the train-test split;
- Random Forest fitting;
- permutation importance calculations;
- repeated stratified cross-validation.

The modelling workflow uses stratified train-test splitting so that the class proportions in the training and test sets remain similar to the original dataset.

Invalid zero values are replaced with missing values before model fitting. Median imputation is performed inside the modelling pipelines, so that imputation values are estimated from the training data only and then applied to the test data. This helps reduce the risk of data leakage during model training and evaluation.

Repeated stratified cross-validation is also implemented using pipelines, so that preprocessing is fitted separately within each training fold.

---

## Requirements

The implementation uses:

- Python;
- pandas;
- NumPy;
- scikit-learn;
- matplotlib.

The main packages can be installed using:

```bash
pip install pandas numpy scikit-learn matplotlib
```

The Kolmogorov-Smirnov statistic uses SciPy when available.

To install SciPy, run:

```bash
pip install scipy
```

If SciPy is not installed, the script uses a manual fallback implementation for the Kolmogorov-Smirnov statistic.

---

## Relationship Between Essay Tables and Output Files

Some tables and figures in the written essay are presentation-formatted versions of the CSV tables and PNG figures generated by the script.

The main table correspondence is:

| Essay Table | Output File |
|---|---|
| Essay Table 2 | `invalid_zero_counts.csv` |
| Essay Table 3 | `missingness_indicator_summary.csv` |
| Essay Table 4 | `class_distribution.csv` |
| Essay Table 5 | `js_divergence_rank.csv` |
| Essay Table 7 | `lr_performance.csv` |
| Essay Table 8 | `lr_odds_ratios.csv` |
| Essay Table 9 | `rf_performance.csv` |
| Essay Table 10 | `model_comparison.csv` |
| Essay Table 11 | `confusion_matrix_counts.csv` |
| Essay Table 12 | `cross_validation_performance.csv` |
| Essay Table 13 | `brier_scores.csv` |
| Essay Table 14 | `threshold_sensitivity.csv` |
| Essay Table 15 | `divergence_importance_comparison.csv` |
| Essay Table 16 | `reduced_feature_table_for_essay.csv` |

Tables 1 and 6 in the essay are explanatory presentation tables rather than direct CSV outputs from the script.

---

## Relationship Between Essay Figures and Output Files

The main figure correspondence is:

| Essay Figure | Output File |
|---|---|
| Essay Figure 1(a) | `glucose_distribution_by_outcome.png` |
| Essay Figure 1(b) | `bloodpressure_distribution_by_outcome.png` |
| Essay Figure 2 | `correlation_heatmap.png` |
| Essay Figure 3 | `rf_importance.png` |
| Essay Figure 4(a) | `roc_curves.png` |
| Essay Figure 4(b) | `pr_curves.png` |

Threshold sensitivity, Brier score analysis, permutation importance, and reduced feature comparison are reported mainly through CSV tables rather than separate figures.

---

## Notes

The full written essay is not reproduced in this repository.

This repository is intended to support the reproducibility of the statistical and machine-learning analysis presented in the accompanying final-year essay. The essay provides the full written explanation, methodological justification, interpretation of results, clinical discussion of false negatives and threshold choice, and limitations of the analysis.

The code should be read as supporting implementation material. The essay remains the main source for the full statistical narrative.
