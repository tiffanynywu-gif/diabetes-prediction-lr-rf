from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    make_scorer,
    roc_curve,
    precision_recall_curve,
)

try:
    from scipy.stats import ks_2samp
except ImportError:
    ks_2samp = None


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CURRENT_DIR = Path.cwd()

DATA_CANDIDATES = [
    CURRENT_DIR / "diabetes.csv",
    CURRENT_DIR / "data" / "diabetes.csv",
    SCRIPT_DIR / "diabetes.csv",
    SCRIPT_DIR / "data" / "diabetes.csv",
]

DATA_PATH = None

for candidate in DATA_CANDIDATES:
    if candidate.exists():
        DATA_PATH = candidate
        break

if DATA_PATH is None:
    checked = "\n".join(str(p) for p in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Could not find diabetes.csv. The script checked these locations:\n"
        f"{checked}\n\n"
        "Fix: put diabetes.csv in the same folder as this script, "
        "or put it inside a data/ folder."
    )

print(f"Using data file: {DATA_PATH}")

OUTPUT_DIR = Path("outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# These constants are stated explicitly so that the code matches the essay text.
JS_BINS = 20
EDA_HIST_BINS = 25
PERMUTATION_REPEATS = 30
THRESHOLDS = [0.30, 0.40, 0.50]

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET = "Outcome"

INVALID_ZERO_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

REDUCED_FEATURES = [
    "Glucose",
    "BMI",
    "Age",
]


# ============================================================
# Helper functions
# ============================================================

def save_table(df: pd.DataFrame, filename: str) -> None:
    """Save a dataframe as a CSV file in outputs/tables."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def save_figure(filename: str) -> None:
    """
    Save the current matplotlib figure.

    The figure is saved in:
    1. outputs/figures/ for project organisation;
    2. the script folder, so LaTeX can find it without changing image names.
    """
    plt.tight_layout()

    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    latex_path = SCRIPT_DIR / filename
    if latex_path.resolve() != output_path.resolve():
        plt.savefig(latex_path, dpi=300, bbox_inches="tight")
        print(f"Saved LaTeX figure copy: {latex_path}")

    plt.close()


def round_table(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round numeric columns in a dataframe."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def prepare_features(data: pd.DataFrame, feature_set=None) -> pd.DataFrame:
    """
    Select features and replace invalid zero values with NaN.

    Median imputation is then performed inside the modelling pipelines.
    """
    if feature_set is None:
        feature_set = FEATURES

    X = data[feature_set].copy()

    zero_cols = [col for col in INVALID_ZERO_COLS if col in feature_set]
    X[zero_cols] = X[zero_cols].replace(0, np.nan)

    return X


def make_lr_pipeline() -> Pipeline:
    """Create Logistic Regression pipeline."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def make_rf_pipeline() -> Pipeline:
    """Create Random Forest pipeline."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            max_features="sqrt",
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])


def format_metrics(y_true, y_pred, y_proba) -> dict:
    """Return the main model evaluation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
    }


def empirical_js_divergence(x0: np.ndarray, x1: np.ndarray, bins: int = JS_BINS) -> float:
    """
    Calculate empirical Jensen-Shannon divergence between two distributions.

    The distributions are approximated using histograms with common bin edges.
    Empty probability bins are handled by evaluating the KL summation only over
    bins with positive probability mass in the numerator. This avoids undefined
    log terms without adding an epsilon constant.
    """
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    combined = np.concatenate([x0, x1])
    min_value = combined.min()
    max_value = combined.max()

    if min_value == max_value:
        return 0.0

    counts0, edges = np.histogram(x0, bins=bins, range=(min_value, max_value))
    counts1, _ = np.histogram(x1, bins=edges)

    if counts0.sum() == 0 or counts1.sum() == 0:
        return np.nan

    p = counts0 / counts0.sum()
    q = counts1 / counts1.sum()
    m = 0.5 * (p + q)

    def kl_divergence(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def manual_ks_statistic(x0: np.ndarray, x1: np.ndarray) -> float:
    """Manual Kolmogorov-Smirnov statistic if SciPy is not installed."""
    x0 = np.sort(np.asarray(x0))
    x1 = np.sort(np.asarray(x1))

    values = np.sort(np.unique(np.concatenate([x0, x1])))

    cdf0 = np.searchsorted(x0, values, side="right") / len(x0)
    cdf1 = np.searchsorted(x1, values, side="right") / len(x1)

    return np.max(np.abs(cdf0 - cdf1))


def ks_statistic(x0: np.ndarray, x1: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic."""
    if ks_2samp is not None:
        return ks_2samp(x0, x1).statistic

    return manual_ks_statistic(x0, x1)


def print_latex_rows(df: pd.DataFrame, columns: list[str], float_decimals: int = 4) -> None:
    """Print simple LaTeX table rows for selected columns."""
    for _, row in df.iterrows():
        values = []

        for col in columns:
            value = row[col]

            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{float_decimals}f}")
            else:
                values.append(str(value))

        print(" & ".join(values) + r" \\")


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

missing_columns = [col for col in FEATURES + [TARGET] if col not in df.columns]

if missing_columns:
    raise ValueError(f"The dataset is missing these required columns: {missing_columns}")

y = df[TARGET].copy()

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head().to_string(index=False))


# ============================================================
# 2. Basic dataset summary and class distribution
# ============================================================

dataset_summary_df = pd.DataFrame({
    "Quantity": [
        "Number of observations",
        "Number of predictor variables",
        "Target variable",
        "Negative class label",
        "Positive class label",
    ],
    "Value": [
        len(df),
        len(FEATURES),
        TARGET,
        "Outcome = 0",
        "Outcome = 1",
    ],
})

save_table(dataset_summary_df, "dataset_summary.csv")

class_counts = df[TARGET].value_counts().sort_index()

class_distribution_df = pd.DataFrame({
    "Outcome": class_counts.index,
    "Count": class_counts.values,
    "Percentage": 100 * class_counts.values / class_counts.values.sum(),
})

save_table(round_table(class_distribution_df, 4), "class_distribution.csv")

print("\nClass distribution:")
print(round_table(class_distribution_df, 4).to_string(index=False))


# ============================================================
# 3. Invalid zero analysis
# ============================================================

invalid_zero_rows = []

for col in INVALID_ZERO_COLS:
    zero_count = int((df[col] == 0).sum())
    invalid_zero_rows.append({
        "Variable": col,
        "Invalid zero count": zero_count,
        "Percentage of dataset": 100 * zero_count / len(df),
    })

invalid_zero_df = pd.DataFrame(invalid_zero_rows)
save_table(round_table(invalid_zero_df, 4), "invalid_zero_counts.csv")

print("\nInvalid zero counts:")
print(round_table(invalid_zero_df, 4).to_string(index=False))


# ============================================================
# 3.1 Missingness indicator analysis by outcome group
# ============================================================

missingness_rows = []

for col in INVALID_ZERO_COLS:
    outcome0_rate = 100 * (df.loc[df[TARGET] == 0, col] == 0).mean()
    outcome1_rate = 100 * (df.loc[df[TARGET] == 1, col] == 0).mean()

    missingness_rows.append({
        "Variable": col,
        "Missingness rate for Outcome = 0": outcome0_rate,
        "Missingness rate for Outcome = 1": outcome1_rate,
    })

missingness_indicator_df = pd.DataFrame(missingness_rows)

save_table(
    round_table(missingness_indicator_df, 4),
    "missingness_indicator_summary.csv",
)

print("\nMissingness indicator summary by outcome:")
print(round_table(missingness_indicator_df, 4).to_string(index=False))

print("\nLaTeX rows for missingness indicator table:")
print_latex_rows(
    missingness_indicator_df,
    [
        "Variable",
        "Missingness rate for Outcome = 0",
        "Missingness rate for Outcome = 1",
    ],
    float_decimals=4,
)


# ============================================================
# 4. Cleaned dataset for EDA, JS divergence, and KS statistic
# ============================================================

clean_df = df.copy()

for col in INVALID_ZERO_COLS:
    clean_df[col] = clean_df[col].astype(float)
    median_value = clean_df.loc[clean_df[col] > 0, col].median()
    clean_df.loc[clean_df[col] == 0, col] = median_value

cleaning_summary_rows = []

for col in INVALID_ZERO_COLS:
    original_zero_count = int((df[col] == 0).sum())
    replacement_value = df.loc[df[col] > 0, col].median()

    cleaning_summary_rows.append({
        "Variable": col,
        "Invalid zero count": original_zero_count,
        "Median replacement value": replacement_value,
    })

cleaning_summary_df = pd.DataFrame(cleaning_summary_rows)
save_table(round_table(cleaning_summary_df, 4), "cleaning_summary.csv")

print("\nCleaning summary:")
print(round_table(cleaning_summary_df, 4).to_string(index=False))


# ============================================================
# 5. EDA figures: class-conditional distributions
# ============================================================

def plot_class_conditional_distribution(feature: str, filename: str) -> None:
    """Plot and save one class-conditional distribution figure."""
    plt.figure(figsize=(5.5, 4))

    plt.hist(
        clean_df.loc[clean_df[TARGET] == 0, feature],
        bins=EDA_HIST_BINS,
        density=True,
        alpha=0.6,
        label="Non-diabetic (Outcome = 0)",
    )

    plt.hist(
        clean_df.loc[clean_df[TARGET] == 1, feature],
        bins=EDA_HIST_BINS,
        density=True,
        alpha=0.6,
        label="Diabetic (Outcome = 1)",
    )

    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"Class-Conditional Distribution of {feature}")
    plt.legend(fontsize=8)

    save_figure(filename)


# These filenames match the LaTeX file.
plot_class_conditional_distribution(
    "Glucose",
    "glucose_distribution_by_outcome.png",
)

plot_class_conditional_distribution(
    "BloodPressure",
    "bloodpressure_distribution_by_outcome.png",
)


# ============================================================
# 6. Correlation analysis
# ============================================================

corr = clean_df[FEATURES + [TARGET]].corr()

corr_table = corr.reset_index().rename(columns={"index": "Variable"})
save_table(round_table(corr_table, 4), "correlation_matrix.csv")

plt.figure(figsize=(9, 7))
im = plt.imshow(corr, aspect="auto")
plt.colorbar(im, label="Pearson correlation")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

plt.title("Correlation Heatmap for Predictors and Outcome Variable")
save_figure("correlation_heatmap.png")

print("\nCorrelation with Outcome:")
print(
    round_table(
        corr[[TARGET]]
        .drop(index=TARGET)
        .sort_values(TARGET, ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", TARGET: "Correlation with Outcome"}),
        4,
    ).to_string(index=False)
)


# ============================================================
# 7. Distributional divergence table
# ============================================================

divergence_rows = []

for feature in FEATURES:
    x0 = clean_df.loc[clean_df[TARGET] == 0, feature].to_numpy()
    x1 = clean_df.loc[clean_df[TARGET] == 1, feature].to_numpy()

    divergence_rows.append({
        "Feature": feature,
        "JS Divergence": empirical_js_divergence(x0, x1, bins=JS_BINS),
        "KS Statistic": ks_statistic(x0, x1),
    })

divergence_df = pd.DataFrame(divergence_rows)
divergence_df_sorted = divergence_df.sort_values("JS Divergence", ascending=False)

save_table(round_table(divergence_df_sorted, 4), "js_ks_divergence.csv")

# Table 5 style output: JS divergence with relative rank.
js_rank_df = divergence_df_sorted[["Feature", "JS Divergence"]].copy()
js_rank_df = js_rank_df.rename(columns={"Feature": "Variable"})
js_rank_df["Relative Rank"] = range(1, len(js_rank_df) + 1)

save_table(round_table(js_rank_df, 4), "js_divergence_rank.csv")

print("\nJS divergence ranking table:")
print(round_table(js_rank_df, 4).to_string(index=False))

print("\nLaTeX rows for JS divergence ranking table:")
print_latex_rows(
    js_rank_df,
    ["Variable", "JS Divergence", "Relative Rank"],
    float_decimals=4,
)

print("\nJS divergence and KS statistics:")
print(round_table(divergence_df_sorted, 4).to_string(index=False))

print("\nLaTeX rows for JS and KS divergence table:")
print_latex_rows(
    divergence_df_sorted,
    ["Feature", "JS Divergence", "KS Statistic"],
    float_decimals=4,
)


# ============================================================
# 8. Train-test split
# ============================================================

X = prepare_features(df, FEATURES)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

split_summary_df = pd.DataFrame({
    "Split": ["Training set", "Test set"],
    "Number of observations": [len(X_train), len(X_test)],
    "Positive cases": [int(y_train.sum()), int(y_test.sum())],
    "Negative cases": [int((y_train == 0).sum()), int((y_test == 0).sum())],
    "Positive class percentage": [100 * y_train.mean(), 100 * y_test.mean()],
})

save_table(round_table(split_summary_df, 4), "train_test_split_summary.csv")

print("\nTrain-test split summary:")
print(round_table(split_summary_df, 4).to_string(index=False))


# ============================================================
# 9. Logistic Regression
# ============================================================

lr_pipe = make_lr_pipeline()
lr_pipe.fit(X_train, y_train)

lr_pred = lr_pipe.predict(X_test)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
lr_cm = confusion_matrix(y_test, lr_pred, labels=[0, 1])

lr_metrics = format_metrics(y_test, lr_pred, lr_proba)

lr_performance_df = pd.DataFrame({
    "Metric": list(lr_metrics.keys()),
    "Logistic Regression": list(lr_metrics.values()),
})

save_table(round_table(lr_performance_df, 4), "lr_performance.csv")

lr_model = lr_pipe.named_steps["model"]
lr_coefficients = lr_model.coef_[0]
lr_odds_ratios = np.exp(lr_coefficients)

lr_odds_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": lr_coefficients,
    "Odds Ratio": lr_odds_ratios,
}).sort_values("Odds Ratio", ascending=False)

save_table(round_table(lr_odds_df, 4), "lr_odds_ratios.csv")

print("\nLogistic Regression confusion matrix:")
print(lr_cm)

print("\nLogistic Regression performance:")
print(round_table(lr_performance_df, 4).to_string(index=False))

print("\nLogistic Regression odds ratios:")
print(round_table(lr_odds_df, 4).to_string(index=False))


# ============================================================
# 10. Random Forest
# ============================================================

rf_pipe = make_rf_pipeline()
rf_pipe.fit(X_train, y_train)

rf_pred = rf_pipe.predict(X_test)
rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
rf_cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])

rf_metrics = format_metrics(y_test, rf_pred, rf_proba)

rf_performance_df = pd.DataFrame({
    "Metric": list(rf_metrics.keys()),
    "Random Forest": list(rf_metrics.values()),
})

save_table(round_table(rf_performance_df, 4), "rf_performance.csv")

rf_model = rf_pipe.named_steps["model"]

rf_importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "RF Gini Importance": rf_model.feature_importances_,
}).sort_values("RF Gini Importance", ascending=False)

save_table(round_table(rf_importance_df, 4), "rf_importance.csv")

print("\nRandom Forest confusion matrix:")
print(rf_cm)

print("\nRandom Forest performance:")
print(round_table(rf_performance_df, 4).to_string(index=False))

print("\nRandom Forest Gini importance:")
print(round_table(rf_importance_df, 4).to_string(index=False))


# ============================================================
# 11. Model comparison
# ============================================================

comparison_df = pd.DataFrame({
    "Metric": list(lr_metrics.keys()),
    "Logistic Regression": list(lr_metrics.values()),
    "Random Forest": list(rf_metrics.values()),
})

save_table(round_table(comparison_df, 4), "model_comparison.csv")

print("\nModel comparison:")
print(round_table(comparison_df, 4).to_string(index=False))


# ============================================================
# 11.1 ROC and Precision-Recall curve figures
# ============================================================

# ROC curve data
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

plt.figure(figsize=(6, 5))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_metrics['ROC-AUC']:.3f})",
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_metrics['ROC-AUC']:.3f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(fontsize=8)

save_figure("roc_curves.png")


# Precision-Recall curve data
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_proba)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_proba)

baseline_prevalence = y_test.mean()

plt.figure(figsize=(6, 5))

plt.plot(
    lr_recall,
    lr_precision,
    label=f"Logistic Regression (PR-AUC = {lr_metrics['PR-AUC']:.3f})",
)

plt.plot(
    rf_recall,
    rf_precision,
    label=f"Random Forest (PR-AUC = {rf_metrics['PR-AUC']:.3f})",
)

plt.axhline(
    baseline_prevalence,
    linestyle="--",
    label=f"Baseline prevalence = {baseline_prevalence:.3f}",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend(fontsize=8)

save_figure("pr_curves.png")


# ============================================================
# 11.2 Repeated stratified cross-validation robustness check
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=RANDOM_STATE,
)

scoring = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, zero_division=0),
    "Recall": make_scorer(recall_score, zero_division=0),
    "F1-score": make_scorer(f1_score, zero_division=0),
    "ROC-AUC": "roc_auc",
    "PR-AUC": "average_precision",
}

cv_rows = []

for model_name, estimator in [
    ("Logistic Regression", make_lr_pipeline()),
    ("Random Forest", make_rf_pipeline()),
]:
    cv_results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    row = {"Model": model_name}

    for metric_name in scoring.keys():
        scores = cv_results[f"test_{metric_name}"]
        row[f"{metric_name} Mean"] = scores.mean()
        row[f"{metric_name} SD"] = scores.std(ddof=1)

    cv_rows.append(row)

cross_validation_df = pd.DataFrame(cv_rows)
cross_validation_df_rounded = round_table(cross_validation_df, 4)

save_table(cross_validation_df_rounded, "cross_validation_performance.csv")

print("\nRepeated stratified cross-validation performance:")
print(cross_validation_df_rounded.to_string(index=False))

print("\nLaTeX rows for cross-validation table:")
for _, row in cross_validation_df.iterrows():
    print(
        f"{row['Model']} & "
        f"{row['Accuracy Mean']:.4f} $\\pm$ {row['Accuracy SD']:.4f} & "
        f"{row['Precision Mean']:.4f} $\\pm$ {row['Precision SD']:.4f} & "
        f"{row['Recall Mean']:.4f} $\\pm$ {row['Recall SD']:.4f} & "
        f"{row['F1-score Mean']:.4f} $\\pm$ {row['F1-score SD']:.4f} & "
        f"{row['ROC-AUC Mean']:.4f} $\\pm$ {row['ROC-AUC SD']:.4f} & "
        f"{row['PR-AUC Mean']:.4f} $\\pm$ {row['PR-AUC SD']:.4f} \\\\"
    )


# ============================================================
# 12. Confusion matrix counts
# ============================================================

lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()
rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()

confusion_counts_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "TN": [lr_tn, rf_tn],
    "FP": [lr_fp, rf_fp],
    "FN": [lr_fn, rf_fn],
    "TP": [lr_tp, rf_tp],
})

save_table(confusion_counts_df, "confusion_matrix_counts.csv")

print("\nConfusion matrix counts:")
print(confusion_counts_df.to_string(index=False))

print("\nLaTeX rows for confusion matrix count table:")
print_latex_rows(confusion_counts_df, ["Model", "TN", "FP", "FN", "TP"], float_decimals=0)


# ============================================================
# 13. Random Forest feature importance figure
# ============================================================

rf_importance_plot_df = rf_importance_df.sort_values("RF Gini Importance", ascending=True)

plt.figure(figsize=(7, 5))
plt.barh(
    rf_importance_plot_df["Feature"],
    rf_importance_plot_df["RF Gini Importance"],
)

max_importance = rf_importance_plot_df["RF Gini Importance"].max()

for i, value in enumerate(rf_importance_plot_df["RF Gini Importance"]):
    plt.text(
        value + max_importance * 0.01,
        i,
        f"{value:.3f}",
        va="center",
        fontsize=8,
    )

plt.xlim(0, max_importance * 1.18)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
save_figure("rf_importance.png")


# ============================================================
# 14. Brier score
# ============================================================

brier_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Brier Score": [
        brier_score_loss(y_test, lr_proba),
        brier_score_loss(y_test, rf_proba),
    ],
})

brier_df_rounded = round_table(brier_df, 4)
save_table(brier_df_rounded, "brier_scores.csv")

print("\nBrier scores:")
print(brier_df_rounded.to_string(index=False))

print("\nLaTeX rows for Brier score table:")
print_latex_rows(brier_df, ["Model", "Brier Score"], float_decimals=4)


# ============================================================
# 15. Threshold sensitivity analysis
# ============================================================

threshold_rows = []

for model_name, proba in [
    ("Logistic Regression", lr_proba),
    ("Random Forest", rf_proba),
]:
    for threshold in THRESHOLDS:
        pred = (proba >= threshold).astype(int)

        threshold_rows.append({
            "Model": model_name,
            "Threshold": threshold,
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
        })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df_rounded = round_table(threshold_df, 4)

save_table(threshold_df_rounded, "threshold_sensitivity.csv")

print("\nThreshold sensitivity analysis:")
print(threshold_df_rounded.to_string(index=False))

print("\nLaTeX rows for threshold sensitivity table:")
print_latex_rows(
    threshold_df,
    ["Model", "Threshold", "Precision", "Recall", "F1-score"],
    float_decimals=4,
)


# ============================================================
# 16. Random Forest permutation importance
# ============================================================

# PR-AUC is used because the diabetic class is the positive minority class.
# The value is the mean decrease in PR-AUC after permuting each feature.

perm_result = permutation_importance(
    rf_pipe,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_permutation_df = pd.DataFrame({
    "Feature": FEATURES,
    "RF Permutation Importance": perm_result.importances_mean,
    "RF Permutation Importance SD": perm_result.importances_std,
}).sort_values("RF Permutation Importance", ascending=False)

rf_permutation_df_rounded = round_table(rf_permutation_df, 4)
save_table(rf_permutation_df_rounded, "rf_permutation_importance.csv")

print("\nRandom Forest permutation importance based on PR-AUC decrease:")
print(rf_permutation_df_rounded.to_string(index=False))

print("\nLaTeX rows for RF permutation importance table:")
print_latex_rows(
    rf_permutation_df,
    ["Feature", "RF Permutation Importance", "RF Permutation Importance SD"],
    float_decimals=4,
)


# ============================================================
# 17. Final divergence and model-based importance comparison
# ============================================================

linking_df = divergence_df.merge(
    lr_odds_df[["Feature", "Odds Ratio"]],
    on="Feature",
    how="left",
).merge(
    rf_importance_df,
    on="Feature",
    how="left",
).merge(
    rf_permutation_df[["Feature", "RF Permutation Importance"]],
    on="Feature",
    how="left",
)

linking_df = linking_df.rename(columns={
    "Odds Ratio": "LR Odds Ratio",
})

linking_df = linking_df[
    [
        "Feature",
        "JS Divergence",
        "KS Statistic",
        "LR Odds Ratio",
        "RF Gini Importance",
        "RF Permutation Importance",
    ]
].sort_values("JS Divergence", ascending=False)

linking_df_rounded = round_table(linking_df, 4)

save_table(linking_df_rounded, "divergence_importance_comparison.csv")

print("\nDivergence and model-based importance comparison:")
print(linking_df_rounded.to_string(index=False))

print("\nLaTeX rows for final importance comparison table:")
print_latex_rows(
    linking_df,
    [
        "Feature",
        "JS Divergence",
        "KS Statistic",
        "LR Odds Ratio",
        "RF Gini Importance",
        "RF Permutation Importance",
    ],
    float_decimals=4,
)


# ============================================================
# 18. Reduced feature model comparison
# ============================================================

X_reduced = prepare_features(df, REDUCED_FEATURES)

# Use exactly the same train-test split indices as the full-feature analysis.
X_train_red = X_reduced.loc[X_train.index]
X_test_red = X_reduced.loc[X_test.index]
y_train_red = y_train
y_test_red = y_test

lr_red_pipe = make_lr_pipeline()
rf_red_pipe = make_rf_pipeline()

lr_red_pipe.fit(X_train_red, y_train_red)
rf_red_pipe.fit(X_train_red, y_train_red)

lr_red_pred = lr_red_pipe.predict(X_test_red)
rf_red_pred = rf_red_pipe.predict(X_test_red)

lr_red_proba = lr_red_pipe.predict_proba(X_test_red)[:, 1]
rf_red_proba = rf_red_pipe.predict_proba(X_test_red)[:, 1]

lr_red_cm = confusion_matrix(y_test_red, lr_red_pred, labels=[0, 1])
rf_red_cm = confusion_matrix(y_test_red, rf_red_pred, labels=[0, 1])

print("\nReduced Logistic Regression confusion matrix:")
print(lr_red_cm)

print("\nReduced Random Forest confusion matrix:")
print(rf_red_cm)

lr_red_tn, lr_red_fp, lr_red_fn, lr_red_tp = lr_red_cm.ravel()
rf_red_tn, rf_red_fp, rf_red_fn, rf_red_tp = rf_red_cm.ravel()

reduced_confusion_counts_df = pd.DataFrame({
    "Model": ["Reduced Logistic Regression", "Reduced Random Forest"],
    "TN": [lr_red_tn, rf_red_tn],
    "FP": [lr_red_fp, rf_red_fp],
    "FN": [lr_red_fn, rf_red_fn],
    "TP": [lr_red_tp, rf_red_tp],
})

save_table(reduced_confusion_counts_df, "reduced_confusion_matrix_counts.csv")

print("\nReduced confusion matrix counts:")
print(reduced_confusion_counts_df.to_string(index=False))

lr_red_metrics = format_metrics(y_test_red, lr_red_pred, lr_red_proba)
rf_red_metrics = format_metrics(y_test_red, rf_red_pred, rf_red_proba)

reduced_comparison_df = pd.DataFrame({
    "Metric": list(lr_metrics.keys()),
    "LR Full Features": list(lr_metrics.values()),
    "LR Reduced Features": list(lr_red_metrics.values()),
    "RF Full Features": list(rf_metrics.values()),
    "RF Reduced Features": list(rf_red_metrics.values()),
})

reduced_comparison_df_rounded = round_table(reduced_comparison_df, 4)
save_table(reduced_comparison_df_rounded, "reduced_feature_comparison.csv")

# Essay Table 16 style output: long-format reduced feature comparison.
# This version reports the same main metrics as the full model comparison.
reduced_feature_table_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Feature Set": "Full features",
        "Accuracy": lr_metrics["Accuracy"],
        "Precision": lr_metrics["Precision"],
        "Recall": lr_metrics["Recall"],
        "F1-score": lr_metrics["F1-score"],
        "ROC-AUC": lr_metrics["ROC-AUC"],
        "PR-AUC": lr_metrics["PR-AUC"],
    },
    {
        "Model": "Logistic Regression",
        "Feature Set": "Glucose, BMI, Age",
        "Accuracy": lr_red_metrics["Accuracy"],
        "Precision": lr_red_metrics["Precision"],
        "Recall": lr_red_metrics["Recall"],
        "F1-score": lr_red_metrics["F1-score"],
        "ROC-AUC": lr_red_metrics["ROC-AUC"],
        "PR-AUC": lr_red_metrics["PR-AUC"],
    },
    {
        "Model": "Random Forest",
        "Feature Set": "Full features",
        "Accuracy": rf_metrics["Accuracy"],
        "Precision": rf_metrics["Precision"],
        "Recall": rf_metrics["Recall"],
        "F1-score": rf_metrics["F1-score"],
        "ROC-AUC": rf_metrics["ROC-AUC"],
        "PR-AUC": rf_metrics["PR-AUC"],
    },
    {
        "Model": "Random Forest",
        "Feature Set": "Glucose, BMI, Age",
        "Accuracy": rf_red_metrics["Accuracy"],
        "Precision": rf_red_metrics["Precision"],
        "Recall": rf_red_metrics["Recall"],
        "F1-score": rf_red_metrics["F1-score"],
        "ROC-AUC": rf_red_metrics["ROC-AUC"],
        "PR-AUC": rf_red_metrics["PR-AUC"],
    },
])

reduced_feature_table_df_rounded = round_table(reduced_feature_table_df, 4)

save_table(
    reduced_feature_table_df_rounded,
    "reduced_feature_table_for_essay.csv",
)

print("\nReduced feature table for essay:")
print(reduced_feature_table_df_rounded.to_string(index=False))

print("\nLaTeX rows for essay reduced feature table:")
print_latex_rows(
    reduced_feature_table_df,
    [
        "Model",
        "Feature Set",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC",
    ],
    float_decimals=4,
)

print("\nReduced feature model comparison:")
print(reduced_comparison_df_rounded.to_string(index=False))

print("\nLaTeX rows for reduced feature comparison table:")
print_latex_rows(
    reduced_comparison_df,
    [
        "Metric",
        "LR Full Features",
        "LR Reduced Features",
        "RF Full Features",
        "RF Reduced Features",
    ],
    float_decimals=4,
)


# ============================================================
# 19. Final message
# ============================================================

print("\nDone. Output files have been saved.")

print("\nImportant table files used in the essay:")

# 1. Data preprocessing and EDA tables
print(f"- {TABLE_DIR / 'class_distribution.csv'}")                # Table 2
print(f"- {TABLE_DIR / 'invalid_zero_counts.csv'}")               # Table 3
print(f"- {TABLE_DIR / 'missingness_indicator_summary.csv'}")     # Table 4
print(f"- {TABLE_DIR / 'js_divergence_rank.csv'}")                # Table 5

# 2. Model performance and interpretation tables
print(f"- {TABLE_DIR / 'lr_performance.csv'}")                    # Table 7
print(f"- {TABLE_DIR / 'lr_odds_ratios.csv'}")                    # Table 8
print(f"- {TABLE_DIR / 'rf_performance.csv'}")                    # Table 9
print(f"- {TABLE_DIR / 'model_comparison.csv'}")                  # Table 10
print(f"- {TABLE_DIR / 'confusion_matrix_counts.csv'}")           # Table 11
print(f"- {TABLE_DIR / 'cross_validation_performance.csv'}")      # Table 12
print(f"- {TABLE_DIR / 'brier_scores.csv'}")                      # Table 13
print(f"- {TABLE_DIR / 'threshold_sensitivity.csv'}")             # Table 14
print(f"- {TABLE_DIR / 'divergence_importance_comparison.csv'}")  # Table 15
print(f"- {TABLE_DIR / 'reduced_feature_table_for_essay.csv'}")   # Table 16

print("\nImportant figure files used in the essay:")

print(f"- {FIGURE_DIR / 'glucose_distribution_by_outcome.png'}")        # Figure 1(a)
print(f"- {FIGURE_DIR / 'bloodpressure_distribution_by_outcome.png'}")  # Figure 1(b)
print(f"- {FIGURE_DIR / 'correlation_heatmap.png'}")                    # Figure 2
print(f"- {FIGURE_DIR / 'rf_importance.png'}")                          # Figure 3
print(f"- {FIGURE_DIR / 'roc_curves.png'}")                             # Figure 4(a)
print(f"- {FIGURE_DIR / 'pr_curves.png'}")                              # Figure 4(b)

