import time
import warnings
from math import comb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import (
    VarianceThreshold, f_classif, mutual_info_classif, SelectKBest, RFE, RFECV
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
K_VALUES = [2, 5, 10, 15, 20, 25, 30]



def make_model():
    """A fresh scaled logistic-regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])


def cv_accuracy(X, y, features=None):
    """Mean 5-fold CV accuracy, with scaling fitted separately in each fold."""
    X_part = X if features is None else X.loc[:, list(features)]
    scores = cross_val_score(make_model(), X_part, y, cv=CV, scoring="accuracy")
    return scores.mean(), scores.std()


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Drop one feature from each pair whose absolute correlation exceeds
    threshold. Keep the feature with the larger absolute correlation
    with y. The decision is learned from the training fold only.
    """
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        corr = X.corr().abs()
        target_corr = X.apply(lambda col: abs(np.corrcoef(col, np.asarray(y))[0, 1]))
        keep = list(X.columns)

        for i, col_a in enumerate(X.columns):
            if col_a not in keep:
                continue
            for col_b in X.columns[i + 1:]:
                if col_b not in keep:
                    continue
                if corr.loc[col_a, col_b] > self.threshold:
                    if target_corr[col_a] >= target_corr[col_b]:
                        keep.remove(col_b)
                    else:
                        keep.remove(col_a)
                        break

        self.selected_features_ = keep
        self.support_ = np.array([name in keep for name in self.feature_names_in_])
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.loc[:, self.selected_features_].to_numpy()

    def get_support(self):
        return self.support_


class FixedFeatureSelector(BaseEstimator, TransformerMixin):
    """Select a previously validation-chosen list of columns inside a Pipeline."""
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        self.support_ = np.array([name in self.features for name in self.feature_names_in_])
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.loc[:, list(self.features)].to_numpy()

    def get_support(self):
        return self.support_


def manual_forward_selection(X, y, max_features=10, epsilon=0.001):
    """Greedy forward selection using 5-fold CV accuracy."""
    remaining = list(X.columns)
    selected = []
    history = []
    previous_score = 0.0

    while remaining and len(selected) < max_features:
        candidates = []
        for feature in remaining:
            candidate = selected + [feature]
            mean_score, std_score = cv_accuracy(X, y, candidate)
            candidates.append((mean_score, std_score, feature))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_std, best_feature = candidates[0]

        if selected and best_score - previous_score < epsilon:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        history.append((len(selected), best_score, best_std, best_feature))
        previous_score = best_score

    return selected, history


def manual_backward_selection(X, y, min_features=10):
    """Greedy backward elimination: remove the feature with least CV damage."""
    remaining = list(X.columns)
    history = []

    while len(remaining) > min_features:
        candidates = []
        for feature in remaining:
            candidate = [name for name in remaining if name != feature]
            mean_score, std_score = cv_accuracy(X, y, candidate)
            candidates.append((mean_score, std_score, feature))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_std, removed_feature = candidates[0]
        remaining.remove(removed_feature)
        history.append((len(remaining), best_score, best_std, removed_feature))

    return remaining, history


def wrapper_rfe(estimator=None, n_features=10):
    """RFE estimator with scaling performed inside each estimator fit."""
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    inner = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator)
    ])
    return RFE(
        estimator=inner,
        n_features_to_select=n_features,
        importance_getter="named_steps.classifier.coef_"
    )


def make_rfecv():
    inner = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    return RFECV(
        estimator=inner,
        step=1,
        cv=CV,
        scoring="accuracy",
        importance_getter="named_steps.classifier.coef_",
        min_features_to_select=1
    )


def print_result(label, features, X, y):
    mean, std = cv_accuracy(X, y, features)
    print(f"{label}: {len(features)} features; CV accuracy={mean:.4f} "
          f"(std={std:.4f})")
    print(features)

 

data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

print("\n" + "=" * 78)
print("PART A – DATA INSPECTION AND BASELINE")
print("=" * 78)

print("\nDataset shape:", X.shape)
print("\nData types:\n", X.dtypes)
print("\nDescriptive statistics:\n", X.describe().T)
print("\nMissing values:", int(X.isna().sum().sum()))
print("Duplicated rows:", int(X.duplicated().sum()))
print("\nTarget counts:\n", y.value_counts().sort_index())
print("Target labels: 0 = malignant; 1 = benign")

y.value_counts().sort_index().plot(kind="bar", rot=0)
plt.title("Target distribution (0 = malignant, 1 = benign)")
plt.xlabel("Target")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_accuracy = accuracy_score(y_test, dummy.predict(X_test))
print(f"\nDummyClassifier test accuracy: {dummy_accuracy:.4f}")

baseline_start = time.perf_counter()
baseline = make_model()
baseline.fit(X_train, y_train)
baseline_train_time = time.perf_counter() - baseline_start

baseline_train_pred = baseline.predict(X_train)
baseline_test_pred = baseline.predict(X_test)
baseline_train_acc = accuracy_score(y_train, baseline_train_pred)
baseline_train_f1 = f1_score(y_train, baseline_train_pred)
baseline_test_acc = accuracy_score(y_test, baseline_test_pred)
baseline_test_f1 = f1_score(y_test, baseline_test_pred)

print("\nBaseline – Logistic Regression with all 30 features")
print(f"Training accuracy: {baseline_train_acc:.4f}")
print(f"Training F1-score: {baseline_train_f1:.4f}")
print(f"Test accuracy:     {baseline_test_acc:.4f}")
print(f"Test F1-score:     {baseline_test_f1:.4f}")
print(f"Training time:     {baseline_train_time:.4f} seconds")

corr_train = X_train.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr_train, cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix – training features")
plt.tight_layout()
plt.show()

print("""
Part A interpretation:
- The dataset has 569 observations and 30 numerical predictors, with no
  missing values. The target is imbalanced (357 benign and 212 malignant).
- Several measurements describe closely related properties of a nucleus.
  Radius, perimeter and area are therefore strongly correlated, especially
  among their mean and worst versions.
- This redundancy makes the dataset suitable for feature selection:
  a smaller subset may retain predictive information while reducing
  complexity and improving interpretability.
""")



print("\n" + "=" * 78)
print("PART B – FILTER METHODS")
print("=" * 78)

print("\nB.1 – Variance threshold")
variance_rows = []
for threshold in [0, 0.01, 0.1]:
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)  # unscaled training data
    kept = int(selector.get_support().sum())
    removed = X_train.shape[1] - kept
    variance_rows.append({
        "Threshold": threshold,
        "Remaining features": kept,
        "Removed features": removed
    })
variance_table = pd.DataFrame(variance_rows)
print(variance_table.to_string(index=False))
print("""
Explanation:
A variance threshold must be applied before standardization. After
StandardScaler, every non-constant training feature has variance close to 1,
so thresholds such as 0.01 and 0.1 would remove almost nothing. This filter
has limited value here because the original measurements are continuous and
most have substantial variance; it is more useful for constant or nearly
constant predictors.
""")

print("\nB.2 – Removing correlated features")
correlation_rows = []
correlation_selected = {}
for threshold in [0.95, 0.90, 0.80]:
    start = time.perf_counter()
    selector = CorrelationSelector(threshold=threshold)
    selector.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    selected = selector.selected_features_
    correlation_selected[threshold] = selected
    mean, std = cv_accuracy(X_train, y_train, selected)
    correlation_rows.append({
        "Threshold": threshold,
        "Remaining features": len(selected),
        "CV accuracy": mean,
        "CV std": std,
        "Selection time (s)": elapsed
    })
correlation_table = pd.DataFrame(correlation_rows)
print(correlation_table.to_string(index=False))
print("""
Interpretation:
Correlation filtering removes redundancy, but redundancy removal alone
does not guarantee better predictive accuracy. Two correlated variables
may still contain useful information, and the result depends on the chosen
threshold and which variable is retained. CV accuracy is reported here to
respect the instruction not to use the test set before Part D.3.
""")

print("\nB.3 – ANOVA F-score, mutual information and SelectKBest")
f_scores, f_pvalues = f_classif(X_train, y_train)
mi_scores = mutual_info_classif(
    X_train, y_train, random_state=RANDOM_STATE
)

f_ranking = pd.Series(f_scores, index=feature_names).sort_values(ascending=False)
mi_ranking = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

print("\nTop 10 features by ANOVA F-score:\n", f_ranking.head(10))
print("\nTop 10 features by mutual information:\n", mi_ranking.head(10))
rho, rho_p = spearmanr(f_scores, mi_scores)
print(f"\nSpearman correlation between rankings: rho={rho:.4f}, p={rho_p:.4g}")

plt.figure(figsize=(12, 6))
f_ranking.plot(kind="bar")
plt.title("Features ranked by ANOVA F-score")
plt.ylabel("F-score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
mi_ranking.plot(kind="bar")
plt.title("Features ranked by mutual information")
plt.ylabel("Mutual information")
plt.tight_layout()
plt.show()

filter_cv_results = []
for score_name, score_function in [
    ("ANOVA F-score", f_classif),
    ("Mutual information", lambda a, b: (
        mutual_info_classif(a, b, random_state=RANDOM_STATE), None
    ))
]:
    for k in K_VALUES:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=score_function, k=k)),
            ("classifier", LogisticRegression(max_iter=1000,
                                               random_state=RANDOM_STATE))
        ])
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV, scoring="accuracy"
        )
        filter_cv_results.append({
            "Score": score_name, "k": k,
            "Mean CV accuracy": scores.mean(),
            "Std CV accuracy": scores.std()
        })

filter_cv_table = pd.DataFrame(filter_cv_results)
print("\nSelectKBest cross-validation results:")
print(filter_cv_table.to_string(index=False))

chosen_k = {}
for score_name in ["ANOVA F-score", "Mutual information"]:
    subset = filter_cv_table[filter_cv_table["Score"] == score_name]
    best_row = subset.loc[subset["Mean CV accuracy"].idxmax()]
    cutoff = best_row["Mean CV accuracy"] - best_row["Std CV accuracy"]
    eligible = subset[subset["Mean CV accuracy"] >= cutoff]
    chosen = int(eligible["k"].min())
    chosen_k[score_name] = chosen
    print(f"{score_name}: best mean={best_row['Mean CV accuracy']:.4f}; "
          f"one-standard-deviation cutoff={cutoff:.4f}; chosen smallest k={chosen}")

plt.figure(figsize=(9, 5))
for score_name in ["ANOVA F-score", "Mutual information"]:
    subset = filter_cv_table[filter_cv_table["Score"] == score_name]
    plt.errorbar(subset["k"], subset["Mean CV accuracy"],
                 yerr=subset["Std CV accuracy"], marker="o", capsize=3,
                 label=score_name)
plt.xlabel("Number of selected features (k)")
plt.ylabel("5-fold CV accuracy")
plt.title("SelectKBest: accuracy versus number of features")
plt.xticks(K_VALUES)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("""
Why can a univariate filter select redundant variables?
Each variable is scored independently against the target. If radius,
perimeter and area are each strongly associated with the target, all three
can receive high scores. A univariate filter does not penalize selecting
several variables that carry nearly the same information.
""")



print("\n" + "=" * 78)
print("PART C – WRAPPER METHODS")
print("=" * 78)

print("\nC.1 – Forward selection")
forward_estimator = make_model()
sfs_forward = SequentialFeatureSelector(
    estimator=forward_estimator,
    n_features_to_select=10,
    direction="forward",
    scoring="accuracy",
    cv=CV,
    n_jobs=-1
)
start = time.perf_counter()
sfs_forward.fit(X_train, y_train)
forward_sklearn_time = time.perf_counter() - start
forward_sklearn_features = list(X_train.columns[sfs_forward.get_support()])
print_result("Scikit-learn Forward Selection", forward_sklearn_features,
             X_train, y_train)
print(f"SequentialFeatureSelector elapsed time: {forward_sklearn_time:.2f}s")

start = time.perf_counter()
forward_manual_features, forward_history = manual_forward_selection(
    X_train, y_train, max_features=10, epsilon=0.001
)
forward_manual_time = time.perf_counter() - start
print_result("Manual Forward Selection", forward_manual_features, X_train, y_train)
print(f"Manual forward elapsed time: {forward_manual_time:.2f}s")
print("Same selected set:",
      set(forward_sklearn_features) == set(forward_manual_features))

if forward_history:
    forward_hist_df = pd.DataFrame(
        forward_history, columns=["Number of features", "CV accuracy",
                                  "CV std", "Added feature"]
    )
    print("\nManual forward selection history:\n", forward_hist_df.to_string(index=False))
    plt.figure(figsize=(8, 5))
    plt.plot(forward_hist_df["Number of features"],
             forward_hist_df["CV accuracy"], marker="o")
    plt.xlabel("Number of selected features")
    plt.ylabel("5-fold CV accuracy")
    plt.title("Manual Forward Selection")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("""
Forward-selection comparison:
The two implementations can differ because of tie-breaking, implementation
details, and the epsilon stopping rule. Scikit-learn is explicitly asked to
select 10 features, whereas the manual implementation may stop earlier when
the improvement is below 0.001.
""")

print("\nC.2 – Backward elimination")
sfs_backward = SequentialFeatureSelector(
    estimator=make_model(),
    n_features_to_select=10,
    direction="backward",
    scoring="accuracy",
    cv=CV,
    n_jobs=-1
)
start = time.perf_counter()
sfs_backward.fit(X_train, y_train)
backward_sklearn_time = time.perf_counter() - start
backward_sklearn_features = list(X_train.columns[sfs_backward.get_support()])
print_result("Scikit-learn Backward Elimination", backward_sklearn_features,
             X_train, y_train)
print(f"SequentialFeatureSelector elapsed time: {backward_sklearn_time:.2f}s")

start = time.perf_counter()
backward_manual_features, backward_history = manual_backward_selection(
    X_train, y_train, min_features=10
)
backward_manual_time = time.perf_counter() - start
print_result("Manual Backward Elimination", backward_manual_features,
             X_train, y_train)
print(f"Manual backward elapsed time: {backward_manual_time:.2f}s")

backward_hist_df = pd.DataFrame(
    backward_history, columns=["Remaining features", "CV accuracy",
                               "CV std", "Removed feature"]
)
print("\nManual backward elimination history:\n",
      backward_hist_df.to_string(index=False))
plt.figure(figsize=(8, 5))
plt.plot(backward_hist_df["Remaining features"],
         backward_hist_df["CV accuracy"], marker="o")
plt.xlabel("Number of remaining features")
plt.ylabel("5-fold CV accuracy")
plt.title("Manual Backward Elimination")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("""
Forward and backward subsets need not be identical. Forward selection
chooses features conditionally from an initially empty set; backward
elimination removes features conditionally from the complete set. With
correlated predictors, the value of a feature depends on which other
features are already present. Both are greedy procedures.
""")

print("\nC.3 – Recursive Feature Elimination")
rfe = wrapper_rfe(n_features=10)
start = time.perf_counter()
rfe.fit(X_train, y_train)
rfe_time = time.perf_counter() - start
rfe_features = list(X_train.columns[rfe.support_])
print("RFE selected features:", rfe_features)
print("RFE support_:", rfe.support_)
print("RFE ranking_:", rfe.ranking_)
print(f"RFE time: {rfe_time:.2f}s")

rfecv = make_rfecv()
start = time.perf_counter()
rfecv.fit(X_train, y_train)
rfecv_time = time.perf_counter() - start
rfecv_features = list(X_train.columns[rfecv.support_])
print("\nRFECV optimal number of features:", rfecv.n_features_)
print("RFECV selected features:", rfecv_features)
print(f"RFECV time: {rfecv_time:.2f}s")

rfecv_scores = rfecv.cv_results_["mean_test_score"]
rfecv_sizes = np.arange(1, len(rfecv_scores) + 1)
plt.figure(figsize=(8, 5))
plt.plot(rfecv_sizes, rfecv_scores, marker="o")
plt.axvline(rfecv.n_features_, linestyle="--",
            label=f"Selected: {rfecv.n_features_}")
plt.xlabel("Number of selected features")
plt.ylabel("Mean 5-fold CV accuracy")
plt.title("RFECV – cross-validation score versus feature count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

rfe_svc = wrapper_rfe(
    estimator=SVC(kernel="linear", random_state=RANDOM_STATE),
    n_features=10
)
rfe_svc.fit(X_train, y_train)
rfe_svc_features = list(X_train.columns[rfe_svc.support_])
print("\nRFE with linear SVC:", rfe_svc_features)
print("Same as logistic-regression RFE:",
      set(rfe_svc_features) == set(rfe_features))

print("""
RFE removes features using the estimator's importance (absolute coefficient
magnitude for linear logistic regression here), then refits the estimator.
Backward elimination instead evaluates the cross-validated predictive score
after each possible single-feature removal and removes the feature causing
the least decrease. RFE is generally cheaper because it does not evaluate
every candidate removal with a full cross-validation at every step.
Different estimators can select different features because their importance
criteria and fitted decision boundaries differ.
""")

print("\nC.4 – Cost of wrapper search")
p, k, folds = 30, 10, 5
forward_candidate_models = sum(p - i for i in range(k))
forward_cv_fits = forward_candidate_models * folds
backward_candidate_models = sum(range(k + 1, p + 1))
backward_cv_fits = backward_candidate_models * folds
all_subsets = 2 ** p
subsets_up_to_k = sum(comb(p, size) for size in range(k + 1))
print(f"Forward candidates evaluated: {forward_candidate_models}")
print(f"Forward CV model fits (5 folds): {forward_cv_fits}")
print(f"Backward candidates evaluated from 30 to 10: {backward_candidate_models}")
print(f"Backward CV model fits (5 folds): {backward_cv_fits}")
print(f"All possible subsets: 2^{p} = {all_subsets:,}")
print(f"Subsets of size 0 through {k}: {subsets_up_to_k:,}")
print("""
For forward selection, the number of candidate subsets evaluated is:
    p + (p-1) + ... + (p-k+1) = kp - k(k-1)/2.
For p=30 and k=10, this is 255 candidate subsets, or 1,275 model fits
with 5-fold CV. An exhaustive search over all subsets would consider
2^30 = 1,073,741,824 subsets (over one billion), before accounting for
cross-validation. This explains why wrapper search is computationally
expensive.
""")


print("\n" + "=" * 78)
print("PART D – NOISE INJECTION, LEAKAGE AND FINAL COMPARISON")
print("=" * 78)

rng = np.random.default_rng(RANDOM_STATE)
noise = pd.DataFrame(
    rng.standard_normal((len(X), 20)),
    columns=[f"noise_{i:02d}" for i in range(20)],
    index=X.index
)
X_with_noise = pd.concat([X, noise], axis=1)
Xn_train = X_with_noise.loc[X_train.index]
Xn_test = X_with_noise.loc[X_test.index]
noise_names = list(noise.columns)

print("\nD.1 – Can the methods find the noise?")
noise_selection = {}

noise_f_selector = SelectKBest(score_func=f_classif, k=10)
noise_f_selector.fit(Xn_train, y_train)
noise_selection["SelectKBest F-score"] = list(
    Xn_train.columns[noise_f_selector.get_support()]
)

noise_forward = SequentialFeatureSelector(
    estimator=make_model(), n_features_to_select=10,
    direction="forward", scoring="accuracy", cv=CV, n_jobs=-1
)
noise_forward.fit(Xn_train, y_train)
noise_selection["Forward Selection"] = list(
    Xn_train.columns[noise_forward.get_support()]
)

noise_backward = SequentialFeatureSelector(
    estimator=make_model(), n_features_to_select=10,
    direction="backward", scoring="accuracy", cv=CV, n_jobs=-1
)
noise_backward.fit(Xn_train, y_train)
noise_selection["Backward Elimination"] = list(
    Xn_train.columns[noise_backward.get_support()]
)

noise_rfe = wrapper_rfe(n_features=10)
noise_rfe.fit(Xn_train, y_train)
noise_selection["RFE"] = list(Xn_train.columns[noise_rfe.support_])

noise_rows = []
for method, selected in noise_selection.items():
    noise_count = sum(name in noise_names for name in selected)
    noise_rows.append({
        "Method": method,
        "Selected features": len(selected),
        "Noise features selected": noise_count,
        "Real features selected": len(selected) - noise_count
    })
    print(f"\n{method}:")
    print(selected)
noise_table = pd.DataFrame(noise_rows)
print("\nNoise-selection summary:\n", noise_table.to_string(index=False))

print("""
Interpretation:
Noise variables are independent of the target by construction. A method
that selects many noise variables is more easily fooled. Univariate filters
can select noise due to random sample associations, especially when many
candidate variables are tested. Wrapper methods evaluate predictive
performance and can reject some irrelevant variables, but they are not
immune to overfitting, particularly when the sample is small and the number
of candidate features is large. Use the actual counts printed above to
identify the most robust method in this run.
""")

print("\nD.2 – Leakage in feature selection")

leaky_selector = SelectKBest(score_func=f_classif, k=10)
Xn_train_leaky = leaky_selector.fit_transform(Xn_train, y_train)
incorrect_scores = cross_val_score(
    make_model(), Xn_train_leaky, y_train, cv=CV, scoring="accuracy"
)

correct_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(score_func=f_classif, k=10)),
    ("classifier", LogisticRegression(max_iter=1000,
                                      random_state=RANDOM_STATE))
])
correct_scores = cross_val_score(
    correct_pipeline, Xn_train, y_train, cv=CV, scoring="accuracy"
)

print(f"Incorrect CV accuracy: {incorrect_scores.mean():.4f} "
      f"+/- {incorrect_scores.std():.4f}")
print(f"Correct CV accuracy:   {correct_scores.mean():.4f} "
      f"+/- {correct_scores.std():.4f}")

print("""
Explanation:
In the incorrect approach, feature selection sees the labels of every
training observation, including observations later used as validation folds.
The validation labels therefore influence which features are selected,
making the CV estimate optimistically biased. In the correct Pipeline,
scaling and selection are fitted only on each fold's training portion.
The difference may be small here because there are 569 observations and
only 50 predictors, but it can become substantial with many features,
few observations, or weak signals.
""")

print("\nD.3 – Final comparison on the test set (30 original features)")

final_methods = []

final_methods.append({
    "Method": "Baseline (all features)",
    "Features": feature_names,
    "Selector": None,
    "Selection time (s)": 0.0
})

start = time.perf_counter()
corr_final_selector = CorrelationSelector(threshold=0.90)
corr_final_selector.fit(X_train, y_train)
corr_final_time = time.perf_counter() - start
corr_final_features = corr_final_selector.selected_features_
final_methods.append({
    "Method": "Filter: correlation (0.90)",
    "Features": corr_final_features,
    "Selector": CorrelationSelector(threshold=0.90),
    "Selection time (s)": corr_final_time
})

for score_name, score_function in [
    ("F-score", f_classif),
    ("mutual information", lambda a, b: (
        mutual_info_classif(a, b, random_state=RANDOM_STATE), None
    ))
]:
    k = chosen_k["ANOVA F-score" if score_name == "F-score"
                 else "Mutual information"]
    start = time.perf_counter()
    selector = SelectKBest(score_func=score_function, k=k)
    selector.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    selected = list(X_train.columns[selector.get_support()])
    final_methods.append({
        "Method": f"Filter: SelectKBest ({score_name})",
        "Features": selected,
        "Selector": SelectKBest(score_func=score_function, k=k),
        "Selection time (s)": elapsed
    })

final_methods.append({
    "Method": "Wrapper: Forward Selection",
    "Features": forward_sklearn_features,
    "Selector": FixedFeatureSelector(forward_sklearn_features),
    "Selection time (s)": forward_sklearn_time
})
final_methods.append({
    "Method": "Wrapper: Backward Elimination",
    "Features": backward_sklearn_features,
    "Selector": FixedFeatureSelector(backward_sklearn_features),
    "Selection time (s)": backward_sklearn_time
})
final_methods.append({
    "Method": "Wrapper: RFE",
    "Features": rfe_features,
    "Selector": wrapper_rfe(n_features=10),
    "Selection time (s)": rfe_time
})
final_methods.append({
    "Method": "Wrapper: RFECV",
    "Features": rfecv_features,
    "Selector": make_rfecv(),
    "Selection time (s)": rfecv_time
})

final_rows = []
presence = pd.DataFrame(index=feature_names)

for item in final_methods:
    method = item["Method"]
    selected_features = item["Features"]

    if item["Selector"] is None:
        model = make_model()
    else:
        model = Pipeline([
            ("selector", item["Selector"]),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000,
                                              random_state=RANDOM_STATE))
        ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    final_rows.append({
        "Method": method,
        "Nb features": len(selected_features),
        "Test accuracy": accuracy_score(y_test, predictions),
        "Test F1-score": f1_score(y_test, predictions),
        "Selection time (s)": item["Selection time (s)"]
    })

    presence[method] = [
        1 if feature in selected_features else 0 for feature in feature_names
    ]

final_table = pd.DataFrame(final_rows)
print("\nFINAL TEST COMPARISON:")
print(final_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

print("\nFeature presence matrix (1 = selected, 0 = not selected):")
print(presence.to_string())

common_features = [
    feature for feature in feature_names
    if all(feature in item["Features"] for item in final_methods[1:])
]
print("\nFeatures selected by every selection method:", common_features)

plt.figure(figsize=(15, 10))
sns.heatmap(presence, cmap="Blues", cbar=False, linewidths=0.2,
            linecolor="lightgray")
plt.title("Feature selection presence matrix (1 = selected)")
plt.xlabel("Selection method")
plt.ylabel("Original feature")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()

print("""
FINAL DISCUSSION

1. Best test accuracy with the fewest features:
   Compare the Test accuracy and Nb features columns in the final table.
   A method is attractive when it obtains accuracy close to the 30-feature
   baseline with substantially fewer predictors. The test set contains
   approximately 171 observations, so a small difference in accuracy may
   correspond to only one or two observations and should not automatically
   be considered statistically significant.

2. Filter methods versus wrapper methods:
   Filter methods are generally fast and model-independent. They are useful
   for an initial reduction of very high-dimensional data, but univariate
   scores can retain redundant features and do not directly measure model
   performance. Wrapper methods evaluate feature subsets using a predictive
   model and can account for feature interactions, but they are slower and
   their selected subset depends on the estimator.

3. Why wrappers are more expensive:
   A wrapper repeatedly trains models to compare candidate subsets. Forward
   selection evaluates kp - k(k-1)/2 candidates when selecting k features
   from p features. With p=30 and k=10, that is 255 candidates, or 1,275
   fits under 5-fold CV.

4. Greedy search and optimality:
   Forward Selection and Backward Elimination make locally best decisions
   at each step. They do not examine every possible subset and therefore
   do not guarantee the globally optimal feature subset.

5. Behaviour with noise:
   Use the D.1 noise-selection summary. Selecting noise features indicates
   sensitivity to chance correlations. The comparison is empirical and
   depends on this random sample and the model; no family is guaranteed to
   reject every irrelevant feature.

6. Why selection belongs inside cross-validation:
   If selection is performed before CV, validation-fold labels influence
   the selected features. This leaks information and can inflate the score.
   A Pipeline ensures that scaling, selection and model fitting are learned
   only from each fold's training data.

7. When to choose each family:
   - Thousands of features / limited computing time: begin with a filter
     method to reduce dimensionality quickly.
   - Small or moderate feature sets and sufficient computing resources:
     wrapper methods can search for subsets tailored to the model.
   - Need for interpretability: correlation filtering or SelectKBest can
     provide a simple ranking; wrapper subsets can also be interpreted but
     are model-dependent.
   - Limited computing time: prefer filters; avoid exhaustive wrapper search.
""")

print("\nFinished. All plots and numerical tables have been generated.")
