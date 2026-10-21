"""Breast cancer diagnosis: replicate the course results, compare models with
repeated cross-validation, and tune the decision threshold.

Run from the project root:
    python src/train.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from style import BLUE, GRAY, INK, MUTED, ORANGE, RED

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42
TARGET_RECALL = 0.99  # share of malignant tumors the tuned threshold must catch
MODEL_COLORS = {"Logistic Regression": BLUE, "SVM": ORANGE, "Random Forest": "#1baf7a",
                "Decision Tree": "#eda100"}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Wisconsin Diagnostic Breast Cancer data, with malignant as the positive class.

    scikit-learn codes malignant as 0 and benign as 1. Metrics like recall and
    F1 are computed for class 1 by default, so with the original coding they
    describe benign tumors. Flipping the labels makes "recall" mean "share of
    cancers caught", which is the number that matters clinically.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns="target")
    y = (data.frame["target"] == 0).astype(int).rename("malignant")
    return X, y


def build_models() -> dict:
    # Scaled models as in the course project; trees use raw features.
    return {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=5000, random_state=SEED)),
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=SEED)),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }


def scores(model, X) -> np.ndarray:
    """Probability of malignant if available, otherwise the SVM decision value."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def test_metrics(y_true, pred, score) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, score)), 4),
        "missed_cancers": int(((y_true == 1) & (pred == 0)).sum()),
        "false_alarms": int(((y_true == 0) & (pred == 1)).sum()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_repeated_cv(cv_results: dict) -> None:
    names = sorted(cv_results, key=lambda n: cv_results[n]["accuracy_mean"])
    means = [cv_results[n]["accuracy_mean"] * 100 for n in names]
    stds = [cv_results[n]["accuracy_std"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.errorbar(means, names, xerr=stds, fmt="o", color=BLUE, markersize=8,
                ecolor=BLUE, elinewidth=2, capsize=4)
    ax.grid(axis="x", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    for m, s, n in zip(means, stds, names):
        ax.text(m + s + 0.4, n, f"{m:.1f}% ± {s:.1f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(88, 101)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("Accuracy across 50 cross-validation folds (dot = mean, line = ± 1 std)")
    ax.set_title("Model comparison with repeated cross-validation")
    ax.tick_params(axis="y", colors=INK, length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison_cv.png")
    plt.close(fig)


def plot_roc(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[name], label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random guess")
    ax.set_xlabel("False positive rate (benign flagged as malignant)")
    ax.set_ylabel("True positive rate (malignant caught)")
    ax.set_title("ROC curves on the test set")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png")
    plt.close(fig)


def plot_confusion_pair(cms: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    labels = ["Benign", "Malignant"]
    for ax, (title, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.tick_params(length=0, colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(2):
            for j in range(2):
                dark = cm[i, j] > cm.max() * 0.6
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                        color="white" if dark else INK)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Actual")
    fig.suptitle("Logistic Regression on the 114 test patients", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png")
    plt.close(fig)


def plot_threshold(y_train, oof_prob, chosen: float) -> None:
    thresholds = np.linspace(0.02, 0.98, 97)
    recall = [recall_score(y_train, oof_prob >= t) * 100 for t in thresholds]
    precision = [precision_score(y_train, oof_prob >= t, zero_division=1) * 100 for t in thresholds]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.plot(thresholds, recall, color=RED, lw=2, label="Recall: malignant tumors caught")
    ax.plot(thresholds, precision, color=BLUE, lw=2, label="Precision: flagged tumors that are malignant")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.axvline(chosen, color=INK, lw=1, ls="--")
    ax.text(0.5, 101.3, " default 0.5", color=MUTED, fontsize=9)
    ax.text(chosen, 101.3, f" tuned {chosen:.2f}", color=INK, fontsize=9)
    ax.text(0.58, 83.6, "Recall: share of\nmalignant tumors caught", color=RED, fontsize=9)
    ax.text(0.17, 87.0, "Precision: share of flagged\ntumors that are malignant", color=BLUE, fontsize=9)
    ax.set_ylim(80, 103)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Decision threshold (predict malignant if probability ≥ threshold)")
    ax.set_ylabel("Percent  ·  axis starts at 80%")
    ax.set_title("Trading false alarms for missed cancers (training-set CV)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_tradeoff.png")
    plt.close(fig)


def plot_coefficients(model, feature_names) -> pd.Series:
    coefs = pd.Series(model[-1].coef_.ravel(), index=feature_names)
    top = pd.concat([coefs.nsmallest(5), coefs.nlargest(6).iloc[::-1]])
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.barh(top.index, top.values, color=[RED if v > 0 else BLUE for v in top.values], height=0.55)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Coefficient on standardized feature  (→ malignant   ·   ← benign)")
    ax.set_title("Strongest features in each direction (Logistic Regression)")
    ax.tick_params(axis="y", colors=INK, length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_features.png")
    plt.close(fig)
    return coefs


# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    # Stratify on scikit-learn's original coding (benign = 1) so the split, and
    # therefore the test patients, are exactly the ones the course project used.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=1 - y, random_state=SEED
    )
    out = {"n_samples": len(X), "n_malignant": int(y.sum()), "n_train": len(X_train),
           "n_test": len(X_test), "n_test_malignant": int(y_test.sum())}

    # 1. Same split as the course project: test-set results.
    single_split, curves, fitted = {}, {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred, score = model.predict(X_test), scores(model, X_test)
        single_split[name] = test_metrics(y_test, pred, score)
        # Benign-as-positive F1, the definition used on the slides.
        single_split[name]["f1_benign_positive"] = round(float(f1_score(1 - y_test, 1 - pred)), 4)
        fpr, tpr, _ = roc_curve(y_test, score)
        curves[name] = (fpr, tpr, single_split[name]["roc_auc"])
        fitted[name] = model
    out["single_split"] = single_split

    # 2. Repeated stratified 5-fold CV (10 repeats = 50 folds) on all 569 samples.
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    cv_results = {}
    for name, model in build_models().items():
        res = cross_validate(model, X, y, cv=rcv, scoring=["accuracy", "recall", "roc_auc"])
        cv_results[name] = {
            f"{m}_{s}": round(float(getattr(res[f"test_{m}"], s)()), 4)
            for m in ["accuracy", "recall", "roc_auc"] for s in ["mean", "std"]
        }
    out["repeated_cv"] = cv_results

    # 3. Threshold tuning for the chosen model, using only training data.
    best = fitted["Logistic Regression"]
    oof = cross_val_predict(clone(best), X_train, y_train, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
    candidates = np.round(np.arange(0.95, 0.0, -0.01), 2)
    chosen = next(t for t in candidates if recall_score(y_train, oof >= t) >= TARGET_RECALL)
    prob_test = best.predict_proba(X_test)[:, 1]
    tuned_pred = (prob_test >= chosen).astype(int)
    out["threshold"] = {
        "target_recall": TARGET_RECALL,
        "chosen_threshold": float(chosen),
        "train_oof_at_0.5": test_metrics(y_train, (oof >= 0.5).astype(int), oof),
        "train_oof_at_chosen": test_metrics(y_train, (oof >= chosen).astype(int), oof),
        "test_at_0.5": single_split["Logistic Regression"],
        "test_at_chosen": test_metrics(y_test, tuned_pred, prob_test),
    }

    plot_repeated_cv(cv_results)
    plot_roc(curves)
    plot_confusion_pair({
        "Default threshold 0.5": confusion_matrix(y_test, best.predict(X_test)),
        f"Tuned threshold {chosen:.2f}": confusion_matrix(y_test, tuned_pred),
    })
    plot_threshold(y_train, oof, chosen)
    coefs = plot_coefficients(best, X.columns)
    out["top_coefficients"] = {"toward_malignant": {k: round(float(v), 3) for k, v in coefs.nlargest(6).items()},
                              "toward_benign": {k: round(float(v), 3) for k, v in coefs.nsmallest(5).items()}}

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(single_split).T.to_string())
    print()
    print(pd.DataFrame(cv_results).T.to_string())
    print()
    print(json.dumps(out["threshold"], indent=1))
    print(out["top_coefficients"])


if __name__ == "__main__":
    main()
