"""
Evaluation functions for the churn prediction model.
Covers confusion matrix, ROC, PR curves, feature importance, learning curves, and error analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import learning_curve


def plot_confusion_matrix(y_true, y_pred, model_name="Model", ax=None):
    """
    Plots a confusion matrix with both raw counts and percentages.
    Labels each quadrant with its business meaning.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # build annotation labels
    labels = np.array([
        [f"TN\n{cm[0,0]}\n({cm_pct[0,0]:.1f}%)\nCorrectly predicted STAY",
         f"FP\n{cm[0,1]}\n({cm_pct[0,1]:.1f}%)\nFalse alarm — stayed"],
        [f"FN\n{cm[1,0]}\n({cm_pct[1,0]:.1f}%)\nMISSED churner",
         f"TP\n{cm[1,1]}\n({cm_pct[1,1]:.1f}%)\nCaught churner"],
    ])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(cm_pct, annot=labels, fmt="", cmap="Blues", ax=ax,
                linewidths=1, linecolor="white",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                cbar_kws={"label": "Row %"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")

    return ax


def plot_roc_curves(models_dict, X_test, y_test, ax=None):
    """
    Plots ROC curves for multiple models on the same axes.
    models_dict: {name: fitted_model}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("tab10", len(models_dict))
    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    return ax


def plot_pr_curves(models_dict, X_test, y_test, ax=None):
    """
    Plots Precision-Recall curves for multiple models.
    For imbalanced datasets PR curves are more informative than ROC curves
    because they focus on the minority class performance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    baseline = y_test.mean()
    colors = sns.color_palette("tab10", len(models_dict))

    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})",
                color=color, linewidth=2)

    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (churn rate = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    return ax


def plot_feature_importance(model, feature_names, top_n=20, ax=None):
    """
    Horizontal bar chart of top N feature importances.
    Works for tree-based models (RF, XGBoost, GBM).
    """
    if not hasattr(model, "feature_importances_"):
        print(f"Model does not have feature_importances_ attribute.")
        return None

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    colors = sns.color_palette("viridis", len(importances))
    importances.plot.barh(ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")

    return ax


def plot_learning_curves(model, X, y, cv=5, scoring="f1", ax=None):
    """
    Plots training and validation score vs training set size.
    Useful for diagnosing overfitting or underfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    colors = sns.color_palette("Set2", 2)
    ax.plot(train_sizes, train_mean, label="Training score", color=colors[0], linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=colors[0])
    ax.plot(train_sizes, val_mean, label="Validation score", color=colors[1], linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=colors[1])

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(f"Score ({scoring})")
    ax.set_title("Learning Curves", fontweight="bold")
    ax.legend()

    return ax


def plot_threshold_analysis(y_true, y_prob, ax=None):
    """
    Shows how precision, recall, and F1 change as the classification threshold moves.
    """
    thresholds = np.arange(0.1, 0.91, 0.02)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))

    optimal_t = thresholds[np.argmax(f1s)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    colors = sns.color_palette("Set2", 3)
    ax.plot(thresholds, precisions, label="Precision", color=colors[0], linewidth=2)
    ax.plot(thresholds, recalls,    label="Recall",    color=colors[1], linewidth=2)
    ax.plot(thresholds, f1s,        label="F1",        color=colors[2], linewidth=2)
    ax.axvline(0.5,       color="gray", linestyle="--", linewidth=1.2, label="Default (0.5)")
    ax.axvline(optimal_t, color="red",  linestyle="--", linewidth=1.5,
               label=f"Optimal ({optimal_t:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Classification Threshold", fontweight="bold")
    ax.legend()

    return ax, optimal_t


def get_misclassified(model, X_test, y_test, X_test_raw, threshold=0.5, n=5):
    """
    Returns DataFrames of the worst false negatives and false positives.

    False negatives (missed churners) are ranked by how confidently wrong the model was.
    False positives (false alarms) are ranked the same way.

    X_test_raw: the un-preprocessed test features (readable column names)
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    results = X_test_raw.copy().reset_index(drop=True)
    results["churn_prob"]  = y_prob
    results["actual"]      = y_test.values
    results["predicted"]   = y_pred

    # false negatives: actual=1, predicted=0, sorted by highest probability (most confident wrong)
    fn = results[(results["actual"] == 1) & (results["predicted"] == 0)]
    fn = fn.sort_values("churn_prob", ascending=False).head(n)

    # false positives: actual=0, predicted=1, sorted by highest probability
    fp = results[(results["actual"] == 0) & (results["predicted"] == 1)]
    fp = fp.sort_values("churn_prob", ascending=False).head(n)

    return fn, fp
