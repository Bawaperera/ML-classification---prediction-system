"""
SHAP-based explainability for the churn prediction model.

SHAP (SHapley Additive exPlanations) uses game theory to explain individual
predictions. Unlike feature importance, SHAP shows both direction and magnitude
for each feature on each individual prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def compute_shap_values(model, X, sample_size=None):
    """
    Compute SHAP values using TreeExplainer (works for XGBoost, RF, GBM).

    Returns the explainer and a shap.Explanation object.
    Set sample_size to limit computation time on large datasets.
    """
    if sample_size and X.shape[0] > sample_size:
        idx = np.random.RandomState(42).choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    return explainer, shap_values, X_sample


def plot_beeswarm(shap_values, max_display=20, save_path=None):
    """
    SHAP beeswarm (summary) plot — the most informative global view.

    Each dot is one prediction. Position on x-axis = SHAP value (impact on output).
    Colour = feature value (red = high, blue = low).
    Features are sorted by mean absolute SHAP value (most impactful at top).
    """
    plt.figure(figsize=(10, max_display * 0.4 + 2))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_bar_summary(shap_values, max_display=15, save_path=None):
    """
    SHAP bar plot — simpler view showing mean absolute SHAP value per feature.
    Less detail than beeswarm but easier to read at a glance.
    """
    plt.figure(figsize=(9, max_display * 0.4 + 1))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_dependence(shap_values, X_sample, feature, feature_names,
                    interaction_feature=None, save_path=None):
    """
    SHAP dependence plot — shows how one feature's value affects its SHAP contribution.

    If interaction_feature is provided, points are coloured by that second feature,
    which reveals interactions between the two features.
    """
    feat_idx = list(feature_names).index(feature) if feature in feature_names else feature

    interact_idx = None
    if interaction_feature and interaction_feature in feature_names:
        interact_idx = list(feature_names).index(interaction_feature)

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values.values,
        X_sample,
        feature_names=feature_names,
        interaction_index=interact_idx,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP Dependence — {feature}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_waterfall(shap_values, index, feature_names=None, save_path=None):
    """
    SHAP waterfall plot — explains one specific prediction.

    Shows the chain: base value -> each feature pushes up or down -> final prediction.
    Use this to answer "why did the model predict churn for THIS customer?"
    """
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], max_display=15, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def get_top_churn_drivers(shap_values, feature_names, top_n=10):
    """
    Returns a DataFrame with the top N features ranked by mean absolute SHAP value.
    This is the global importance ranking — which features drive predictions the most.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    drivers = pd.DataFrame({
        "feature":    feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    drivers.index += 1
    return drivers


def find_example_indices(model, X_test_proc, y_test, optimal_threshold=0.5):
    """
    Finds four useful example predictions for the individual explanation section:
    - high_risk_tp:  high probability churner who actually churned (correct)
    - low_risk_tn:   low probability non-churner who actually stayed (correct)
    - false_neg:     churner the model missed (predicted stay, actually left)
    - false_pos:     non-churner the model flagged (predicted churn, actually stayed)

    Returns a dict of {label: index_in_X_test_proc}.
    """
    y_prob  = model.predict_proba(X_test_proc)[:, 1]
    y_pred  = (y_prob >= optimal_threshold).astype(int)
    y_true  = np.array(y_test)

    tp_idx = np.where((y_pred == 1) & (y_true == 1))[0]
    tn_idx = np.where((y_pred == 0) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]

    # pick most confident examples for clarity
    high_risk_tp = tp_idx[np.argmax(y_prob[tp_idx])]       if len(tp_idx) else 0
    low_risk_tn  = tn_idx[np.argmin(y_prob[tn_idx])]       if len(tn_idx) else 1
    false_neg    = fn_idx[np.argmax(y_prob[fn_idx])]       if len(fn_idx) else 2
    false_pos    = fp_idx[np.argmax(y_prob[fp_idx])]       if len(fp_idx) else 3

    return {
        "high_risk_tp": int(high_risk_tp),
        "low_risk_tn":  int(low_risk_tn),
        "false_neg":    int(false_neg),
        "false_pos":    int(false_pos),
    }
