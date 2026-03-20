"""
Standalone training script — trains the best model from pre-processed data
already stored in data/processed/ and saves the artefacts needed by the dashboard.

Run from the repo root:
    python train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

# so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import create_preprocessor

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR     = os.path.join(BASE_DIR, "models")

TRAIN_PATH    = os.path.join(DATA_DIR,  "train.csv")
TEST_PATH     = os.path.join(DATA_DIR,  "test.csv")
PREP_PATH     = os.path.join(MODEL_DIR, "preprocessor.joblib")
MODEL_PATH    = os.path.join(MODEL_DIR, "best_model.joblib")
META_PATH     = os.path.join(MODEL_DIR, "model_meta.json")

# ---------------------------------------------------------------------------
# feature groups — must match notebook 02
# ---------------------------------------------------------------------------
NUMERICAL_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "monthly_tenure_ratio", "total_services", "avg_monthly_spend",
    "contract_risk",
]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_group",
]

# binary engineered features that pass through the remainder transformer
BINARY_ENGINEERED = [
    "has_security_bundle", "has_support", "has_streaming_bundle",
    "is_new_customer", "is_high_value",
]


def load_split(path: str):
    df = pd.read_csv(path)
    y = df["Churn"].astype(int)
    X = df.drop("Churn", axis=1)
    return X, y


def find_optimal_threshold_cv(model_params, X_train_proc, y_train_arr, n_splits: int = 5):
    """
    Find the optimal decision threshold via cross-validation on the training set.

    For each fold the held-out probabilities are used to sweep thresholds and pick
    the one with the best F1.  The per-fold optimal thresholds are averaged so the
    final threshold is never tuned on the test set.

    Returns
    -------
    optimal_threshold : float
    cv_f1_mean        : float – mean F1 across folds at each fold's best threshold
    cv_f1_std         : float
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    thresholds_per_fold = []
    f1_scores_per_fold  = []

    for train_idx, val_idx in skf.split(X_train_proc, y_train_arr):
        X_tr, X_val = X_train_proc[train_idx], X_train_proc[val_idx]
        y_tr, y_val = y_train_arr[train_idx],  y_train_arr[val_idx]

        fold_model = RandomForestClassifier(**model_params)
        fold_model.fit(X_tr, y_tr)
        y_prob_val = fold_model.predict_proba(X_val)[:, 1]

        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.10, 0.90, 0.01):
            y_pred = (y_prob_val >= t).astype(int)
            score  = f1_score(y_val, y_pred, zero_division=0)
            if score > best_f:
                best_f, best_t = score, t

        thresholds_per_fold.append(best_t)
        f1_scores_per_fold.append(best_f)

    optimal_threshold = float(np.mean(thresholds_per_fold))
    cv_f1_mean        = float(np.mean(f1_scores_per_fold))
    cv_f1_std         = float(np.std(f1_scores_per_fold))
    return optimal_threshold, cv_f1_mean, cv_f1_std


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load processed data (already feature-engineered by notebook 02)
    # ------------------------------------------------------------------
    print("Loading processed data …")
    X_train, y_train = load_split(TRAIN_PATH)
    X_test,  y_test  = load_split(TEST_PATH)
    print(f"  Train: {X_train.shape}  churn rate: {y_train.mean():.1%}")
    print(f"  Test : {X_test.shape}   churn rate: {y_test.mean():.1%}")

    # ------------------------------------------------------------------
    # 2. Build and fit the preprocessor
    # ------------------------------------------------------------------
    print("\nFitting preprocessor …")
    preprocessor = create_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    print(f"  Feature matrix: {X_train_proc.shape[1]} columns after OHE")

    joblib.dump(preprocessor, PREP_PATH)
    print(f"  Saved: {PREP_PATH}")

    # ------------------------------------------------------------------
    # 3. Find optimal threshold and CV metrics via cross-validation
    #    (threshold is tuned on training folds only — never on the test set)
    # ------------------------------------------------------------------
    RF_PARAMS = dict(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nRunning 5-fold CV to find optimal threshold …")
    optimal_threshold, cv_f1_mean, cv_f1_std = find_optimal_threshold_cv(
        RF_PARAMS, X_train_proc, y_train.values
    )
    print(f"  CV optimal threshold : {optimal_threshold:.2f}")
    print(f"  CV F1 mean           : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    # ------------------------------------------------------------------
    # 4. Train the final model on all training data
    # ------------------------------------------------------------------
    print("\nTraining final Random Forest on all training data …")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_proc, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate on the held-out test set (read-only — no threshold tuning)
    # ------------------------------------------------------------------
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    metrics = {
        "model_name":           "Random Forest",
        "optimal_threshold":    round(optimal_threshold, 2),
        "tuned_f1_test":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "tuned_roc_auc_test":   round(float(roc_auc_score(y_test, y_prob)), 4),
        "tuned_precision_test": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "tuned_recall_test":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "tuned_accuracy_test":  round(float(accuracy_score(y_test, y_pred)), 4),
        "cv_f1_mean":           round(cv_f1_mean, 4),
        "cv_f1_std":            round(cv_f1_std, 4),
    }

    print("\nTest-set metrics (threshold from CV):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 6. Save model and metadata
    # ------------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")

    with open(META_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Saved: {META_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
