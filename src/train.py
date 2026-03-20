"""
Model training pipeline — trains and compares 6 classification models.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not found — XGBoost model will be skipped.")


class ModelTrainer:
    """
    Trains multiple classifiers and keeps the results for comparison.

    Usage:
        trainer = ModelTrainer(X_train, y_train, X_test, y_test)
        comparison_df = trainer.train_all()
        trainer.save_models()
    """

    def __init__(self, X_train, y_train, X_test, y_test, feature_names=None):
        self.X_train       = X_train
        self.y_train       = y_train
        self.X_test        = X_test
        self.y_test        = y_test
        self.feature_names = feature_names
        self.models        = {}
        self.results       = {}
        self.training_times = {}

    def _train_and_evaluate(self, name: str, model):
        """Train one model and compute all evaluation metrics."""
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        start = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        y_pred = model.predict(self.X_test)
        y_prob = (
            model.predict_proba(self.X_test)[:, 1]
            if hasattr(model, "predict_proba") else None
        )

        metrics = {
            "accuracy":           accuracy_score(self.y_test, y_pred),
            "precision":          precision_score(self.y_test, y_pred, zero_division=0),
            "recall":             recall_score(self.y_test, y_pred, zero_division=0),
            "f1":                 f1_score(self.y_test, y_pred, zero_division=0),
            "roc_auc":            roc_auc_score(self.y_test, y_prob) if y_prob is not None else None,
            "train_time_seconds": round(train_time, 2),
        }

        self.models[name]         = model
        self.results[name]        = metrics
        self.training_times[name] = train_time

        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")
        if metrics["roc_auc"]:
            print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"  Time      : {train_time:.2f}s")

        return model

    def train_logistic_regression(self):
        """Logistic Regression — simplest baseline. Good starting point."""
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver="lbfgs",
        )
        return self._train_and_evaluate("Logistic Regression", model)

    def train_decision_tree(self):
        """Decision Tree — interpretable but prone to overfitting."""
        model = DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            min_samples_split=20,
            random_state=42,
        )
        return self._train_and_evaluate("Decision Tree", model)

    def train_random_forest(self):
        """Random Forest — ensemble of trees. Usually a strong baseline."""
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
        return self._train_and_evaluate("Random Forest", model)

    def train_gradient_boosting(self):
        """Gradient Boosting — each tree corrects the errors of the previous one."""
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        return self._train_and_evaluate("Gradient Boosting", model)

    def train_xgboost(self):
        """
        XGBoost — strong regularisation, handles tabular data well.
        Uses scale_pos_weight to deal with class imbalance.
        """
        if not XGBOOST_AVAILABLE:
            print("  Skipping XGBoost (not installed).")
            return None

        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        return self._train_and_evaluate("XGBoost", model)

    def train_svm(self):
        """
        SVM with RBF kernel.
        probability=True is needed for predict_proba and ROC-AUC.
        Slower than the tree models on this size of dataset.
        """
        model = SVC(
            class_weight="balanced",
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42,
        )
        return self._train_and_evaluate("SVM", model)

    def train_all(self) -> pd.DataFrame:
        """Train all 6 models in sequence and return a comparison DataFrame."""
        print("\n" + "="*50)
        print("  TRAINING ALL 6 MODELS")
        print("="*50)

        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        self.train_svm()

        return self.get_comparison_df()

    def get_comparison_df(self) -> pd.DataFrame:
        """Returns a DataFrame with all model results sorted by F1 score."""
        df = pd.DataFrame(self.results).T
        return df.sort_values("f1", ascending=False).round(4)

    def save_models(self, directory: str = "models/baseline_models"):
        """Save all trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            safe_name = name.lower().replace(" ", "_")
            path = os.path.join(directory, f"{safe_name}.joblib")
            joblib.dump(model, path)
            print(f"  Saved: {path}")

    def get_top_n_models(self, n: int = 2, metric: str = "f1") -> list:
        """Returns names of the top N models sorted by the given metric."""
        return self.get_comparison_df().head(n).index.tolist()

    def print_classification_reports(self, top_n: int = 3):
        """Prints sklearn classification_report for the top N models."""
        for name in self.get_top_n_models(n=top_n):
            y_pred = self.models[name].predict(self.X_test)
            print(f"\n{'─'*50}")
            print(f"  {name}")
            print(f"{'─'*50}")
            print(classification_report(
                self.y_test, y_pred,
                target_names=["No Churn", "Churn"],
            ))
