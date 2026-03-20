"""
ML Classification & Prediction System
Feature Engineering Pipeline for Churn Prediction

Transforms raw cleaned data into an ML-ready feature matrix.
10 engineered features that encode business knowledge about churn.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 10 new features that encode business logic about churn behaviour.

    New features:
    1.  tenure_group           — bucketed tenure ('0-12', '13-24', '25-48', '49-72')
    2.  monthly_tenure_ratio   — MonthlyCharges / (tenure + 1)
    3.  total_services         — count of active add-on services
    4.  has_security_bundle    — 1 if both OnlineSecurity AND OnlineBackup == 'Yes'
    5.  has_support            — 1 if TechSupport == 'Yes'
    6.  has_streaming_bundle   — 1 if both StreamingTV AND StreamingMovies == 'Yes'
    7.  is_new_customer        — 1 if tenure <= 6 months
    8.  is_high_value          — 1 if MonthlyCharges > 70 (above median)
    9.  contract_risk          — ordinal risk score (Month-to-month=2, One year=1, Two year=0)
    10. avg_monthly_spend      — TotalCharges / (tenure + 1)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of clean_data).

    Returns
    -------
    pd.DataFrame
        Original columns + 10 new engineered columns.
    """
    df = df.copy()

    # 1. Tenure group (categorical bucket)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True,
    ).astype(str)   # cast to str so it flows cleanly through OHE

    # 2. Monthly-to-tenure ratio (avoid divide-by-zero)
    df["monthly_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # 3. Total active services
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        if col in df.columns:
            df[col + "_flag"] = df[col].apply(
                lambda x: 1 if x in ("Yes", "Fiber optic", "DSL") else 0
            )
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    df["total_services"] = df[flag_cols].sum(axis=1)
    df.drop(columns=flag_cols, inplace=True)

    # 4. Security bundle
    df["has_security_bundle"] = (
        (df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes")
    ).astype(int)

    # 5. Tech support
    df["has_support"] = (df["TechSupport"] == "Yes").astype(int)

    # 6. Streaming bundle
    df["has_streaming_bundle"] = (
        (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # 7. New customer flag
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    # 8. High-value customer flag (above $70/month)
    df["is_high_value"] = (df["MonthlyCharges"] > 70).astype(int)

    # 9. Contract risk score (ordinal)
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk"] = df["Contract"].map(contract_map)

    # 10. Average monthly spend
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def create_preprocessor(
    numerical_cols: list,
    categorical_cols: list,
) -> ColumnTransformer:
    """
    Create a sklearn ColumnTransformer pipeline.

    - Numerical features  → median imputation → StandardScaler
    - Categorical features → mode imputation → OneHotEncoder (ignore unknown)

    CRITICAL: Fit ONLY on training data to prevent data leakage.
              The test set is never seen during fitting.

    Parameters
    ----------
    numerical_cols   : list of numerical column names
    categorical_cols : list of categorical column names

    Returns
    -------
    sklearn ColumnTransformer (unfitted)
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline,  numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="passthrough",   # keep any integer binary cols as-is
    )

    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: list,
    categorical_cols: list,
) -> list:
    """
    Extract full feature names after fitting the ColumnTransformer.

    OHE expands each categorical column into multiple binary columns,
    so we need this helper to reconstruct names for SHAP and importance plots.

    Parameters
    ----------
    preprocessor     : fitted ColumnTransformer
    numerical_cols   : original numerical column names
    categorical_cols : original categorical column names

    Returns
    -------
    list of feature name strings
    """
    cat_features = (
        preprocessor
        .named_transformers_["cat"]["encoder"]
        .get_feature_names_out(categorical_cols)
    )
    # remainder columns (passthrough binary flags)
    remainder_cols = [
        preprocessor.feature_names_in_[i]
        for i in preprocessor.transformers_[-1][2]   # indices of remainder
    ] if preprocessor.transformers_[-1][0] == "remainder" else []

    return list(numerical_cols) + list(cat_features) + remainder_cols
