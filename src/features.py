"""
Feature engineering and preprocessing pipeline for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 10 new features based on patterns found during EDA.

    1.  tenure_group         - bucket tenure into 4 groups
    2.  monthly_tenure_ratio - monthly charge relative to how long they've been a customer
    3.  total_services       - count of active add-on services
    4.  has_security_bundle  - 1 if both OnlineSecurity and OnlineBackup are active
    5.  has_support          - 1 if TechSupport is active
    6.  has_streaming_bundle - 1 if both StreamingTV and StreamingMovies are active
    7.  is_new_customer      - 1 if tenure <= 6 months
    8.  is_high_value        - 1 if MonthlyCharges > 70
    9.  contract_risk        - ordinal risk score (Month-to-month=2, One year=1, Two year=0)
    10. avg_monthly_spend    - TotalCharges divided by tenure
    """
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True,
    ).astype(str)

    df["monthly_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

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

    df["has_security_bundle"] = (
        (df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes")
    ).astype(int)

    df["has_support"] = (df["TechSupport"] == "Yes").astype(int)

    df["has_streaming_bundle"] = (
        (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
    ).astype(int)

    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    df["is_high_value"] = (df["MonthlyCharges"] > 70).astype(int)

    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk"] = df["Contract"].map(contract_map)

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df


def create_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
    - Scales numerical features with StandardScaler
    - One-hot encodes categorical features

    Should be fit on training data only.
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
            ("num", numerical_pipeline,   numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="passthrough",
    )

    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: list,
    categorical_cols: list,
) -> list:
    """
    Returns the full list of feature names after the ColumnTransformer has been fit.
    Needed because OHE expands categorical columns into multiple binary columns.
    """
    cat_features = (
        preprocessor
        .named_transformers_["cat"]["encoder"]
        .get_feature_names_out(categorical_cols)
    )
    remainder_cols = [
        preprocessor.feature_names_in_[i]
        for i in preprocessor.transformers_[-1][2]
    ] if preprocessor.transformers_[-1][0] == "remainder" else []

    return list(numerical_cols) + list(cat_features) + remainder_cols
