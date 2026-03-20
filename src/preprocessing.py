"""
ML Classification & Prediction System
Telco Customer Churn — Data Preprocessing Pipeline

Handles: loading, cleaning, type fixing, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Telco churn CSV.

    IMPORTANT: TotalCharges has hidden spaces that make it a string.
    We convert it to numeric and let pandas set spaces to NaN —
    these ~11 rows are new customers with 0 tenure (no charges yet).

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with TotalCharges cast to float.
    """
    df = pd.read_csv(filepath)
    # Replace whitespace-only strings with NaN, then cast to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:

    1. Fill TotalCharges NaN with 0  (new customers, tenure == 0, no charges)
    2. Drop customerID            (not a feature — it's a key)
    3. Convert SeniorCitizen 0/1 → 'No'/'Yes' for consistency with other binary cols
    4. Encode target 'Churn'  Yes → 1, No → 0
    5. Verify no remaining nulls

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = df.copy()

    # 1. Fill TotalCharges NaN with 0 (tenure=0 new customers)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 2. Drop customerID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # 3. SeniorCitizen: 0/1 → 'No'/'Yes'
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 4. Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 5. Sanity check
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"⚠️  Remaining nulls:\n{null_counts[null_counts > 0]}")
    else:
        print("✅  No remaining nulls after cleaning.")

    return df


def identify_column_types(df: pd.DataFrame) -> dict:
    """
    Categorise every column in the cleaned dataframe.

    Returns
    -------
    dict with keys:
        'numerical'  : ['tenure', 'MonthlyCharges', 'TotalCharges']
        'binary'     : columns with exactly 2 unique values (Yes/No pattern)
        'categorical': columns with 3+ unique values (InternetService, Contract, etc.)
        'target'     : 'Churn'
    """
    target = "Churn"
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical = [c for c in numerical if c != target]

    categorical_all = df.select_dtypes(include=["object"]).columns.tolist()
    binary = [c for c in categorical_all if df[c].nunique() == 2]
    multi_cat = [c for c in categorical_all if df[c].nunique() > 2]

    return {
        "numerical": numerical,
        "binary": binary,
        "categorical": multi_cat,
        "target": target,
    }


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified train/test split.

    Stratifying on Churn maintains the ~73/27 class ratio in both splits,
    which is critical for fair evaluation on an imbalanced dataset.

    Parameters
    ----------
    df           : cleaned dataframe (output of clean_data)
    test_size    : fraction for test set (default 0.20)
    random_state : seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train set: {X_train.shape[0]} rows | "
          f"Churn rate: {y_train.mean():.1%}")
    print(f"Test  set: {X_test.shape[0]} rows  | "
          f"Churn rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test
