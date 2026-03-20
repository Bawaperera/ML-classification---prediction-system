"""
Data loading and preprocessing for Telco customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV. TotalCharges comes as a string, so we convert it to float."""
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - Fill TotalCharges NaN with 0 (these are new customers with tenure=0)
    - Drop customerID (not a feature)
    - Convert SeniorCitizen 0/1 to No/Yes to match the other binary columns
    - Encode target: Yes=1, No=0
    """
    df = df.copy()

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("Remaining nulls after cleaning:")
        print(null_counts[null_counts > 0])
    else:
        print("No nulls remaining after cleaning.")

    return df


def identify_column_types(df: pd.DataFrame) -> dict:
    """
    Returns a dict categorising all columns into:
    - numerical: tenure, MonthlyCharges, TotalCharges
    - binary: columns with exactly 2 unique values (Yes/No type)
    - categorical: columns with 3+ unique values
    - target: Churn
    """
    target = "Churn"
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical = [c for c in numerical if c != target]

    categorical_all = df.select_dtypes(include=["object"]).columns.tolist()
    binary    = [c for c in categorical_all if df[c].nunique() == 2]
    multi_cat = [c for c in categorical_all if df[c].nunique() > 2]

    return {
        "numerical":   numerical,
        "binary":      binary,
        "categorical": multi_cat,
        "target":      target,
    }


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified 80/20 train/test split.
    Stratifying on Churn keeps the class ratio the same in both sets.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {X_train.shape[0]} rows  |  churn rate: {y_train.mean():.1%}")
    print(f"Test:  {X_test.shape[0]} rows   |  churn rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test
