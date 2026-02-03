"""
preprocess.py

Preprocessing script for Customer Churn Prediction.
- Loads data from Azure Blob or local CSV
- Cleans missing values
- Encodes categorical variables
- Scales numeric features
- Creates train/test sets
- Saves preprocessing artifacts for inference

Author: EmadFS
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from storage_utils import upload_if_not_exists, get_blob_uri

container = "customer-data"
local_csv = "data/customer_churn.csv"
blob_name = "customer_churn.csv"

# Upload only if not present
upload_if_not_exists(container, local_csv, blob_name)

# Get the URI for Azure ML job
data_uri = get_blob_uri(container, blob_name)
print(f"Data available at: {data_uri}")


def load_data(data_path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"Loading dataset from: {data_path}")
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle missing values."""
    df = df.copy()

    # Example domain-specific cleanup
    # Fill numeric missing values with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df


def build_preprocessor(df: pd.DataFrame, target_columns: list[str]):
    """Builds a sklearn ColumnTransformer preprocessing pipeline."""

    X = df.drop(columns=target_columns)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", LabelEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def preprocess_and_split(df: pd.DataFrame, target_column: list[str]):
    """Prepare dataset and return train/test splits and preprocessor."""

    y = df[target_column]
    X = df.drop(columns=target_column)
    preprocessor = build_preprocessor(df, target_column)

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test, preprocessor


def save_artifacts(X_train, X_test, y_train, y_test, preprocessor, output_dir="artifacts"):
    """Save processed datasets and preprocessing pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print("Artifacts saved successfully.")


def main():
    RAW_DATA_PATH = os.getenv("DATA_PATH", "data/customer_churn.csv")
    TARGET = ["Churn","CustomerID"]  # Change if your target column differs

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df, TARGET)

    save_artifacts(X_train, X_test, y_train, y_test, preprocessor)


if __name__ == "__main__":
    main()