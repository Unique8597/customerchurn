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
from io import BytesIO
import io
from urllib.parse import urlparse
import os
import pandas as pd
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from storage_utils import get_blob_service_client

container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
blob_name = os.getenv("AZURE_STORAGE_BLOB_NAME")
prefix = os.getenv("ARTIFACTS_BLOB_PREFIX", "preprocess-artifacts")

blob_service_client = get_blob_service_client()


def download_blob_to_memory():
    """
    Downloads a blob from Azure Blob Storage into memory as a pandas DataFrame.
    """

    # Create a BlobClient using the BlobServiceClient
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download blob into memory
    stream = BytesIO()
    data = blob_client.download_blob()
    data.readinto(stream)
    stream.seek(0)

    # Read CSV into pandas
    df = pd.read_csv(stream)
    return df


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
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def preprocess_and_split(df: pd.DataFrame, target_columns: list):
    """Prepare dataset and return train/test splits and preprocessor."""

    y = df[target_columns[0]]
    X = df.drop(columns=target_columns)
    preprocessor = build_preprocessor(df, target_columns)

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test, preprocessor

def upload_bytes_to_blob(blob_client, data_bytes: bytes):
    blob_client.upload_blob(data_bytes, overwrite=True)


# ------------------------------- #
#   SAVE ARTIFACTS TO BLOB
# ------------------------------- #

def save_artifacts(X_train, X_test, y_train, y_test, preprocessor):

    container_client = blob_service_client.get_container_client(container_name)

    # ---- Save preprocessor ---- #
    preproc_buffer = BytesIO()
    joblib.dump(preprocessor, preproc_buffer)
    upload_bytes_to_blob(
        container_client.get_blob_client(f"{prefix}/preprocessor.joblib"),
        preproc_buffer.getvalue()
    )


    output_dir = os.getenv("AZUREML_OUTPUT_DIR", "src/preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    print("✔ Artifacts successfully uploaded to Blob Storage.")


def main():

    TARGET = ["Churn","CustomerID"]  # Change if your target column differs

    df = download_blob_to_memory()
    df = clean_data(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df, TARGET)

    save_artifacts(X_train, X_test, y_train, y_test, preprocessor)


if __name__ == "__main__":
    main()