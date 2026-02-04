"""
train.py

Train GradientBoostingClassifier on preprocessed Customer Churn data.
- Loads preprocessed data from artifacts/
- Trains model with specified hyperparameters
- Evaluates accuracy
- Saves trained model to artifacts/ and logs via MLflow
"""

import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
from utils import load_blob_numpy, load_blob_joblib, blob_service_client


def load_data_from_blob():
    """
    Loads preprocessed datasets and preprocessor directly from Azure Blob Storage.
    """
    print("Downloading training artifacts from Azure Blob...")

    folder = os.environ["AZUREML_INPUT_PREPROCESSED"]

    X_train = np.load(os.path.join(folder, "X_train.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))
    X_test  = np.load(os.path.join(folder, "X_test.npy"))
    y_test  = np.load(os.path.join(folder, "y_test.npy"))


    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, learning_rate=0.1, max_depth=10, n_estimators=150):
    """Train GradientBoostingClassifier with MLflow autolog enabled."""
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model



def save_model(model, artifacts_dir="artifacts"):
    """Save trained model to local artifacts and log to MLflow."""
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")

    # Log model artifact to MLflow
    mlflow.log_artifact(model_path)


def main():
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")

    # Enable MLflow autolog
    mlflow.sklearn.autolog()

    # Start MLflow run (Azure ML automatically links this)
    with mlflow.start_run():
        # Load preprocessed data
        X_train, X_test, y_train, y_test= load_data_from_blob()

        # Train model
        model = train_model(
            X_train,
            y_train,
            learning_rate=0.1,
            max_depth=10,
            n_estimators=150
        )

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        clf_report = classification_report(y_test, y_pred)
        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


        # Log metric
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_text(clf_report, "classification_report.txt")
        mlflow.log_metric("test_roc_auc", roc)


        # Save model
        save_model(model, artifacts_dir)


if __name__ == "__main__":
    main()