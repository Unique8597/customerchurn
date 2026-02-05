"""
train.py

Train GradientBoostingClassifier on preprocessed Customer Churn data.
- Loads preprocessed data from AzureML job input
- Trains model with MLflow autolog
- Logs metrics and artifacts
"""

import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn


def load_data_from_input():
    """
    Loads preprocessed datasets from Azure ML input folder.
    AzureML sets env var: AZUREML_INPUT_<input_name>
    Example: AZUREML_INPUT_PREPROCESSED
    """

    folder = os.environ["AZUREML_INPUT_PREPROCESSED"]  # set by aml-job.yaml
    print(f"📥 Loading preprocessed data from: {folder}")

    X_train = np.load(os.path.join(folder, "X_train.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))
    X_test  = np.load(os.path.join(folder, "X_test.npy"))
    y_test  = np.load(os.path.join(folder, "y_test.npy"))

    print("✔ Preprocessed data loaded successfully.")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, learning_rate=0.1, max_depth=10, n_estimators=150):
    """Train GradientBoostingClassifier."""
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, artifacts_dir="artifacts"):
    """Save trained model locally and log via MLflow."""
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.joblib")

    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    mlflow.log_artifact(model_path)  # upload to AML run artifacts


def main():
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")

    # AzureML automatically sets tracking URI + experiment
    with mlflow.start_run():
        print("🚀 MLflow run started")

        # Load dataset
        X_train, X_test, y_train, y_test = load_data_from_input()

        # Train
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
        #clf_report = classification_report(y_test, y_pred)

        # Some churn datasets may be binary → safe roc computation
        try:
            roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except:
            roc = None

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        if roc is not None:
            mlflow.log_metric("roc_auc", roc)

        #mlflow.log_text(clf_report, "classification_report.txt")

        print("📊 Metrics logged to MLflow")

        # Save model
        save_model(model, artifacts_dir)
        print("🎉 Training pipeline finished successfully.")


if __name__ == "__main__":
    main()