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
import joblib
import mlflow
import mlflow.sklearn
import argparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    average_precision_score
)



def load_data_from_input(folder):
    """
    Loads preprocessed datasets from Azure ML input folder.
    AzureML sets env var: AZUREML_INPUT_<input_name>
    Example: AZUREML_INPUT_PREPROCESSED
    """
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


def save_model(model, output_dir):
    """Save trained model locally and log via MLflow."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")

    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    mlflow.log_artifact(model_path)  # upload to AML run artifacts


def main():
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()  # set by aml-job.yaml
    folder = args.input
    output_dir = args.output

    # AzureML automatically sets tracking URI + experiment
    with mlflow.start_run():
        print("🚀 MLflow run started")

        # Load dataset
        X_train, X_test, y_train, y_test = load_data_from_input(folder)

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
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        ap_score = average_precision_score(y_test, y_proba)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("balanced_accuracy", balanced_acc)
        mlflow.log_metric("avg_precision_score", ap_score)

        # Confusion matrix metrics
        mlflow.log_metric("true_positive", tp)
        mlflow.log_metric("true_negative", tn)
        mlflow.log_metric("false_positive", fp)
        mlflow.log_metric("false_negative", fn)

        #mlflow.log_text(clf_report, "classification_report.txt")

        print("📊 Metrics logged to MLflow")

        # Save model
        save_model(model, output_dir)
        print("🎉 Training pipeline finished successfully.")


if __name__ == "__main__":
    main()