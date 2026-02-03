"""
inference.py

Customer Churn Prediction Inference Script
- Loads trained GradientBoostingClassifier model
- Loads preprocessing pipeline
- Accepts new customer data (CSV or JSON)
- Outputs predictions with probability
"""

import os
import pandas as pd
import joblib
import argparse


def load_artifacts(artifacts_dir="artifacts"):
    """Load preprocessor and trained model."""
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    model_path = os.path.join(artifacts_dir, "model.joblib")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    print(f"Loaded preprocessor from {preprocessor_path}")
    print(f"Loaded model from {model_path}")

    return preprocessor, model


def load_new_data(input_path):
    """Load new data for prediction (CSV or JSON)."""
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path)
    else:
        raise ValueError("Input file must be CSV or JSON")
    return df


def predict(df, preprocessor, model):
    """Run preprocessing and predict churn probabilities."""
    X_processed = preprocessor.transform(df)
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)[:, 1]  # probability of churn=1

    result = df.copy()
    result["Churn_Prediction"] = y_pred
    result["Churn_Probability"] = y_proba

    return result


def save_predictions(df, output_path="predictions.csv"):
    """Save predictions to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Inference")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Path to artifacts folder")
    args = parser.parse_args()

    preprocessor, model = load_artifacts(args.artifacts)
    df_new = load_new_data(args.input)
    result = predict(df_new, preprocessor, model)


if __name__ == "__main__":
    main()