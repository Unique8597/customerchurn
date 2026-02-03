"""
train.py

Train GradientBoostingClassifier on preprocessed Customer Churn data.
- Loads preprocessed data from artifacts/
- Trains model with specified hyperparameters
- Evaluates accuracy
- Saves trained model to artifacts/
"""

import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(artifacts_dir="artifacts"):
    """Load preprocessed train/test data and preprocessor."""
    X_train = np.load(os.path.join(artifacts_dir, "X_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(artifacts_dir, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"), allow_pickle=True)
    
    preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
    
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(X_train, y_train, learning_rate=0.1, max_depth=10, n_estimators=150):
    """Train GradientBoostingClassifier."""
    clf = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    return acc


def save_model(model, artifacts_dir="artifacts"):
    """Save trained model."""
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")


def main():
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, preprocessor = load_data(artifacts_dir)
    
    # Train model
    model = train_model(X_train, y_train, learning_rate=0.1, max_depth=10, n_estimators=150)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, artifacts_dir)


if __name__ == "__main__":
    main()