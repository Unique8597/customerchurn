import mlflow
import os
import numpy as np


folder = os.environ["AZUREML_INPUT_PREPROCESSED"]  # set by aml-job.yaml
print(f"📥 Loading preprocessed data from: {folder}")

X_train = np.load(os.path.join(folder, "X_train.npy"))
y_train = np.load(os.path.join(folder, "y_train.npy"))
X_test  = np.load(os.path.join(folder, "X_test.npy"))
y_test  = np.load(os.path.join(folder, "y_test.npy"))

mlflow.autolog()

with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.5)

