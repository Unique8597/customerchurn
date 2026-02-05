import mlflow
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()  # set by aml-job.yaml
folder = args.input

X_train = np.load(os.path.join(folder, "X_train.npy"))
y_train = np.load(os.path.join(folder, "y_train.npy"))
X_test  = np.load(os.path.join(folder, "X_test.npy"))
y_test  = np.load(os.path.join(folder, "y_test.npy"))

mlflow.autolog()

with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.5)

