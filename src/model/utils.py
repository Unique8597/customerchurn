import mlflow

mlflow.autolog()

with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.5)

