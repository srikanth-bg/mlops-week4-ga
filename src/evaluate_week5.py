import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import mlflow

def load_model_from_registry(
    metric_name: str = "accuracy",
    experiment_name: str | None = None
):

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError("Experiment not found")

    runs = mlflow.search_runs(
        [exp.experiment_id],
        filter_string=f"metrics.{metric_name} > 0",
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No runs with metric {metric_name} found")

    run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.pyfunc.load_model(model_uri)

def load_data(path="data.csv"):
    return pd.read_csv(path)

def evaluate(data_path="data.csv", target_col="species", metric_name="accuracy", experiment_name="IRIS classifier experiment: Week5GA"):
    model = load_model_from_registry(
        metric_name=metric_name,
        experiment_name=experiment_name,
    )
    df = load_data(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    y_pred = model.predict(X)
    return {"accuracy": accuracy_score(y, y_pred)}