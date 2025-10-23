import argparse, json, pathlib
from datetime import datetime, timezone
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True, help="Path to data.csv")
    parser.add_argument("--model",   default="model.joblib", help="Output model path")
    parser.add_argument("--metrics", default="metrics.json", help="Output metrics path")
    # parser.add_argument("--version", required=True, help="version number for the run")
    parser.add_argument("--random_state", type=int, default=123)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--vm_external_ip", required=True, help="External IP of the Vertex AI instance that has MLflow server running")
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"http://{args.vm_external_ip}:8100")
    client = MlflowClient(mlflow.get_tracking_uri())
    all_experiments = client.search_experiments()
    print(f"MLflow experiments: {all_experiments}")
    mlflow.set_experiment("IRIS classifier experiment: Week5GA")

    df = pd.read_csv(args.data)
    train, test = train_test_split(df, test_size = 0.2, stratify = df['species'], random_state = 42)
    X_tr = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_tr = train.species
    X_te = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_te = test.species

    params = {
    "max_depth": args.max_depth,
    "random_state": args.random_state
    }
    clf = DecisionTreeClassifier(**params)
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    f1m = float(f1_score(y_te, preds, average="macro"))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = {
        # "version": args.version,
        "n_samples": int(len(df)),
        "accuracy": acc,
        "f1_macro": f1m,
        "timestamp": now_str,
        "max_depth": args.max_depth,
        "random_state": args.random_state
    }

    # save artifacts
    joblib.dump(clf, args.model)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {args.model}")
    print(f"Saved metrics -> {args.metrics}")
    print(json.dumps(metrics, indent=2))

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy",acc)
        mlflow.set_tag("Training Info", "Decision tree model for IRIS data")
        signature = infer_signature(X_tr, clf.predict(X_tr))
        
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_tr,
            registered_model_name="IRIS_model_Week5GA",
        )

if __name__ == "__main__":
    main()
