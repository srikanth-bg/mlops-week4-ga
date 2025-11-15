import argparse, json, pathlib
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pprint import pprint


def poison_data(X, y, poison_fraction=0.0, noise_std=1.0, random_state=42):
    """
    Simple feature poisoning function.

    X: numpy array of features
    y: numpy array of labels (returned unchanged here)
    poison_fraction: fraction of rows to poison (0.05 = 5 percent)
    noise_std: standard deviation of Gaussian noise to add
    """
    if poison_fraction <= 0:
        return X, y

    rng = np.random.RandomState(random_state)

    X_poisoned = X.copy()
    y_poisoned = y.copy()

    n_samples = int(poison_fraction * len(X_poisoned))
    if n_samples == 0:
        return X_poisoned, y_poisoned

    poisoned_indices = rng.choice(len(X_poisoned), n_samples, replace=False)

    noise = rng.normal(loc=0.0, scale=noise_std, size=X_poisoned[poisoned_indices].shape)
    X_poisoned[poisoned_indices] += noise

    return X_poisoned, y_poisoned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True, help="Path to data.csv")
    parser.add_argument("--model",   default="model.joblib", help="Output model path")
    parser.add_argument("--metrics", default="metrics.json", help="Output metrics path")
    parser.add_argument("--random_state", type=int, default=123)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument(
        "--poison_fraction",
        type=float,
        default=0.0,
        help="Fraction of training samples to poison (for example 0.05, 0.10, 0.50)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="Standard deviation of Gaussian noise used for poisoning",
    )
    parser.add_argument(
        "--vm_external_ip",
        required=True,
        help="External IP of the Vertex AI instance that has MLflow server running",
    )
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(f"http://{args.vm_external_ip}:8100")
    client = MlflowClient(mlflow.get_tracking_uri())
    all_experiments = client.search_experiments()
    print(f"MLflow experiments: {all_experiments}")
    mlflow.set_experiment("IRIS data poisoning experiment: Week8GA")

    # Load data
    df = pd.read_csv(args.data)
    train, test = train_test_split(
        df,
        test_size=0.3,
        stratify=df["species"],
        random_state=42,
    )
    X_tr = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_tr = train["species"]
    X_te = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_te = test["species"]

    # Apply data poisoning on training features
    X_tr_np, y_tr_np = poison_data(
        X_tr.values,
        y_tr.values,
        poison_fraction=args.poison_fraction,
        noise_std=args.noise_std,
        random_state=args.random_state,
    )

    # Convert back to pandas for compatibility with existing code
    X_tr_poisoned = pd.DataFrame(X_tr_np, columns=X_tr.columns)
    y_tr_poisoned = pd.Series(y_tr_np, name=y_tr.name)

    params = {
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "poison_fraction": args.poison_fraction,
        "noise_std": args.noise_std,
    }

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    clf.fit(X_tr_poisoned, y_tr_poisoned)

    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    f1m = float(f1_score(y_te, preds, average="macro"))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = {
        "n_samples": int(len(df)),
        "accuracy": acc,
        "f1_macro": f1m,
        "timestamp": now_str,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "poison_fraction": args.poison_fraction,
        "noise_std": args.noise_std,
    }

    # Save local artifacts
    joblib.dump(clf, args.model)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {args.model}")
    print(f"Saved metrics -> {args.metrics}")
    print(json.dumps(metrics, indent=2))

    # Log to MLflow
    with mlflow.start_run(run_name=f"poison_{args.poison_fraction}"):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1m)
        mlflow.set_tag("Training Info", "Decision tree model for IRIS data with poisoning")
        signature = infer_signature(X_tr_poisoned, clf.predict(X_tr_poisoned))

        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_tr_poisoned,
            registered_model_name="IRIS_model_Week8GA",
        )


if __name__ == "__main__":
    main()
