import argparse, json, pathlib
from datetime import datetime, timezone
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True, help="Path to data.csv")
    parser.add_argument("--model",   default="model.joblib", help="Output model path")
    parser.add_argument("--metrics", default="metrics.json", help="Output metrics path")
    parser.add_argument("--version", required=True, help="version number for the run")
    parser.add_argument("--random_state", type=int, default=123)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train, test = train_test_split(df, test_size = 0.2, stratify = df['species'], random_state = 42)
    X_tr = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_tr = train.species
    X_te = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_te = test.species

    clf = DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.random_state)
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    f1m = float(f1_score(y_te, preds, average="macro"))

    metrics = {
        "version": args.version,
        "n_samples": int(len(df)),
        "accuracy": acc,
        "f1_macro": f1m
    }

    # save artifacts
    joblib.dump(clf, args.model)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {args.model}")
    print(f"Saved metrics -> {args.metrics}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
