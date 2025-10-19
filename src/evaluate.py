import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def load_model(path="model.joblib"):
    return joblib.load(path)

def load_data(path="data.csv"):
    return pd.read_csv(path)

def evaluate(model_path="model.joblib", data_path="data.csv", target_col="species"):
    model = load_model(model_path)
    df = load_data(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    y_pred = model.predict(X)
    return {"accuracy": accuracy_score(y, y_pred)}
