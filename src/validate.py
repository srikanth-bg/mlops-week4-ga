import pandas as pd

EXPECTED_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

def basic_schema_checks(df: pd.DataFrame):
    # 1) exact column order
    assert list(df.columns) == EXPECTED_COLS
    # 2) no nulls
    assert df.isna().sum().sum() == 0
    # 3) rough value ranges
    assert df["sepal_length"].between(4.0, 8.5).all()
    assert df["sepal_width"].between(1.0, 4.5).all()
    assert df["petal_length"].between(0.5, 7.5).all()
    assert df["petal_width"].between(0.0, 3.0).all()
    # 4) target labels (either strings or encoded ints)
    uniq = set(df["species"].unique())
    assert uniq.issubset({"setosa","versicolor","virginica"})
