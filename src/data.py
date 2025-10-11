
from pathlib import Path
import pandas as pd

def find_data_file():
    for p in [Path("data/rice-final2.csv"), Path("./rice-final2.csv")]:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find data/rice-final2.csv")

def load_data():
    path = find_data_file()
    df = pd.read_csv(path)
    if "class" not in df.columns:
        raise ValueError("Expected 'class' column.")
    y = df["class"].map({"class1":0, "class2":1}).astype(int).values
    X = df.drop(columns=["class"]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y, list(X.columns)
