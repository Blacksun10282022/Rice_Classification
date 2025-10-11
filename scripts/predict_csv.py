
import argparse
import pandas as pd
from pathlib import Path
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/best_model_svm.joblib")
    ap.add_argument("--input", default="data/rice-final2.csv")
    ap.add_argument("--output", default="preds.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)
    X = df.drop(columns=["class"]).copy() if "class" in df.columns else df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    out = df.copy()
    out["pred"] = model.predict(X)
    out.to_csv(args.output, index=False)
    print("Wrote", args.output)

if __name__ == "__main__":
    main()
