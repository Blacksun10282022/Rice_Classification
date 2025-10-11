
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from .data import load_data
from .pipeline import build_preprocess, make_pipe
from .models import get_models, get_param_grids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["svm","rf","logreg","nb","dt","knn","gbdt","ada"], default="svm")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    X, y, feats = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    pipe = make_pipe(build_preprocess(feats), get_models()[args.model])

    grids = get_param_grids()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    if args.model in grids:
        gs = GridSearchCV(pipe, param_grid=grids[args.model], scoring="accuracy", cv=cv, refit=True)
        gs.fit(Xtr, ytr)
        model = gs.best_estimator_
        best_params = gs.best_params_
        cv_best = float(gs.best_score_)
    else:
        scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy")
        model = pipe.fit(Xtr, ytr)
        best_params, cv_best = {}, float(scores.mean())

    ypred = model.predict(Xte)
    if hasattr(model, "predict_proba"):
        yscore = model.predict_proba(Xte)[:,1]
    else:
        yscore = model.decision_function(Xte)

    metrics = {
        "cv_best_accuracy": cv_best,
        "test_accuracy": float(accuracy_score(yte, ypred)),
        "test_f1_macro": float(f1_score(yte, ypred, average="macro")),
        "test_f1_weighted": float(f1_score(yte, ypred, average="weighted")),
        "roc_auc": float(roc_auc_score(yte, yscore)),
        "pr_auc": float(average_precision_score(yte, yscore)),
        "best_params": best_params,
        "features": feats,
        "model": args.model,
    }

    import joblib
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"best_model_{args.model}.joblib"
    joblib.dump(model, model_path)
    (outdir / f"metrics_{args.model}.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model to", model_path)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
