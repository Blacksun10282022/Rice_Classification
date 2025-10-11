
from src.data import load_data
from src.pipeline import build_preprocess, make_pipe
from src.models import get_models
from sklearn.model_selection import train_test_split

def test_smoke_fit_predict():
    X, y, feats = load_data()
    pre = build_preprocess(feats)
    est = get_models()["svm"]
    pipe = make_pipe(pre, est)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    assert preds.shape[0] == yte.shape[0]
