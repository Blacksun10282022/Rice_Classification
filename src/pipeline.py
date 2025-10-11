
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def build_preprocess(numeric_features):
    return ColumnTransformer(
        transformers=[("num", make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler()), numeric_features)],
        remainder="drop"
    )

def make_pipe(preprocess, estimator):
    return Pipeline([("pre", preprocess), ("clf", estimator)])
