
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=0),
        "nb": GaussianNB(),
        "dt": DecisionTreeClassifier(criterion="entropy", random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=5, p=1),
        "svm": SVC(kernel="rbf", probability=True, random_state=0),
        "rf": RandomForestClassifier(random_state=0),
        "gbdt": GradientBoostingClassifier(random_state=0),
        "ada": AdaBoostClassifier(random_state=0),
    }

def get_param_grids():
    return {
        "svm": {"clf__C": [0.5,1,2,5], "clf__gamma": ["scale", 0.5, 1, 2]},
        "rf":  {"clf__n_estimators":[50,100,200], "clf__max_leaf_nodes":[None,12,24],
                "clf__max_features":["sqrt","log2"], "clf__criterion":["gini","entropy"]},
        "logreg": {"clf__C":[0.5,1,2,5]}
    }
