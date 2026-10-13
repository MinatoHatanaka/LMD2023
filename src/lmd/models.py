"""Shallow classifiers configured as in IJIS 2023, Table 10 and Sect. 5.1.1.

Where the paper is ambiguous or uses values that the current scikit-learn rejects, the
choice made here is recorded in DEVIATIONS so that it can be reported alongside results.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

DEVIATIONS = {
    "LR": "elasticnet l1_ratio not reported; 0.5 used. multi_class='multinomial' is the default for saga in sklearn>=1.5.",
    "SGDC": "Table 10 gives max_iter=5e-3 (not a valid iteration count); sklearn default max_iter=1000 used.",
    "DT": "max_features='sqrt' taken from the text (Sect. 5.1.1); Table 10 does not list it.",
    "CB": "'CrossEntropy' is binary-only in CatBoost; loss_function='MultiClass' used for 3 classes.",
    "Bagging": "Base estimators keep their own Table 10 settings; Bagging(CB) n_jobs=1 to bound memory.",
}


def _dt(seed):
    return DecisionTreeClassifier(
        criterion="entropy", max_depth=20, max_leaf_nodes=100, min_samples_leaf=8,
        ccp_alpha=1e-3, max_features="sqrt", random_state=seed,
    )


def _rf(seed):
    return RandomForestClassifier(
        n_estimators=300, criterion="gini", max_depth=20, max_leaf_nodes=100,
        min_samples_leaf=8, min_samples_split=20, ccp_alpha=1e-3, n_jobs=-1, random_state=seed,
    )


def _et(seed):
    return ExtraTreesClassifier(
        n_estimators=1000, criterion="gini", max_depth=300, max_leaf_nodes=650,
        min_samples_leaf=8, min_samples_split=10, ccp_alpha=1e-2, n_jobs=-1, random_state=seed,
    )


def _cb(seed):
    return CatBoostClassifier(
        iterations=70, learning_rate=0.3, loss_function="MultiClass", l2_leaf_reg=1e-7,
        random_seed=seed, verbose=False, thread_count=-1,
    )


def _bagging(base, seed):
    return BaggingClassifier(estimator=base, n_estimators=12, max_samples=0.8, random_state=22)


def build(name: str, seed: int = 0):
    factories = {
        "LR": lambda: LogisticRegression(
            solver="saga", l1_ratio=0.5, max_iter=1000, tol=1e-3, random_state=seed,
        ),
        "SGDC": lambda: SGDClassifier(
            loss="modified_huber", penalty="elasticnet", alpha=1e-3, tol=1e-8,
            learning_rate="optimal", early_stopping=True, random_state=seed,
        ),
        "KNN": lambda: KNeighborsClassifier(
            n_neighbors=10, leaf_size=50, algorithm="auto", weights="uniform",
            metric="minkowski", n_jobs=-1,
        ),
        "NB": lambda: ComplementNB(),
        "LSVC": lambda: LinearSVC(C=1.5, max_iter=30000, penalty="l2", loss="squared_hinge", random_state=seed),
        "LGBM": lambda: LGBMClassifier(
            boosting_type="gbdt", max_bin=20, max_depth=20, num_leaves=20, min_child_samples=30,
            min_data_in_bin=10, min_split_gain=0.1, n_estimators=100, learning_rate=0.1,
            reg_alpha=1e-2, reg_lambda=1e-2, n_jobs=8, random_state=30, verbose=-1,
        ),
        "DT": lambda: _dt(seed),
        "RF": lambda: _rf(seed),
        "ET": lambda: _et(seed),
        "CB": lambda: _cb(seed),
        "Bagging-DT": lambda: _bagging(_dt(seed), seed),
        "Bagging-RF": lambda: _bagging(_rf(seed), seed),
        "Bagging-ET": lambda: _bagging(_et(seed), seed),
        "Bagging-CB": lambda: _bagging(_cb(seed), seed),
        # Sensitivity variants (not in the paper): Table 10 settings without cost-complexity pruning.
        "ET-noccp": lambda: _et(seed).set_params(ccp_alpha=0.0),
        "DT-noccp": lambda: _dt(seed).set_params(ccp_alpha=0.0),
    }
    return factories[name]()


ALL_MODELS = [
    "LR", "SGDC", "KNN", "NB", "LSVC", "LGBM", "DT", "RF", "ET", "CB",
    "Bagging-DT", "Bagging-RF", "Bagging-ET", "Bagging-CB",
]
