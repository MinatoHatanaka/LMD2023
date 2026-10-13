"""Evaluation metrics.

The paper reports AUC computed with a one-vs-rest LabelBinarizer applied to the test labels
and "the equivalent predicted fractions" (thesis p.109), i.e. hard predictions. We report
that variant (auc_hard) next to the standard probability-based OvR AUC (auc_proba).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

CLASSES = [0, 1, 2]


def scores(model, X):
    """Continuous scores for AUC: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except AttributeError:
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate(y_true, y_pred, y_score=None) -> dict:
    lb = LabelBinarizer().fit(CLASSES)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_hard_macro": roc_auc_score(lb.transform(y_true), lb.transform(y_pred), average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, labels=CLASSES, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=CLASSES, zero_division=0)
    out["per_class"] = {str(c): {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i, c in enumerate(CLASSES)}
    if y_score is not None and np.all(np.isfinite(y_score)):
        # OvR macro AUC computed per class so that decision_function scores also work.
        out["auc_score_macro"] = float(np.mean([roc_auc_score(y_true == c, y_score[:, i]) for i, c in enumerate(CLASSES)]))
    return out
