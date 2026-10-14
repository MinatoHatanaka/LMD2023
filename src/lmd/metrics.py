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
    """Metrics for one test set.

    Classes absent from y_true (e.g. a held-out host with a single label) get AUC = NaN and
    are left out of the macro AUCs; f1_macro_present averages F1 over the classes present.
    The attack-vs-normal (0 vs {1,2}) metrics use -score[:, 0] as the attack score.
    """
    lb = LabelBinarizer().fit(CLASSES)
    present = [c for c in CLASSES if np.any(y_true == c)]
    yt_bin, yp_bin = lb.transform(y_true), lb.transform(y_pred)
    auc_hard = [roc_auc_score(yt_bin[:, i], yp_bin[:, i]) for i, c in enumerate(CLASSES) if c in present]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_hard_macro": float(np.mean(auc_hard)) if len(present) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "classes_present": present,
        "f1_macro_present": precision_recall_fscore_support(
            y_true, y_pred, average="macro", labels=present, zero_division=0)[2],
    }
    attack, attack_pred = y_true != 0, y_pred != 0
    out["binary_f1"] = precision_recall_fscore_support(attack, attack_pred, average="binary", zero_division=0)[2]
    out["binary_recall"] = float(attack_pred[attack].mean()) if attack.any() else float("nan")
    out["binary_fpr"] = float(attack_pred[~attack].mean()) if (~attack).any() else float("nan")
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, labels=CLASSES, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=CLASSES, zero_division=0)
    out["per_class"] = {str(c): {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i, c in enumerate(CLASSES)}
    if y_score is not None and np.all(np.isfinite(y_score)):
        # OvR macro AUC computed per class so that decision_function scores also work.
        auc = {str(c): (roc_auc_score(y_true == c, y_score[:, i]) if c in present and len(present) > 1
                        else float("nan")) for i, c in enumerate(CLASSES)}
        out["auc_score_per_class"] = auc
        out["auc_score_macro"] = (float(np.mean([auc[str(c)] for c in present]))
                                  if len(present) > 1 else float("nan"))
        both = attack.any() and (~attack).any()
        out["binary_auc"] = roc_auc_score(attack, -y_score[:, 0]) if both else float("nan")
    return out
