"""Aggregate results/verify/** into results/verify_summary.{csv,md}.

For every (dataset, experiment, featureset, model) two views are reported:
  pooled : metrics on the concatenated out-of-fold predictions (data/verify_preds); every row
           is predicted by a model that did not train on its fold. This is the main view,
           since E2/E3 folds often contain a single class.
  fold   : mean (std) over folds of per-fold macro F1 over the classes present.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmd.data import ROOT, load_matrix  # noqa: E402
from lmd.metrics import evaluate  # noqa: E402

RES, PRED = ROOT / "results/verify", ROOT / "data/verify_preds"
MODEL_ORDER = ["MAJ-host", "MAJ-time", "NB", "LSVC", "DT", "RF", "LGBM", "CB", "ET"]


def main() -> None:
    runs = defaultdict(list)
    for p in sorted(RES.glob("*/*/*/*.json")):
        r = json.loads(p.read_text())
        runs[(r["dataset"], r["experiment"], r["featureset"], r["model"])].append(r)
    labels = {ds: load_matrix(ds)[1] for ds in {k[0] for k in runs}}
    rows = []
    for (ds, exp, fs, model), folds in runs.items():
        y = labels[ds]
        parts = [np.load(PRED / ds / exp / fs / f"{model}_fold{r['fold']}.npz") for r in folds]
        idx = np.concatenate([p["idx"] for p in parts])
        pred = np.concatenate([p["y_pred"] for p in parts]).astype(int)
        score = np.concatenate([p["score"] for p in parts]).astype(np.float64)
        m = evaluate(y[idx], pred, score)
        f1p = [r["f1_macro_present"] for r in folds]
        rows.append({
            "dataset": ds, "experiment": exp, "featureset": fs, "model": model,
            "n_folds": len(folds), "n_eval": len(idx),
            "f1_macro": m["f1_macro"], "auc_score_macro": m.get("auc_score_macro", np.nan),
            "binary_auc": m.get("binary_auc", np.nan), "binary_f1": m["binary_f1"],
            **{f"f1_c{c}": m["per_class"][str(c)]["f1"] for c in range(3)},
            "f1_weighted": m["f1_weighted"], "accuracy": m["accuracy"],
            "auc_hard_macro": m["auc_hard_macro"],
            "fold_f1m_present_mean": np.mean(f1p), "fold_f1m_present_std": np.std(f1p),
            "test_in_train_rate": np.mean([r["test_in_train_rate"] for r in folds]),
            "fit_seconds": np.sum([r["fit_seconds"] for r in folds]),
        })
    df = pd.DataFrame(rows)
    df["_m"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values(["dataset", "experiment", "featureset", "_m"]).drop(columns="_m")
    df.to_csv(ROOT / "results/verify_summary.csv", index=False)

    pct = [c for c in df if c.startswith(("f1", "auc", "binary", "accuracy", "test_in"))]
    md = df.copy()
    md[pct] = (md[pct] * 100).round(2)
    md["fit_seconds"] = md["fit_seconds"].round(0)
    (ROOT / "results/verify_summary.md").write_text(md.to_markdown(index=False) + "\n")
    print(f"{len(df)} configurations summarised")


if __name__ == "__main__":
    main()
