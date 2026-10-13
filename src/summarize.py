"""Collect results/repro/**.json into a comparison table against IJIS 2023 Tables 11 and 13.

Writes results/repro_summary.csv and results/repro_summary.md.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# IJIS 2023 Table 11 (shallow) and Table 13 (DNN): AUC, Prec, Recall, F1, Acc (in %).
PAPER = {
    "LR": (93.51, 98.66, 98.72, 98.68, 98.72), "SGDC": (93.27, 92.49, 90.03, 91.21, 98.47),
    "KNN": (98.33, 96.77, 97.57, 97.16, 99.47), "NB": (98.35, 98.21, 95.85, 96.63, 95.85),
    "LSVC": (92.89, 95.16, 89.45, 92.01, 98.67), "LGBM": (99.61, 99.38, 99.36, 99.37, 99.90),
    "DT": (99.53, 98.70, 99.33, 99.01, 99.82), "RF": (99.68, 98.61, 99.56, 99.08, 99.83),
    "ET": (99.84, 99.05, 99.79, 99.41, 99.89), "CB": (98.09, 98.32, 97.17, 97.73, 99.59),
    "Bagging-DT": (99.80, 99.04, 99.73, 99.38, 99.88), "Bagging-RF": (99.37, 98.50, 99.10, 98.78, 99.77),
    "Bagging-ET": (99.72, 98.88, 99.61, 99.24, 99.86), "Bagging-CB": (97.99, 98.28, 97.04, 97.64, 99.57),
    "DNN-MLP": (95.57, 95.05, 92.22, 93.57, 98.87), "DNN-CNN": (95.12, 95.44, 92.68, 94.43, 98.66),
    "DNN-LSTM": (95.82, 95.11, 94.36, 95.55, 98.93), "DNN-RNN": (95.26, 95.64, 93.12, 94.87, 98.83),
    "DNN-AE": (95.10, 95.17, 92.14, 94.55, 98.24),
}
# Sensitivity variants (not in the paper) are listed after the paper's models.
ORDER = list(PAPER) + ["ET-noccp", "DT-noccp"]


def main() -> None:
    rows = []
    for p in sorted((ROOT / "results/repro").glob("*/*/*_fold*.json")):
        r = json.loads(p.read_text())
        if "_ep" in p.stem:  # smoke-test runs with overridden epochs
            continue
        rows.append({
            "dataset": r["dataset"], "protocol": r["protocol"], "model": r["model"], "fold": r["fold"],
            "auc_hard": r["auc_hard_macro"], "auc_score": r.get("auc_score_macro"),
            "f1_weighted": r["f1_weighted"], "f1_macro": r["f1_macro"],
            "prec_weighted": r["precision_weighted"], "rec_weighted": r["recall_weighted"],
            "prec_macro": r["precision_macro"], "rec_macro": r["recall_macro"],
            "acc": r["accuracy"], "f1_EoHT": r["per_class"]["2"]["f1"],
            "test_in_train": r.get("test_in_train_rate"), "fit_s": r["fit_seconds"],
        })
    df = pd.DataFrame(rows)
    agg = df.groupby(["dataset", "protocol", "model"]).mean(numeric_only=True).drop(columns="fold")
    agg["n_folds"] = df.groupby(["dataset", "protocol", "model"]).size()
    agg = agg.reset_index()
    for i, k in enumerate(["paper_auc", "paper_prec", "paper_rec", "paper_f1", "paper_acc"]):
        agg[k] = agg["model"].map(lambda m: PAPER.get(m, (None,) * 5)[i])
    agg["model"] = pd.Categorical(agg["model"], ORDER, ordered=True)
    agg = agg.sort_values(["dataset", "protocol", "model"])
    agg.to_csv(ROOT / "results/repro_summary.csv", index=False)

    pct = ["auc_hard", "auc_score", "f1_weighted", "f1_macro", "f1_EoHT", "acc", "test_in_train",
           "prec_weighted", "rec_weighted", "prec_macro", "rec_macro"]
    show = agg.copy()
    show[pct] = (show[pct] * 100).round(2)
    cols = ["dataset", "protocol", "model", "n_folds", "paper_auc", "auc_hard", "auc_score",
            "paper_f1", "f1_weighted", "f1_macro", "f1_EoHT", "paper_acc", "acc", "test_in_train", "fit_s"]
    md = show[cols].to_markdown(index=False, floatfmt=".2f")
    (ROOT / "results/repro_summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
