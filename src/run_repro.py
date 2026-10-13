"""Reproduce the shallow-classifier experiments of IJIS 2023 (Table 11).

Protocols
  holdout : one stratified 75/25 split (random_state=42). The paper's "fold" sizes
            (1,314,668 / 438,223) equal a default train_test_split, which this mimics.
  cv10    : stratified 10-fold CV with shuffling, as described in the paper's text.

Datasets: see lmd.data.DATASETS. Results are written per (model, fold) to
results/repro/<dataset>/<protocol>/ and skipped if present.

Usage: uv run python src/run_repro.py --dataset authors --protocol holdout --models DT RF ET
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmd.data import DATASETS, ROOT, load_matrix, test_in_train_rate  # noqa: E402
from lmd.metrics import evaluate, scores  # noqa: E402
from lmd.models import ALL_MODELS, DEVIATIONS, build  # noqa: E402


def splits(protocol: str, y: np.ndarray):
    idx = np.arange(len(y))
    if protocol == "holdout":
        tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=42)
        yield 0, tr, te
    elif protocol == "cv10":
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for k, (tr, te) in enumerate(skf.split(idx, y)):
            yield k, tr, te
    else:
        raise ValueError(protocol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="authors")
    ap.add_argument("--protocol", choices=["holdout", "cv10"], default="holdout")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--folds", type=int, nargs="*", help="subset of fold indices to run")
    args = ap.parse_args()

    X_df, y = load_matrix(args.dataset)
    X64 = X_df.to_numpy(np.float64)
    del X_df
    folds = [(k, tr, te) for k, tr, te in splits(args.protocol, y) if not args.folds or k in args.folds]
    # Exact-duplicate overlap is measured on float64 values, then the float64 copy is released.
    overlaps = {k: test_in_train_rate(X64[tr], X64[te]) for k, tr, te in folds}
    X = X64.astype(np.float32)
    del X64
    out_dir = ROOT / "results/repro" / args.dataset / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, tr, te in folds:
        overlap = overlaps[k]
        for name in args.models:
            path = out_dir / f"{name}_fold{k}.json"
            if path.exists():
                continue
            model = build(name, seed=k)
            t0 = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            y_pred = np.asarray(model.predict(X[te])).ravel().astype(int)
            s = scores(model, X[te])
            pred_s = time.perf_counter() - t0
            res = {
                "model": name, "dataset": args.dataset, "protocol": args.protocol, "fold": k,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "test_in_train_rate": overlap,
                "fit_seconds": fit_s, "predict_seconds": pred_s,
                "deviation": DEVIATIONS.get(name.split("-")[0]),
                **evaluate(y[te], y_pred, s),
            }
            path.write_text(json.dumps(res, indent=1))
            print(f"[{args.dataset} {args.protocol} fold{k}] {name:11s} F1w={res['f1_weighted']:.4f} "
                  f"F1m={res['f1_macro']:.4f} AUChard={res['auc_hard_macro']:.4f} "
                  f"AUCscore={res.get('auc_score_macro', float('nan')):.4f} fit={fit_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
