"""H1/H2/H5 verification experiments (see lmd.verify for feature sets and splits).

Results go to results/verify/<dataset>/<experiment>/<featureset>/<model>_fold<k>.json (same
format as run_repro.py plus split metadata); out-of-fold predictions go to
data/verify_preds/... (not versioned) so that pooled metrics can be computed later.
Existing results are skipped.

Baselines (featureset "base"): MAJ-host / MAJ-time predict the training-majority label of
the row's host / time block (time: nearest known block when unseen; host: global majority).

Usage: uv run python src/run_verify.py --dataset authors --experiment E2 --featureset F1 \
           --models NB LSVC DT RF LGBM CB --n-jobs 4
"""

import argparse
import warnings
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmd.data import ROOT  # noqa: E402
from lmd.metrics import CLASSES, evaluate  # noqa: E402
from lmd.models import DEVIATIONS, build  # noqa: E402
from lmd.verify import EXPERIMENTS, dup_keys, feature_columns, fold_name, load, splits  # noqa: E402

BASELINES = ["MAJ-host", "MAJ-time"]
warnings.filterwarnings("ignore", message="Only one class is present")


def aligned_scores(model, X) -> np.ndarray | None:
    """Per-class scores in CLASSES order, also when the training fold lacked a class."""
    out = np.full((len(X), len(CLASSES)), -1e9 if not hasattr(model, "predict_proba") else 0.0)
    classes = list(np.asarray(model.classes_).ravel().astype(int))
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X)
    else:
        s = model.decision_function(X)
        if s.ndim == 1:
            s = np.column_stack([-s, s])
    for j, c in enumerate(classes):
        out[:, CLASSES.index(c)] = s[:, j]
    return out


def majority(keys_tr, y_tr, keys_te, nearest: bool):
    """Lookup-table classifier: per-key label distribution of the training fold."""
    uk, inv = np.unique(keys_tr, return_inverse=True)
    dist = np.zeros((len(uk), len(CLASSES)))
    np.add.at(dist, (inv, y_tr.astype(int)), 1)
    dist /= dist.sum(1, keepdims=True)
    pos = np.searchsorted(uk, keys_te)
    pos_c = np.clip(pos, 0, len(uk) - 1)
    seen = uk[pos_c] == keys_te
    if nearest:  # numeric keys: fall back to the closest known block
        left = np.clip(pos - 1, 0, len(uk) - 1)
        closer_left = np.abs(keys_te - uk[left]) <= np.abs(uk[pos_c] - keys_te)
        pos_c = np.where(seen, pos_c, np.where(closer_left, left, pos_c))
        s = dist[pos_c]
    else:
        prior = np.bincount(y_tr.astype(int), minlength=len(CLASSES)) / len(y_tr)
        s = np.where(seen[:, None], dist[pos_c], prior)
    return s.argmax(1), s


def set_threads(model, n):
    params = model.get_params()
    if "n_jobs" in params:
        model.set_params(n_jobs=n)
    if "thread_count" in params:
        model.set_params(thread_count=n)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(EXPERIMENTS), required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--featureset", required=True, help="F0..F4 or 'base' for baselines")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    assert args.experiment in EXPERIMENTS[args.dataset], args.experiment

    d = load(args.dataset)
    fs_split = "F0" if args.featureset == "base" else args.featureset
    folds = list(splits(d, args.dataset, args.experiment, fs_split))
    keys = dup_keys(d, fs_split)
    rel = Path(args.dataset) / args.experiment / args.featureset
    out_dir, pred_dir = ROOT / "results/verify" / rel, ROOT / "data/verify_preds" / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.featureset != "base":
        cols = feature_columns(args.dataset, d.X, args.featureset)
        X = d.X[cols].to_numpy(np.float32)
    del d.X

    for k, tr, te in folds:
        overlap = float(np.isin(keys[te], np.unique(keys[tr])).mean())
        for name in args.models:
            path = out_dir / f"{name}_fold{k}.json"
            if path.exists():
                continue
            t0 = time.perf_counter()
            if name in BASELINES:
                if name == "MAJ-host":
                    y_pred, s = majority(d.host[tr], d.y[tr], d.host[te], nearest=False)
                else:
                    y_pred, s = majority(d.time_fine[tr], d.y[tr], d.time_fine[te], nearest=True)
                fit_s, pred_s = time.perf_counter() - t0, 0.0
            else:
                model = set_threads(build(name, seed=k), args.n_jobs)
                model.fit(X[tr], d.y[tr])
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                y_pred = np.asarray(model.predict(X[te])).ravel().astype(int)
                s = aligned_scores(model, X[te])
                pred_s = time.perf_counter() - t0
            res = {
                "model": name, "dataset": args.dataset, "experiment": args.experiment,
                "featureset": args.featureset, "fold": k, "fold_name": fold_name(d, args.experiment, k),
                "n_features": None if name in BASELINES else int(X.shape[1]),
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "train_class_counts": np.bincount(d.y[tr], minlength=3).tolist(),
                "test_in_train_rate": overlap,
                "fit_seconds": fit_s, "predict_seconds": pred_s,
                "deviation": DEVIATIONS.get(name.split("-")[0]),
                **evaluate(d.y[te], y_pred, s),
            }
            np.savez_compressed(pred_dir / f"{name}_fold{k}.npz", idx=te.astype(np.int32),
                                y_pred=y_pred.astype(np.int8), score=s.astype(np.float32))
            path.write_text(json.dumps(res, indent=1, default=float))
            print(f"[{rel} fold{k}] {name:8s} F1m={res['f1_macro']:.4f} "
                  f"F1m_pres={res['f1_macro_present']:.4f} AUCs={res.get('auc_score_macro', float('nan')):.4f} "
                  f"binAUC={res.get('binary_auc', float('nan')):.4f} overlap={overlap:.3f} fit={fit_s:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
