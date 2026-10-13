"""Reproduce the DNN experiments of IJIS 2023 (Table 13) on a stratified 75/25 holdout.

Usage: uv run python src/run_dnn.py --dataset authors --models MLP CNN [--epochs N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmd.data import DATASETS, ROOT, load_matrix  # noqa: E402
from lmd.dnn import DEVIATIONS, SPECS, build  # noqa: E402
from lmd.metrics import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="authors")
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    ap.add_argument("--epochs", type=int, help="override Table 12 epochs (e.g. for a time budget)")
    args = ap.parse_args()

    import keras
    keras.utils.set_random_seed(42)

    X_df, y = load_matrix(args.dataset)
    X = X_df.to_numpy(np.float32)
    del X_df
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=42)
    out_dir = ROOT / "results/repro" / args.dataset / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        tag = f"_ep{args.epochs}" if args.epochs else ""
        path = out_dir / f"DNN-{name}{tag}_fold0.json"
        if path.exists():
            continue
        *_, epochs, batch, standardize = SPECS[name]
        epochs = args.epochs or epochs
        Xtr, Xte = X[tr], X[te]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
        model = build(name, X.shape[1])
        sparse = SPECS[name][4].startswith("sparse")
        ytr = y[tr] if sparse else keras.utils.to_categorical(y[tr], 3)
        t0 = time.perf_counter()
        hist = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch, verbose=2)
        fit_s = time.perf_counter() - t0
        proba = model.predict(Xte, batch_size=4096, verbose=0)
        res = {
            "model": f"DNN-{name}", "dataset": args.dataset, "protocol": "holdout", "fold": 0,
            "n_train": int(len(tr)), "n_test": int(len(te)), "epochs": epochs, "batch_size": batch,
            "fit_seconds": fit_s, "train_loss": hist.history["loss"],
            "deviation": " | ".join(v for k, v in DEVIATIONS.items() if k in ("all", name)),
            **evaluate(y[te], proba.argmax(1), proba),
        }
        path.write_text(json.dumps(res, indent=1))
        print(f"[{args.dataset}] DNN-{name} F1w={res['f1_weighted']:.4f} F1m={res['f1_macro']:.4f} "
              f"AUChard={res['auc_hard_macro']:.4f} AUCscore={res['auc_score_macro']:.4f} fit={fit_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
