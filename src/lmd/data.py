from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    # Authors' preprocessed matrix "full-csv(Evaluated Titles).csv" (Jan 2023, 90 features).
    "authors": ROOT / "data/processed/fullcsv.parquet",
    # Our rebuild of the 15 paper features from the public labelled 1.75M 'Checked' release.
    "public15": ROOT / "data/processed/public15_matrix.parquet",
}


def load_matrix(name: str):
    df = pd.read_parquet(DATASETS[name])
    y = df.pop("Label").to_numpy(np.int8)
    return df, y


def test_in_train_rate(X_train: np.ndarray, X_test: np.ndarray) -> float:
    """Fraction of test rows whose exact feature vector also occurs in the training set.

    Rows are compared through 64-bit hashes (collisions are negligible at this scale).
    """
    return float(np.isin(row_hash(X_test), np.unique(row_hash(X_train))).mean())


def row_hash(a: np.ndarray) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.DataFrame(a), index=False).to_numpy()
