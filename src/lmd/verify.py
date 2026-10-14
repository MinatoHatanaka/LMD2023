"""Feature sets, group keys and splitters for the H1/H2/H5 verification experiments.

Feature sets (columns removed from the paper's 15-feature matrix)
  F0 all | F1 no time (SystemTime-derived) | F2 no host (Computer) | F3 no IDs
  (EventRecordID, Execution_ProcessID, ProcessId) | F4 = F1+F2+F3 removed.

Experiments
  E1  stratified 5-fold (random_state=42)                      reference (paper-like)
  E2  leave-one-host-out                                       H1
  E3  stratified group 5-fold over time blocks                 H2
      authors: ISO year x week, public15: calendar day
  E3h public15 only: group 5-fold over 1-hour blocks           H2 (finer blocks)
  E3c authors only: chronological, train <= 2022-06, test later   H2 (reference)
  E4a exact duplicates (feature vector + label, under the feature set) removed, then E1   H5
  E4b group 5-fold where identical feature vectors never straddle train/test               H5

Duplicates of public15 are identified on the raw un-scaled values (float64 / strings), since
the float32 min-max matrix merges distinct rows.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from lmd.data import ROOT, load_matrix

FEATURESETS = ["F0", "F1", "F2", "F3", "F4"]
EXPERIMENTS = {
    "authors": ["E1", "E2", "E3", "E3c", "E4a", "E4b"],
    "public15": ["E1", "E2", "E3", "E3h", "E4a", "E4b"],
}
DROP = {"F0": [], "F1": ["time"], "F2": ["host"], "F3": ["id"], "F4": ["time", "host", "id"]}

# Column groups in the encoded matrices.
MATRIX_GROUPS = {
    "authors": {
        "time": lambda c: c.startswith("SysTime"),
        "host": lambda c: c.startswith("CompSTA"),
        "id": lambda c: c in ("EventRecID", "ExecProcessID", "ProcessId"),
    },
    "public15": {
        "time": lambda c: c.startswith("SystemTime_"),
        "host": lambda c: c.startswith("Computer_"),
        "id": lambda c: c in ("EventRecordID", "Execution_ProcessID", "ProcessId"),
    },
}
# Raw (un-encoded) public15 columns per group, used for duplicate detection.
PUBLIC_RAW = {
    "time": ["SystemTime_year", "SystemTime_month", "SystemTime_week", "SystemTime_day",
             "SystemTime_hour", "SystemTime_minute", "SystemTime_day_of_week"],
    "host": ["Computer"],
    "id": ["EventRecordID", "Execution_ProcessID", "ProcessId"],
    "rest": ["EventID", "Initiated", "SourceIsIpv6", "DestinationPortName"],
}


@dataclass
class Data:
    X: pd.DataFrame          # encoded matrix (all features)
    y: np.ndarray
    host: np.ndarray         # host id per row (str)
    time_block: np.ndarray   # E3 block key (sortable int)
    time_fine: np.ndarray    # finer time key for MAJ-time / E3h (sortable int)
    dup_frame: pd.DataFrame  # frame on which exact duplicates are judged (all features)
    dup_groups: dict         # feature group -> columns of dup_frame


def load(dataset: str) -> Data:
    X, y = load_matrix(dataset)
    if dataset == "authors":
        hosts = [c for c in X if c.startswith("CompSTA")]
        host = X[hosts].to_numpy().argmax(1)
        host = np.asarray(hosts)[host]
        year = _onehot_value(X, "SysTimeYear_")
        week = _onehot_value(X, "SysTimeWeek_")
        time_block = year * 100 + week
        dup_frame = X.astype(np.float64)
        dup_groups = {g: [c for c in X if f(c)] for g, f in MATRIX_GROUPS[dataset].items()}
        return Data(X, y, host, time_block, time_block, dup_frame, dup_groups)
    raw = pd.read_parquet(ROOT / "data/processed/public15_raw.parquet")
    assert np.array_equal(raw["Label"].to_numpy(), y)
    t = raw["t"]
    day = (t.dt.year * 10000 + t.dt.month * 100 + t.dt.day).to_numpy()
    hour = day * 100 + t.dt.hour.to_numpy()
    cols = sum(PUBLIC_RAW.values(), [])
    return Data(X, y, raw["Computer"].to_numpy(str), day, hour, raw[cols],
                {g: c for g, c in PUBLIC_RAW.items() if g != "rest"})


def _onehot_value(X: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [c for c in X if c.startswith(prefix)]
    vals = np.array([int(c[len(prefix):]) for c in cols])
    return vals[X[cols].to_numpy().argmax(1)]


def feature_columns(dataset: str, X: pd.DataFrame, fs: str) -> list[str]:
    groups = MATRIX_GROUPS[dataset]
    return [c for c in X if not any(groups[g](c) for g in DROP[fs])]


def dup_keys(d: Data, fs: str) -> np.ndarray:
    """64-bit hash of each row's feature vector under feature set fs (exact-duplicate key)."""
    drop = sum((d.dup_groups[g] for g in DROP[fs]), [])
    frame = d.dup_frame.drop(columns=drop)
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def fast_stratified_group_kfold(groups, y, n_splits=5, seed=42):
    """Group k-fold for very many groups: groups are stratified by their majority label,
    shuffled, ordered by size and dealt round-robin to folds (vectorised)."""
    codes, inv = np.unique(groups, return_inverse=True)
    n_g = len(codes)
    size = np.bincount(inv, minlength=n_g)
    counts = np.zeros((n_g, 3), np.int64)
    np.add.at(counts, (inv, y.astype(int)), 1)
    major = counts.argmax(1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_g)
    order = order[np.lexsort((-size[order], major[order]))]  # by label stratum, then size desc
    rank = np.empty(n_g, np.int64)
    for c in range(3):
        idx = order[major[order] == c]
        rank[idx] = np.arange(len(idx))
    fold_of_group = rank % n_splits
    fold = fold_of_group[inv]
    idx = np.arange(len(y))
    for k in range(n_splits):
        yield k, idx[fold != k], idx[fold == k]


def splits(d: Data, dataset: str, exp: str, fs: str):
    """Yield (fold, train_idx, test_idx) in the index space of the full dataset."""
    y, idx = d.y, np.arange(len(d.y))
    if exp == "E1":
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(idx, y)):
            yield k, tr, te
    elif exp == "E2":
        names = np.unique(d.host)
        for tr, te in LeaveOneGroupOut().split(idx, y, d.host):
            yield int(np.searchsorted(names, d.host[te[0]])), tr, te
    elif exp in ("E3", "E3h"):
        g = d.time_block if exp == "E3" else d.time_fine
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        for k, (tr, te) in enumerate(sgkf.split(idx, y, g)):
            yield k, tr, te
    elif exp == "E3c":
        cut = 202225  # authors: train on blocks up to ISO 2022-W25, test on 2022-W47 onwards
        yield 0, idx[d.time_block <= cut], idx[d.time_block > cut]
    elif exp == "E4a":
        key = pd.DataFrame({"k": dup_keys(d, fs), "y": y})
        keep = idx[~key.duplicated().to_numpy()]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for k, (tr, te) in enumerate(skf.split(keep, y[keep])):
            yield k, keep[tr], keep[te]
    elif exp == "E4b":
        yield from fast_stratified_group_kfold(dup_keys(d, fs), y)
    else:
        raise ValueError(exp)


def fold_name(d: Data, exp: str, k: int) -> str:
    return str(np.unique(d.host)[k]) if exp == "E2" else str(k)
