"""Convert the authors' preprocessed LMD-2023 CSV to Parquet and print basic statistics.

Input: data/extracted/fullcsv/full-csv(Evaluated Titles).csv  (OHE + MinMax, as used in IJIS 2023)
Output: data/processed/fullcsv.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/extracted/fullcsv/full-csv(Evaluated Titles).csv"
DST = ROOT / "data/processed/fullcsv.parquet"


def main() -> None:
    header = pd.read_csv(SRC, nrows=0).columns.tolist()
    # The original header contains a duplicated "CompSTA3"; disambiguate it.
    seen: dict[str, int] = {}
    cols = []
    for c in header:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}__dup{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)

    df = pd.read_csv(SRC, engine="pyarrow").astype(np.float64)
    df.columns = cols
    df["Label"] = df["Label"].astype(np.int8)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)

    print(f"rows={len(df):,} cols={df.shape[1]} (features={df.shape[1] - 1})")
    print("duplicated header names:", [c for c in cols if "__dup" in c])
    print("label counts:\n", df["Label"].value_counts().sort_index())
    print("NaN cells:", int(df.isna().sum().sum()))
    n_dup = int(df.duplicated().sum())
    print(f"exact duplicate rows: {n_dup:,} ({n_dup / len(df):.1%})")
    feats = df.drop(columns="Label")
    conflict = df.groupby(list(feats.columns), sort=False)["Label"].nunique()
    print(f"unique feature vectors: {len(conflict):,}; with conflicting labels: {(conflict > 1).sum():,}")
    print("columns that are constant:", [c for c in feats.columns if feats[c].nunique() <= 1])


if __name__ == "__main__":
    main()
