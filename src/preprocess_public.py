"""Rebuild the paper's 15-feature matrix from the public labelled LMD-2023 (1.75M, 'Checked').

Follows IJIS 2023 Sect. 4.1-4.2 and the authors' dfToOHEMinMax.py:
  * keep EventID, SystemTime, EventRecordID, Execution_ProcessID, Computer, ProcessId,
    Initiated, SourceIsIpv6, DestinationPortName, Label
  * drop rows whose DestinationPortName is '-', and rows whose numeric fields are not parseable
  * split SystemTime into year, month, ISO week, day, hour, minute, day_of_week
  * one-hot: EventID, Computer, Initiated, SourceIsIpv6, DestinationPortName, year, month, week, dow
  * min-max: EventRecordID, Execution_ProcessID, ProcessId, day, hour, minute
Scaling/encoding are fitted on the whole table, as the authors did (before any split).

Outputs
  data/processed/public15_raw.parquet    : the 15 un-encoded features + Label (+ Computer kept as text)
  data/processed/public15_matrix.parquet : encoded matrix + Label
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/lmd2023_1.75M_labelled.parquet"
OUT_RAW = ROOT / "data/processed/public15_raw.parquet"
OUT_MAT = ROOT / "data/processed/public15_matrix.parquet"

NUMERIC = ["EventRecordID", "Execution_ProcessID", "ProcessId"]
CATEG = ["EventID", "Computer", "Initiated", "SourceIsIpv6", "DestinationPortName"]
TIME_OHE = ["SystemTime_year", "SystemTime_month", "SystemTime_week", "SystemTime_day_of_week"]
TIME_NUM = ["SystemTime_day", "SystemTime_hour", "SystemTime_minute"]


def build_raw() -> pd.DataFrame:
    df = pd.read_parquet(SRC, columns=["SystemTime", "Label"] + NUMERIC + CATEG)
    n0 = len(df)
    df = df[df["DestinationPortName"].fillna("") != "-"]
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    t = pd.to_datetime(df["SystemTime"], format="mixed", errors="coerce")
    df["EventID"] = pd.to_numeric(df["EventID"], errors="coerce")
    ok = df[NUMERIC + ["EventID"]].notna().all(axis=1) & t.notna()
    df, t = df[ok].copy(), t[ok]
    df["Label"] = df["Label"].astype(np.int8)
    df["EventID"] = df["EventID"].astype(int)
    df["SystemTime_year"] = t.dt.year
    df["SystemTime_month"] = t.dt.month
    df["SystemTime_week"] = t.dt.isocalendar().week.astype(int)
    df["SystemTime_day"] = t.dt.day
    df["SystemTime_hour"] = t.dt.hour
    df["SystemTime_minute"] = t.dt.minute
    df["SystemTime_day_of_week"] = t.dt.dayofweek
    df["t"] = t
    for c in ["Computer", "Initiated", "SourceIsIpv6", "DestinationPortName"]:
        df[c] = df[c].fillna("0").astype(str)
    print(f"rows kept {len(df):,} / {n0:,} (dropped {n0 - len(df):,})")
    return df.drop(columns=["SystemTime"]).reset_index(drop=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[CATEG + TIME_OHE].astype(str), columns=CATEG + TIME_OHE, dtype=np.float32)
    num = df[NUMERIC + TIME_NUM].astype(np.float64)
    num = (num - num.min()) / (num.max() - num.min()).replace(0, 1)
    out = pd.concat([num.astype(np.float32), X], axis=1)
    out["Label"] = df["Label"].to_numpy()
    return out


def main() -> None:
    raw = build_raw()
    raw.to_parquet(OUT_RAW, index=False)
    mat = encode(raw)
    mat.to_parquet(OUT_MAT, index=False)
    print("label counts:", raw["Label"].value_counts().sort_index().to_dict())
    print(f"matrix: {mat.shape[0]:,} rows x {mat.shape[1] - 1} features")


if __name__ == "__main__":
    main()
