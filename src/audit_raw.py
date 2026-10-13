"""Audit the official labelled LMD-2023 (1.75M) raw table.

Writes results/audit/raw_audit.json and prints a human-readable summary.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/lmd2023_1.75M_labelled.parquet"
POLICY = ROOT / "data/extracted/rbpolicy/RBPolicy"
OUT = ROOT / "results/audit"

COLS = ["SystemTime", "Label", "EventID", "EventRecordID", "Execution_ProcessID", "Computer",
        "ProcessId", "Image", "CommandLine", "Initiated", "SourceIsIpv6", "DestinationPortName", "UtcTime"]


def relabel(df: pd.DataFrame) -> pd.Series:
    """Re-implementation of ETCExp labelling.py: exact match of Image/CommandLine to the policy lists."""
    eors = set((POLICY / "EoRSPolicy.txt").read_text(encoding="latin-1").splitlines())
    eoht = set((POLICY / "EoHTPolicy.txt").read_text(encoding="latin-1").splitlines())
    img, cmd = df["Image"].fillna("nan").astype(str), df["CommandLine"].fillna("nan").astype(str)
    lab = np.zeros(len(df), dtype=np.int8)
    is1 = img.isin(eors) | cmd.isin(eors)
    is2 = ~is1 & (img.isin(eoht) | cmd.isin(eoht))
    lab[is2.to_numpy()] = 2
    lab[is1.to_numpy()] = 1
    return pd.Series(lab, index=df.index)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC, columns=COLS)
    res: dict = {"n_rows": len(df)}
    print("Label raw values:", df["Label"].value_counts(dropna=False).to_dict())
    df["y"] = pd.to_numeric(df["Label"], errors="coerce")
    res["label_counts"] = df["y"].value_counts(dropna=False).sort_index().to_dict()

    # Host x label
    hl = pd.crosstab(df["Computer"].fillna("<NA>"), df["y"])
    res["host_label"] = hl.to_dict(orient="index")
    print("\nHost x label:\n", hl)

    # Time ranges per host x label
    t = pd.to_datetime(df["SystemTime"], errors="coerce", format="mixed")
    res["systemtime_unparsable"] = int(t.isna().sum())
    df["t"] = t
    tr = df.groupby([df["Computer"].fillna("<NA>"), "y"])["t"].agg(["min", "max", "count"])
    print("\nSystemTime range per host x label:\n", tr)
    res["time_range"] = {f"{h}|{int(y)}": [str(r["min"]), str(r["max"]), int(r["count"])] for (h, y), r in tr.iterrows()}
    ym = pd.crosstab(df["t"].dt.to_period("M").astype(str), df["y"])
    print("\nYear-month x label:\n", ym)
    res["month_label"] = ym.to_dict(orient="index")

    # Top images per label
    res["top_images"] = {}
    for y in sorted(df["y"].dropna().unique()):
        top = df.loc[df["y"] == y, "Image"].fillna("<NA>").value_counts().head(15)
        res["top_images"][int(y)] = top.to_dict()
        print(f"\nTop Image for label {int(y)}:\n", top)

    # Re-labelling with the published rule lists
    rl = relabel(df)
    agree = pd.crosstab(df["y"], rl, rownames=["dataset"], colnames=["relabel"])
    print("\nDataset label vs. re-applied published policy:\n", agree)
    res["relabel_crosstab"] = agree.to_dict(orient="index")
    res["relabel_agreement"] = float((df["y"].to_numpy() == rl.to_numpy()).mean())

    # Duplicates
    res["dup_rows_selected_cols"] = int(df[COLS].duplicated().sum())
    rid = df.groupby(df["Computer"].fillna("<NA>"))["EventRecordID"].agg(["size", "nunique"])
    print("\nEventRecordID uniqueness per host:\n", rid)
    res["recordid_uniqueness"] = rid.to_dict(orient="index")
    key = ["Computer", "EventRecordID"]
    res["dup_host_recordid"] = int(df.duplicated(key).sum())
    print("\nduplicate (Computer, EventRecordID) rows:", res["dup_host_recordid"])
    print("duplicate rows over audited columns:", res["dup_rows_selected_cols"])

    (OUT / "raw_audit.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
