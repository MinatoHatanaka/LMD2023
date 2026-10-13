"""Stream the official labelled LMD-2023 CSV (1.75M version, 'Checked', 2024-03) to Parquet.

All columns are kept as strings so that nothing is silently coerced; analysis code casts
what it needs. Streaming keeps memory bounded (the CSV is ~1 GB with 93 columns).

Usage: uv run python src/convert_raw.py <csv> <out.parquet>
"""

import sys

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def main(src: str, dst: str) -> None:
    with open(src, "rb") as f:
        header = f.readline().decode("utf-8", "replace").rstrip("\r\n").split(",")
    # Disambiguate duplicated header names, if any.
    names, seen = [], {}
    for h in header:
        seen[h] = seen.get(h, -1) + 1
        names.append(h if seen[h] == 0 else f"{h}__dup{seen[h]}")
    reader = pacsv.open_csv(
        src,
        read_options=pacsv.ReadOptions(column_names=names, skip_rows=1, block_size=64 << 20, encoding="latin-1"),
        convert_options=pacsv.ConvertOptions(column_types={n: pa.string() for n in names}),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
    )
    writer, n = None, 0
    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(dst, batch.schema, compression="zstd")
        writer.write_batch(batch)
        n += batch.num_rows
    writer.close()
    print(f"{n:,} rows, {len(names)} columns -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
