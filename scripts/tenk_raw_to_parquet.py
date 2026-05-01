"""Convert raw 10-K audit files to ZSTD-compressed Parquet.

Streaming writer using PyArrow's ParquetWriter — each source file is read
one-at-a-time and written as a single row group directly to the output
Parquet file.  When the output file reaches *TARGET_BYTES* the writer
is closed and a new file is started.

Peak RAM: one source file's text content + one row group buffer (~1 row).

Usage
-----
    python scripts/tenk_raw_to_parquet.py
"""

import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "data/tenk_items"
DST = "hf_upload_folder"

# Close the current output file and start a new one when it exceeds this
# many bytes on disk.
TARGET_BYTES = 10 * 1024**3   # 10 GiB

_SCHEMA = pa.schema([
    ("ticker_symbol",    pa.utf8()),
    ("accession_number", pa.utf8()),
    ("raw_text",         pa.utf8()),
])

# ── helpers ───────────────────────────────────────────────────────────────────

def _count_files() -> int:
    total = 0
    for entry in os.scandir(SRC):
        if not entry.is_dir():
            continue
        for fe in os.scandir(entry.path):
            if fe.name.endswith(".htm") or fe.name.endswith(".txt"):
                total += 1
    return total


def _iter_files():
    """Lazily yield (ticker, accession_number, filepath) tuples."""
    for entry in os.scandir(SRC):
        if not entry.is_dir():
            continue
        ticker = entry.name
        for fe in os.scandir(entry.path):
            if fe.name.endswith(".htm") or fe.name.endswith(".txt"):
                acc_no = fe.name.removesuffix(".htm").removesuffix(".txt")
                yield ticker, acc_no, fe.path


def _human(b: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(b) < 1024:
            return f"{b:.0f} {unit}"
        b /= 1024
    return f"{b:.1f} TiB"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DST, exist_ok=True)

    if not os.path.isdir(SRC):
        print(f"Source directory not found: {SRC}", file=sys.stderr)
        sys.exit(1)

    print("Counting files …", end=" ", flush=True)
    total = _count_files()
    print(f"found {total} file(s)\n")

    t_start = time.perf_counter()
    n_done = 0
    output_idx = 0
    writer: pq.ParquetWriter | None = None
    cur_path: str | None = None

    for ticker, acc_no, fpath in _iter_files():
        # Open a new output file if needed.
        if writer is None:
            output_idx += 1
            cur_path = os.path.join(DST, f"batch_{output_idx:04d}.parquet")
            writer = pq.ParquetWriter(
                cur_path, _SCHEMA, compression="ZSTD"
            )

        # Read exactly one file.
        with open(fpath, "r", errors="replace") as fh:
            text = fh.read()

        # Build a single-row RecordBatch and write it as a row group.
        batch = pa.record_batch(
            [pa.array([ticker]), pa.array([acc_no]), pa.array([text])],
            schema=_SCHEMA,
        )
        writer.write_batch(batch)
        n_done += 1

        # Close the current file and start a new one if it's big enough.
        if cur_path and os.path.getsize(cur_path) >= TARGET_BYTES:
            writer.close()
            writer = None
            cur_path = None

        pct = n_done / total * 100
        print(
            f"\r  [{n_done:>6}/{total}]  "
            f"output {output_idx:>4}  "
            f"({pct:>5.1f} %)".ljust(60),
            end="",
            flush=True,
        )

    if writer is not None:
        writer.close()

    elapsed = time.perf_counter() - t_start
    print(f"\n\nDone — {n_done} file(s) → {DST}/  in {elapsed:.1f} s")
    print(f"       {output_idx} Parquet file(s) created")

    # Show final sizes.
    total_bytes = 0
    for f in sorted(os.listdir(DST)):
        fp = os.path.join(DST, f)
        if f.endswith(".parquet"):
            sz = os.path.getsize(fp)
            total_bytes += sz
            print(f"         {f}:  {_human(sz)}")
    print(f"         total: {_human(total_bytes)}")


if __name__ == "__main__":
    main()
