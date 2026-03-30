#!/usr/bin/env python3
"""
Uniformly sample rows *across* many Parquet files and emit sharded JSONL.

• Strict per-row uniformity (alias-table weighted sampler)
• Streaming – memory bounded by: 1 GiB buffer + O(#files) metadata
• Flushes to shard_<idx>_<uid>.jsonl when buffer reaches buffer_max_bytes
"""

from __future__ import annotations
import argparse, json, os, random, uuid
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import time
import pyarrow.json as paj
import numpy as np


# --------------------------------------------------------------------- #
# Alias-table categorical sampler (Vose, 1991)
# --------------------------------------------------------------------- #
def build_alias(weights: List[float]) -> tuple[list[float], list[int]]:
    """Return (prob, alias) tables for O(1) sampling of range(len(weights))."""
    n = len(weights)
    if n == 0 or sum(weights) == 0:
        return [], []
    scaled = [w * n / sum(weights) for w in weights]
    prob = [0.0] * n
    alias = [0] * n
    small, large = [], []
    for idx, p in enumerate(scaled):
        (small if p < 1.0 else large).append(idx)
    while small and large:
        s, l = small.pop(), large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1
        (small if scaled[l] < 1.0 else large).append(l)
    for i in large + small:   # leftovers
        prob[i] = 1.0
    return prob, alias


def sample_alias(prob: list[float], alias: list[int], rng=random.random) -> int:
    """Draw one index according to the distribution encoded by (prob, alias)."""
    n = len(prob)
    if n == 0:
        raise RuntimeError("Alias table empty")
    i = int(rng() * n)
    return i if rng() < prob[i] else alias[i]


# --------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------- #
def approx_json_size(obj) -> int:
    """UTF-8 bytes needed to jsonify obj plus newline."""
    return len(json.dumps(obj, ensure_ascii=False)) + 1


def count_rows(file_path: Path) -> int:
    """Return #rows in either a Parquet or JSONL file."""
    if file_path.suffix == ".parquet":
        pf = pq.ParquetFile(file_path)
        row_total = 0
        for batch in pf.iter_batches(batch_size=8192, columns=[]):  # zero-column batches are cheap
            row_total += batch.num_rows
        return row_total
    elif file_path.suffix == ".jsonl":
        # plain line count; fast enough when done once at start
        with file_path.open("rb") as f:
            return sum(1 for _ in f)
    else:
        raise ValueError(f"Unsupported extension: {file_path}")

def iter_jsonl_batches(path: Path, batch_size: int, cols=("text",)) -> pa.RecordBatchReader:
    """Yield RecordBatches of ≤ batch_size rows from a JSONL file."""
    with path.open("rb") as fh:
        rows, buf = 0, {c: [] for c in cols}
        for line in fh:
            obj = json.loads(line)
            for c in cols:
                buf[c].append(obj.get(c))
            rows += 1
            if rows == batch_size:
                yield pa.RecordBatch.from_pydict(buf)
                rows, buf = 0, {c: [] for c in cols}
        if rows:
            yield pa.RecordBatch.from_pydict(buf)

def make_reader(path: Path, batch_size: int):
    """Return an iterator of RecordBatches for either format."""
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                 use_threads=True,
                                                 columns=["text", "messages", "file_contents", "content"])
    elif path.suffix == ".jsonl":
        return iter_jsonl_batches(path, batch_size)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")
# --------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------- #
def main():
    args = parse_cli()
    if args.output_dir is None:
        args.output_dir = Path(str(args.parquet_dir).replace("data_2B", "data_2B_shuffle"))

    random.seed(args.seed)

    parquet_files = sorted(Path(args.parquet_dir).rglob("*.parquet")) + sorted(Path(args.parquet_dir).rglob("*.jsonl"))
    if not parquet_files:
        raise FileNotFoundError(f"No *.parquet found under {args.parquet_dir}")

    print(f"Found {len(parquet_files):,} parquet files → scanning row counts …")

    # ------------- first pass: metadata only (fast) ------------------ #
    # shuffle files order
    random.shuffle(parquet_files)
    # use start and end indices to limit processing
    if args.start_idx is None:
        args.start_idx = 0
    else:
        args.start_idx = max(0, args.start_idx)
        if args.end_idx is None:
            args.end_idx = int(args.start_idx + 256)
    if args.end_idx is not None:
        parquet_files = parquet_files[args.start_idx:args.end_idx + 1]
    else:
        parquet_files = parquet_files[args.start_idx:]
    if len(parquet_files) == 0:
        print(f"No parquet files in range {args.start_idx} to {args.end_idx} found.")
        return 0
    row_counts = [count_rows(fp) for fp in parquet_files]
    total_rows = sum(row_counts)
    total_rows_all = total_rows
    print(f"Total rows: {total_rows:,}")

    # live state
    remaining = row_counts[:]                        # mutable copy
    # parquet_readers = [pq.ParquetFile(fp).iter_batches(
    #                        batch_size=args.batch_size, use_threads=True, columns=["text"])
    #                    for fp in parquet_files]
    parquet_readers = [make_reader(fp, args.batch_size) for fp in parquet_files]

    prob, alias = build_alias(remaining)             # build initial sampler

    # ------------------------------------------------------------------
    # Buffers for shuffling / writing
    # ------------------------------------------------------------------
    buffer_batches: List[pa.RecordBatch] = []
    buffer_rows = 0
    shard_idx   = 0
    out_dir     = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Streaming loop
    # ------------------------------------------------------------------
    start_time = time.time()
    last_progress = 0
    while total_rows:
        f_idx = sample_alias(prob, alias)

        if remaining[f_idx] == 0:      # defensive (shouldn't happen)
            continue

        try:
            batch = next(parquet_readers[f_idx])
        except StopIteration:
            remaining[f_idx] = 0
            prob, alias = build_alias(remaining)
            continue

        batch_rows = batch.num_rows
        buffer_batches.append(batch)
        buffer_rows += batch_rows

        remaining[f_idx] -= batch_rows
        total_rows       -= batch_rows
        if remaining[f_idx] == 0:
            prob, alias = build_alias(remaining)

        # Flush when buffer exceeds threshold
        if buffer_rows >= args.buffer_rows:
            flush_buffer(buffer_batches, shard_idx, out_dir, args.start_idx, args.end_idx)
            shard_idx   += 1
            buffer_rows  = 0
            buffer_batches.clear()
        
        if shard_idx > last_progress:
            last_progress = shard_idx
            print(f"Progress: {total_rows_all - total_rows:,} / {total_rows_all:,} rows processed")
            print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
            start_time = time.time()  # reset timer
            # if shard_idx == 4:
            #     import sys
            #     sys.exit(0)  # early exit for testing

    # Final flush
    if buffer_rows:
        flush_buffer(buffer_batches, shard_idx, out_dir, args.start_idx, args.end_idx)

    print(f"✔ Completed – produced {shard_idx + 1:,} shard(s)")


# ──────────────────────────────────────────────────────────────────────
#  Flush helper – Arrow shuffle + C++ JSON writer
# ──────────────────────────────────────────────────────────────────────
def flush_buffer(batches: List[pa.RecordBatch], shard_idx: int, out_dir: Path, start_idx: int = 0, end_idx: int = None):
    total = sum(b.num_rows for b in batches)
    if end_idx is None:
        end_idx = 0
    print(f"• Flushing {total:,} rows → shard_{shard_idx:05d}_{start_idx}_{end_idx}.parquet")

    tables = [pa.Table.from_batches([b]) for b in batches]

    # 2) Union the schemas so missing columns become null-filled
    # table = pa.concat_tables(tables, unify_schemas=True)   # Arrow ≥ 10
    table = pa.Table.from_batches(batches,schema=None) 
    rng = np.random.default_rng()
    perm = rng.permutation(total)
    shuffled = table.take(pa.array(perm, type=pa.int64()))

    fname = out_dir / f"shard_{shard_idx:05d}_{start_idx}_{end_idx}_{uuid.uuid4().hex[:8]}.parquet"
    pq.write_table(
        shuffled,
        fname,
        compression=None,                # or 'zstd' for smaller files
        write_statistics=False,              # LM pre-training rarely needs stats
        data_page_size=32_768,         # 32 KiB * 2 for better compression
    )
    print(f"  → wrote {shuffled.num_rows:,} rows to {fname}")



# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uniformly shuffle rows across Parquet files into JSONL shards "
                    "using an alias sampler and Arrow's fast JSON writer.")
    p.add_argument("--parquet-dir", required=True, type=Path,
                   help="Directory (recursively) containing *.parquet")
    p.add_argument("--output-dir",  required=False, type=Path, default=None,
                   help="Destination folder for *.jsonl shards")
    p.add_argument("--batch-size",  type=int, default=256,
                   help="Rows read per Parquet batch (default 4096)")
    p.add_argument("--buffer-rows", type=int, default=2_00_000,
                   help="Flush when buffered rows ≥ this (default 2 M)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible shuffling")
    p.add_argument("--start-idx", type=int, default=None,
                   help="Start shard index (default 0)")
    p.add_argument("--end-idx", type=int, default=None,
                   help="End shard index (inclusive, default all)")
    return p.parse_args()


if __name__ == "__main__":
    main()

