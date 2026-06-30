#!/usr/bin/env python3
"""Slim iCite metadata CSV (~28 GB) into a PMID-keyed Parquet.

Keeps only the columns we'll actually join into the Q&A dataset:
  pmid, doi, year, journal, citation_count, relative_citation_ratio,
  nih_percentile, is_research_article, is_clinical, expected_citations_per_year,
  field_citation_rate, human, animal, molecular_cellular, apt

Streams the CSV in chunks so memory stays bounded; writes one Parquet shard per
chunk under data/interim/icite_lookup/, then writes a single coalesced output
file (icite_lookup.parquet) at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

DEFAULT_CSV = "./external/data/raw/icite/icite_metadata.csv"
DEFAULT_OUT = "./data/interim/icite_lookup"

KEEP_COLUMNS = [
    "pmid",
    "doi",
    "year",
    "journal",
    "citation_count",
    "relative_citation_ratio",
    "nih_percentile",
    "is_research_article",
    "is_clinical",
    "expected_citations_per_year",
    "citations_per_year",
    "field_citation_rate",
    "human",
    "animal",
    "molecular_cellular",
    "apt",
]

# Force types where pyarrow's auto-inference is unreliable on huge CSVs.
COLUMN_TYPES = {
    "pmid": pa.int64(),
    "doi": pa.string(),
    "year": pa.int32(),
    "journal": pa.string(),
    "citation_count": pa.int32(),
    "relative_citation_ratio": pa.float32(),
    "nih_percentile": pa.float32(),
    "is_research_article": pa.bool_(),
    "is_clinical": pa.bool_(),
    "expected_citations_per_year": pa.float32(),
    "citations_per_year": pa.float32(),
    "field_citation_rate": pa.float32(),
    "human": pa.float32(),
    "animal": pa.float32(),
    "molecular_cellular": pa.float32(),
    "apt": pa.float32(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--chunk-rows", type=int, default=500_000,
                    help="rows per parquet shard")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    final = out_dir / "icite_lookup.parquet"

    read_opts = pacsv.ReadOptions(block_size=64 << 20)  # 64 MB blocks
    parse_opts = pacsv.ParseOptions(quote_char='"', double_quote=True)
    convert_opts = pacsv.ConvertOptions(
        include_columns=KEEP_COLUMNS,
        column_types=COLUMN_TYPES,
        null_values=["", "NA", "None"],
        strings_can_be_null=True,
    )

    t0 = time.time()
    n_rows = 0
    n_shards = 0
    buffer = []
    buffered_rows = 0

    print(f"[icite] reading {args.csv}", flush=True)
    with pacsv.open_csv(args.csv,
                         read_options=read_opts,
                         parse_options=parse_opts,
                         convert_options=convert_opts) as reader:
        for batch in reader:
            buffer.append(batch)
            buffered_rows += batch.num_rows
            if buffered_rows >= args.chunk_rows:
                tbl = pa.Table.from_batches(buffer)
                shard_path = shard_dir / f"shard_{n_shards:05d}.parquet"
                pq.write_table(tbl, shard_path, compression="zstd",
                               compression_level=3)
                n_shards += 1
                n_rows += buffered_rows
                if n_shards % 5 == 0 or n_shards <= 3:
                    elapsed = time.time() - t0
                    rate = n_rows / max(elapsed, 1)
                    print(f"[icite] shards={n_shards} rows={n_rows:,} "
                          f"elapsed={elapsed:.1f}s rate={rate:,.0f} rows/s",
                          flush=True)
                buffer = []
                buffered_rows = 0

        if buffer:
            tbl = pa.Table.from_batches(buffer)
            shard_path = shard_dir / f"shard_{n_shards:05d}.parquet"
            pq.write_table(tbl, shard_path, compression="zstd",
                           compression_level=3)
            n_shards += 1
            n_rows += buffered_rows

    elapsed = time.time() - t0
    print(f"[icite] CSV pass done: rows={n_rows:,} shards={n_shards} "
          f"elapsed={elapsed:.1f}s", flush=True)

    # Coalesce shards -> single sorted parquet
    print(f"[icite] coalescing -> {final}", flush=True)
    shards = sorted(shard_dir.glob("shard_*.parquet"))
    tables = [pq.read_table(p) for p in shards]
    big = pa.concat_tables(tables)
    # Sort by pmid for fast point lookups via row-group statistics.
    sort_idx = pa.compute.sort_indices(big.column("pmid"))
    big = big.take(sort_idx)
    pq.write_table(big, final, compression="zstd", compression_level=5,
                   row_group_size=200_000)
    final_size = final.stat().st_size
    print(f"[icite] DONE rows={big.num_rows:,} file={final} "
          f"size={final_size/1024/1024:.1f}MB elapsed={time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
