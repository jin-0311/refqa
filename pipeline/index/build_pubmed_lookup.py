#!/usr/bin/env python3
"""Flatten the upstream source PubMed nested parquets into a slim PMID-keyed lookup table.

Input shards (1,426 files, ~40M articles total):
  ./external/data/raw/pubmed/parsed/
      baseline/pubmed26nXXXX.parquet
      updatefiles/pubmed26nXXXX.parquet

Output: data/interim/pubmed_lookup/
  shards/lookup_*.parquet  (one per input shard)
  pubmed_lookup.parquet    (coalesced, sorted by pmid)

Schema (kept slim because this index is meant for joins, not full retrieval):
  pmid (int64)
  doi (string)
  title (string)
  abstract (string)              # joined from sections with newlines
  journal_title (string)
  journal_iso (string)
  pub_year (int32)
  pub_types (list<string>)
  mesh_descriptors (list<string>)
  reference_pmids (list<int64>)  # cross-references parsed by the upstream source
  reference_dois (list<string>)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_IN = "./external/data/raw/pubmed/parsed"
DEFAULT_OUT = "./data/interim/pubmed_lookup"


def shard_to_lookup(shard_path: str, out_dir: str) -> tuple[str, int, float]:
    t0 = time.time()
    name = Path(shard_path).stem  # e.g. pubmed26n0001
    out = Path(out_dir) / "shards" / f"lookup_{name}.parquet"
    if out.exists():
        rows = pq.read_metadata(out).num_rows
        return shard_path, rows, 0.0  # already done

    tbl = pq.read_table(shard_path)

    # --- pmid -> int64 (drop rows without numeric pmid)
    pmid_str = tbl.column("pmid")
    pmid_int = pc.cast(pmid_str, pa.int64(), safe=False)
    valid_mask = pc.is_valid(pmid_int)

    # --- doi: from article_ids list-of-struct, pick id_type=='doi'
    def first_doi(ids):
        if ids is None:
            return None
        for rec in ids:
            if rec is not None and rec.get("id_type", "").lower() == "doi":
                return rec.get("value")
        return None

    article_ids = tbl.column("article_ids").to_pylist()
    dois = [first_doi(x) for x in article_ids]

    # --- title
    title = tbl.column("title")

    # --- abstract: join sections with double newline
    abstract_lst = tbl.column("abstract").to_pylist()
    def join_abs(secs):
        if not secs:
            return None
        parts = []
        for s in secs:
            if not s:
                continue
            label = s.get("label") or ""
            txt = s.get("text") or ""
            if not txt:
                continue
            parts.append(f"{label}: {txt}".strip(": ").strip() if label else txt)
        return "\n\n".join(parts) if parts else None
    abstracts = [join_abs(x) for x in abstract_lst]

    # --- journal info
    journal = tbl.column("journal").to_pylist()
    journal_title = [(j or {}).get("title") for j in journal]
    journal_iso = [(j or {}).get("iso_abbrev") for j in journal]

    # --- pub_year: prefer article_date.year, fallback to journal.pub_year
    article_date = tbl.column("article_date").to_pylist()
    def year_of(ad, j):
        if ad and ad.get("year"):
            return int(ad["year"])
        if j and j.get("pub_year"):
            try:
                return int(j["pub_year"])
            except (TypeError, ValueError):
                return None
        return None
    pub_years = [year_of(ad, j) for ad, j in zip(article_date, journal)]

    # --- publication_types
    pub_types_lst = tbl.column("publication_types").to_pylist()
    pub_types = [list(pt) if pt else [] for pt in pub_types_lst]

    # --- mesh_descriptors
    mesh_lst = tbl.column("mesh_headings").to_pylist()
    def mesh_names(m):
        if not m:
            return []
        return [x.get("descriptor_name") for x in m if x and x.get("descriptor_name")]
    mesh_descriptors = [mesh_names(m) for m in mesh_lst]

    # --- references: parse out PMIDs/DOIs
    refs_lst = tbl.column("references").to_pylist()
    ref_pmids_all, ref_dois_all = [], []
    for refs in refs_lst:
        rp, rd = [], []
        if refs:
            for r in refs:
                if r is None:
                    continue
                p = r.get("pmid")
                if p:
                    try:
                        rp.append(int(p))
                    except (TypeError, ValueError):
                        pass
                d = r.get("doi")
                if d:
                    rd.append(d)
        ref_pmids_all.append(rp)
        ref_dois_all.append(rd)

    out_tbl = pa.table({
        "pmid": pmid_int,
        "doi": pa.array(dois, type=pa.string()),
        "title": title,
        "abstract": pa.array(abstracts, type=pa.string()),
        "journal_title": pa.array(journal_title, type=pa.string()),
        "journal_iso": pa.array(journal_iso, type=pa.string()),
        "pub_year": pa.array(pub_years, type=pa.int32()),
        "pub_types": pa.array(pub_types, type=pa.list_(pa.string())),
        "mesh_descriptors": pa.array(mesh_descriptors, type=pa.list_(pa.string())),
        "reference_pmids": pa.array(ref_pmids_all, type=pa.list_(pa.int64())),
        "reference_dois": pa.array(ref_dois_all, type=pa.list_(pa.string())),
    })
    out_tbl = out_tbl.filter(valid_mask)

    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, out, compression="zstd", compression_level=3)
    return shard_path, out_tbl.num_rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--coalesce", action="store_true",
                    help="after shard pass, write a single sorted parquet")
    args = ap.parse_args()

    shards = sorted(Path(args.in_dir).glob("baseline/*.parquet")) + \
             sorted(Path(args.in_dir).glob("updatefiles/*.parquet"))
    print(f"[pubmed] found {len(shards)} input parquets", flush=True)
    Path(args.out_dir, "shards").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = 0
    total_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(shard_to_lookup, str(p), args.out_dir): p for p in shards}
        for fut in as_completed(futs):
            try:
                shard, rows, elapsed = fut.result()
                done += 1
                total_rows += rows
                if done % 50 == 0 or done <= 3 or done == len(shards):
                    el = time.time() - t0
                    print(f"[pubmed] {done}/{len(shards)} shards rows={total_rows:,} "
                          f"elapsed={el:.1f}s", flush=True)
            except Exception as e:
                p = futs[fut]
                print(f"[pubmed] FAIL {p}: {e}", flush=True)

    print(f"[pubmed] shard pass done: {done} shards rows={total_rows:,} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)

    if args.coalesce:
        # Write a partitioned dataset (one file per pmid_bucket = pmid // 1_000_000)
        # to dodge the 32-bit string-offset overflow that bites a single concat over
        # 41M rows (abstract column alone is multi-GB). pyarrow.dataset gives us
        # transparent multi-file lookups via pmid filter pushdown.
        import pyarrow.dataset as pds
        out = f"{args.out_dir}/coalesced"
        print(f"[pubmed] coalescing -> {out} (partitioned by pmid_bucket)", flush=True)
        src = pds.dataset(f"{args.out_dir}/shards", format="parquet")

        # Cast string columns to large_string in the schema we hand to write_dataset
        # so per-file string buffers can grow past 2 GB when needed.
        big_fields = []
        for f in src.schema:
            if pa.types.is_string(f.type):
                big_fields.append(pa.field(f.name, pa.large_string()))
            else:
                big_fields.append(f)
        target_schema = pa.schema(big_fields)

        def cast_batch(batch):
            cols = {}
            for name in batch.schema.names:
                col = batch.column(name)
                if pa.types.is_string(col.type):
                    cols[name] = col.cast(pa.large_string())
                else:
                    cols[name] = col
            cols["pmid_bucket"] = pc.divide(batch.column("pmid"),
                                             pa.scalar(1_000_000, pa.int64())).cast(pa.int32())
            return pa.RecordBatch.from_pydict(cols)

        # Stream batches into a partitioned dataset.
        scanner = src.scanner(batch_size=50_000)
        batches = (cast_batch(b) for b in scanner.to_batches())
        out_schema = target_schema.append(pa.field("pmid_bucket", pa.int32()))
        rb_reader = pa.RecordBatchReader.from_batches(out_schema, batches)
        pds.write_dataset(
            rb_reader, out, format="parquet",
            partitioning=pds.partitioning(pa.schema([("pmid_bucket", pa.int32())]),
                                          flavor="hive"),
            existing_data_behavior="overwrite_or_ignore",
            file_options=pds.ParquetFileFormat().make_write_options(
                compression="zstd", compression_level=5),
            max_rows_per_file=2_000_000,
        )
        print(f"[pubmed] coalesce DONE elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
