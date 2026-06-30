#!/usr/bin/env python3
"""Stream all PMC OA citation JSONLs, keep clinical-both citations, join with
PubMed (citing+cited) and iCite (cited), write enriched JSONL ready for GLM.

Strategy:
  1. Load clinical PMID set (~1.3M) from iCite.
  2. Pre-load PubMed metadata for all clinical PMIDs into a {pmid: dict} map
     by streaming the pubmed_lookup dataset (41M rows, keep ~1.3M).
  3. Pre-load iCite slim columns for clinical PMIDs.
  4. Fork worker pool; each worker reads one JATS JSONL.gz, emits clinical-both
     citations to its own enriched JSONL fragment.
  5. Main process concatenates fragments.

Memory: ~5-6 GB (1.3M clinical PMIDs × ~3KB PubMed record).
Workers share memory via fork() COW.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

CIT_ROOT  = "./data/interim/citations"
PUBMED_DS = "./data/interim/pubmed_lookup/shards"
ICITE     = "./data/interim/icite_lookup/icite_lookup.parquet"
DEFAULT_OUT = "./data/interim/pilot/enriched_clinical_full"

# Globals populated in main process before fork
_CLINICAL: frozenset[int] = frozenset()
_PM: dict[int, dict] = {}
_IC: dict[int, dict] = {}

SECTION_BUCKETS = {
    "introduction": re.compile(r"\b(introduction|background|overview)\b", re.I),
    "methods":      re.compile(r"\b(method|materials|design|experimental|protocol)\b", re.I),
    "results":      re.compile(r"\b(result|finding)\b", re.I),
    "discussion":   re.compile(r"\b(discussion|conclusion|interpretation|implication)\b", re.I),
}


def section_bucket(name: str | None) -> str:
    if not name:
        return "other"
    for b, rx in SECTION_BUCKETS.items():
        if rx.search(name):
            return b
    return "other"


def load_clinical_pmids() -> frozenset[int]:
    t0 = time.time()
    t = pq.read_table(ICITE, columns=["pmid", "is_clinical"],
                      filters=[("is_clinical", "=", True)])
    s = frozenset(t.column("pmid").to_pylist())
    print(f"[load] clinical PMIDs: {len(s):,}  ({time.time()-t0:.1f}s)", flush=True)
    return s


def load_pubmed_for(clinical: frozenset[int]) -> dict[int, dict]:
    """Stream pubmed_lookup shards, keep only clinical PMIDs."""
    t0 = time.time()
    d = ds.dataset(PUBMED_DS, format="parquet")
    cols = ["pmid", "doi", "title", "abstract", "journal_title",
            "pub_year", "pub_types", "mesh_descriptors"]
    out: dict[int, dict] = {}
    scanner = d.scanner(columns=cols, batch_size=50_000)
    n_batches = 0
    n_seen = 0
    for batch in scanner.to_batches():
        n_batches += 1
        pmids = batch.column("pmid").to_pylist()
        n_seen += len(pmids)
        keep_idx = [i for i, p in enumerate(pmids) if p in clinical]
        if not keep_idx:
            continue
        sub = batch.take(keep_idx)
        rows = sub.to_pylist()
        for r in rows:
            out[r["pmid"]] = r
        if n_batches % 100 == 0:
            print(f"[load] pubmed batch {n_batches}  rows_seen={n_seen:,}  "
                  f"hits={len(out):,}  elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[load] pubmed lookup: {len(out):,} clinical hits "
          f"(of {n_seen:,} rows, {time.time()-t0:.1f}s)", flush=True)
    return out


def load_icite_for(clinical: frozenset[int]) -> dict[int, dict]:
    t0 = time.time()
    t = pq.read_table(
        ICITE,
        columns=["pmid", "citation_count", "relative_citation_ratio",
                 "nih_percentile", "is_clinical", "is_research_article",
                 "year", "journal"],
        filters=[("is_clinical", "=", True)],
    )
    out = {r["pmid"]: r for r in t.to_pylist()}
    print(f"[load] icite lookup: {len(out):,} clinical "
          f"({time.time()-t0:.1f}s)", flush=True)
    return out


def build_enriched(art: dict, cit: dict, ref: dict,
                   citing_pm: dict | None, cited_pm: dict, cited_ic: dict) -> dict:
    return {
        "citing": {
            "pmcid": art.get("pmcid"),
            "pmid": art.get("pmid"),
            "doi": art.get("doi"),
            "title": art.get("title") or (citing_pm or {}).get("title"),
            "journal": art.get("journal") or (citing_pm or {}).get("journal_title"),
            "year": art.get("year") or (citing_pm or {}).get("pub_year"),
            "license": art.get("license"),
            "subset": art.get("subset"),
            "mesh": (citing_pm or {}).get("mesh_descriptors"),
            "pub_types": (citing_pm or {}).get("pub_types"),
        },
        "context": {
            "section": cit.get("section"),
            "section_bucket": section_bucket(cit.get("section")),
            "paragraph_idx": cit.get("paragraph_idx"),
            "in_table": cit.get("in_table"),
            "in_fig": cit.get("in_fig"),
            "evidence_sentence": cit.get("sentence"),
            "n_co_cited_refs": len(cit.get("ref_ids", [])),
            "co_cited_ref_ids": cit.get("ref_ids"),
        },
        "cited": {
            "ref_id": ref.get("ref_id"),
            "pmid": ref["pmid"],
            "doi": ref.get("doi") or cited_pm.get("doi"),
            "title": cited_pm.get("title") or ref.get("title"),
            "abstract": cited_pm.get("abstract"),
            "journal": cited_pm.get("journal_title") or ref.get("journal"),
            "year": cited_pm.get("pub_year") or ref.get("year"),
            "mesh": cited_pm.get("mesh_descriptors"),
            "pub_types": cited_pm.get("pub_types"),
            "icite": {
                "citation_count":          cited_ic.get("citation_count"),
                "relative_citation_ratio": cited_ic.get("relative_citation_ratio"),
                "nih_percentile":          cited_ic.get("nih_percentile"),
                "is_clinical":             cited_ic.get("is_clinical"),
                "is_research_article":     cited_ic.get("is_research_article"),
            } if cited_ic else None,
        },
    }


def process_file(fp: str, out_dir: str) -> tuple[str, int, int, int]:
    """Per-worker. Returns (filename, n_articles_seen, n_emitted, n_no_abstract)."""
    base = Path(fp).name.replace(".jsonl.gz", ".enriched.jsonl")
    out_fp = Path(out_dir) / base
    if out_fp.exists():  # resumable: skip already-done files
        return Path(fp).name, 0, sum(1 for _ in open(out_fp)), 0
    n_arts = n_emit = n_no_abs = 0
    tmp = str(out_fp) + ".tmp"
    with gzip.open(fp, "rt", encoding="utf-8") as fin, open(tmp, "w") as fout:
        for line in fin:
            if not line:
                continue
            try:
                art = json.loads(line)
            except Exception:
                continue
            n_arts += 1
            citing_pm_id = art.get("pmid")
            if citing_pm_id is None or citing_pm_id not in _CLINICAL:
                continue
            ref_idx = {r["ref_id"]: r for r in (art.get("references") or [])
                       if r.get("ref_id") and r.get("pmid")}
            citing_pm = _PM.get(citing_pm_id)
            for cit in (art.get("citations") or []):
                for rid in cit.get("ref_ids", []):
                    ref = ref_idx.get(rid)
                    if not ref:
                        continue
                    cited_pm_id = ref["pmid"]
                    if cited_pm_id not in _CLINICAL:
                        continue
                    cited_pm = _PM.get(cited_pm_id)
                    if not cited_pm or not (cited_pm.get("abstract") or "").strip():
                        n_no_abs += 1
                        continue
                    cited_ic = _IC.get(cited_pm_id)
                    rec = build_enriched(art, cit, ref, citing_pm, cited_pm, cited_ic)
                    fout.write(json.dumps(rec, ensure_ascii=False))
                    fout.write("\n")
                    n_emit += 1
    os.replace(tmp, out_fp)
    return Path(fp).name, n_arts, n_emit, n_no_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "fragments").mkdir(parents=True, exist_ok=True)

    # Make sure fork is used (so workers share the loaded lookups).
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    global _CLINICAL, _PM, _IC
    _CLINICAL = load_clinical_pmids()
    _PM = load_pubmed_for(_CLINICAL)
    _IC = load_icite_for(_CLINICAL)

    files = []
    for sub in ("oa_comm", "oa_noncomm", "oa_other"):
        files.extend(sorted(glob(f"{CIT_ROOT}/{sub}/*.jsonl.gz")))
    print(f"[main] {len(files)} JSONL files to process", flush=True)

    t0 = time.time()
    done = tot_arts = tot_emit = tot_noabs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_file, fp, str(out_dir / "fragments")): fp
                for fp in files}
        for fut in as_completed(futs):
            name, na, ne, nx = fut.result()
            done += 1
            tot_arts += na
            tot_emit += ne
            tot_noabs += nx
            if done % 20 == 0 or done == len(files):
                el = time.time() - t0
                print(f"[main] {done}/{len(files)} files  arts={tot_arts:,}  "
                      f"emitted={tot_emit:,}  no_abs={tot_noabs:,}  "
                      f"elapsed={el:.0f}s", flush=True)

    print(f"[main] fragment pass done. emitted={tot_emit:,}", flush=True)

    # Concatenate fragments to a single big file (so we can resume GLM cleanly).
    big = out_dir / "enriched_clinical_full.jsonl"
    print(f"[main] concatenating fragments -> {big}", flush=True)
    n = 0
    with open(big, "w") as out_f:
        for fp in sorted((out_dir / "fragments").glob("*.enriched.jsonl")):
            with open(fp) as f:
                for line in f:
                    if line.strip():
                        out_f.write(line)
                        n += 1
    print(f"[main] wrote {n:,} records to {big}  total elapsed={time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
