#!/usr/bin/env python3
"""Dedupe the v0.3 production output by canonical resume-key, keeping the
first-attempt-success record per key.

Why dedupe:
  Empirical inspection shows that duplicate-key clusters (28-, 27-, 26-count)
  consist of records with **near-identical input prompts** but **different
  LLM outputs** (one unique full evidence_sentence, but 28 different
  attributed_claims). For SFT this is harmful — the model is asked to fit
  multiple inconsistent outputs to the same input. Dedupe collapses each key
  cluster to one canonical record.

Selection rule per cluster (in priority order):
  1. attempts == 1 and ok                    (cleanest)
  2. attempts == 2 and ok
  3. attempts == 3 and ok
  4. ok regardless of attempts
  5. lowest line index (any record)

The 1 schema-fail record is preserved (different key from any ok record
naturally; dedupe doesn't touch it).

Output: data/interim/pilot/full_qa_clinical_v03_unique.jsonl

For paper:
  - "1,582,658 xref citation events"  (raw production count)
  - "1,524,426 unique canonical citations after dedupe"  (training corpus)

The original full_qa_clinical_v03.jsonl is preserved (DESIGN.md frozen-output policy).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT  = "./data/interim/pilot/full_qa_clinical_v03.jsonl"
DEFAULT_OUTPUT = "./data/interim/pilot/full_qa_clinical_v03_unique.jsonl"


def key(rec_input: dict) -> str:
    c = rec_input["citing"]; ctx = rec_input["context"]; cd = rec_input["cited"]
    return f"{c.get('pmcid')}|{cd.get('pmid')}|{ctx.get('paragraph_idx')}|{(ctx.get('evidence_sentence') or '')[:80]}"


def priority_score(r: dict) -> tuple:
    """Lower is better."""
    ok = r.get("ok", False)
    attempts = r.get("attempts") or 99
    return (
        0 if (ok and attempts == 1) else
        1 if (ok and attempts == 2) else
        2 if (ok and attempts == 3) else
        3 if ok else
        4
    ), attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    # Pass 1: scan all records, for each key remember the best (lowest score) line_idx
    print("[dedupe] pass 1: scanning + scoring...", flush=True)
    best: dict = {}     # key -> (score_tuple, line_idx)
    n_total = 0
    with open(args.input) as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            n_total += 1
            r = json.loads(line)
            k = key(r["input"])
            score = priority_score(r)
            if k not in best or score < best[k][0]:
                best[k] = (score, line_idx)
    print(f"[dedupe] total records:  {n_total:,}", flush=True)
    print(f"[dedupe] unique keys:    {len(best):,}", flush=True)
    print(f"[dedupe] dropped (dup):  {n_total - len(best):,}  "
          f"({100*(n_total-len(best))/n_total:.2f} %)", flush=True)

    # Pass 2: write only the chosen line indices
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    keep_set = {li for _, li in best.values()}
    print("[dedupe] pass 2: writing deduped output...", flush=True)
    n_written = n_ok = n_fail = 0
    attempts_dist = Counter()
    relev_dist = Counter()
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line_idx, line in enumerate(fin):
            if line_idx not in keep_set:
                continue
            line = line.rstrip()
            if not line:
                continue
            fout.write(line + "\n")
            n_written += 1
            r = json.loads(line)
            attempts_dist[r.get("attempts")] += 1
            if r.get("ok"):
                n_ok += 1
                relev_dist[r["extracted"].get("relevance_check")] += 1
            else:
                n_fail += 1

    print(f"[dedupe] wrote {n_written:,} → {args.output}")
    print(f"[dedupe]   ok={n_ok:,}  fail={n_fail}")
    print(f"[dedupe]   attempts: {dict(attempts_dist)}")
    print(f"[dedupe]   relevance: {dict(relev_dist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
