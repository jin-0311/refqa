#!/usr/bin/env python3
"""Deterministic train/val/test split for RefQA v0.3 production output.

Splits 1,582,657 ok records (+ 1 fail, ignored) by citing-paper PMCID hash so
that **all citation events from one paper land in the same split** — preventing
sentence-level leakage between train and test for the same publication.

Ratios:
  gold  : ~1,000 records   (those already drawn into gold_validation_unannotated.jsonl)
  train : 85 %
  val   :  5 %
  test  : 10 %

The ~1 K gold records are removed from the train/val/test pool first; the
remaining ~1,581,657 records are then bucketed by hash. Adversarial-benchmark
records (relevance_check=='mismatch') are NOT a separate split — they appear
naturally inside test (and elsewhere), so the fine-tuned model sees mismatch
examples during training and is evaluated on held-out mismatch records inside
the test split.

Output: a single TSV index file (~70 MB) plus a one-page summary printed to
stdout. Materialization of train/val/test JSONL files is a separate step
(`materialize_splits.py` if needed for fine-tune feeding).

Determinism: same input + same SEED string produces bit-identical output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults

DEFAULT_INPUT = "./data/interim/pilot/full_qa_clinical_v03.jsonl"
DEFAULT_GOLD  = "./data/processed/gold_validation_unannotated.jsonl"
DEFAULT_OUT_INDEX = "./data/processed/splits/v03_split_index.tsv"
DEFAULT_SEED = "refqa-v03-split"

# Hash thresholds out of 1000 (remaining after gold removal):
#   val   :  0 ..  49   ( 50 / 1000 = 5 %)
#   test  : 50 .. 149   (100 / 1000 = 10 %)
#   train : 150 .. 999  (850 / 1000 = 85 %)
VAL_LO, VAL_HI = 0, 50
TEST_LO, TEST_HI = 50, 150
# train = remainder

# ---------------------------------------------------------------------------
# Key functions (matching extract_qa_glm_v03._record_key for cross-script consistency)

def record_key(rec_input: dict) -> str:
    citing = rec_input["citing"]; ctx = rec_input["context"]; cited = rec_input["cited"]
    return f"{citing.get('pmcid')}|{cited.get('pmid')}|{ctx.get('paragraph_idx')}|{(ctx.get('evidence_sentence') or '')[:80]}"


def split_of(citing_pmcid: str | None, seed: str) -> str:
    """Returns 'train' | 'val' | 'test' based on hashed citing_pmcid.
    Same pmcid → same split, regardless of cited paper or sentence position."""
    if citing_pmcid is None:
        citing_pmcid = "__NULL_PMCID__"
    h = hashlib.sha256(f"{seed}|{citing_pmcid}".encode()).hexdigest()
    n = int(h[:8], 16) % 1000
    if VAL_LO <= n < VAL_HI:
        return "val"
    if TEST_LO <= n < TEST_HI:
        return "test"
    return "train"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEFAULT_INPUT)
    ap.add_argument("--gold",   default=DEFAULT_GOLD)
    ap.add_argument("--out",    default=DEFAULT_OUT_INDEX)
    ap.add_argument("--seed",   default=DEFAULT_SEED)
    args = ap.parse_args()

    # 1) Load gold record_keys (these override and become 'gold' split)
    gold_keys: set[str] = set()
    if Path(args.gold).exists():
        with open(args.gold) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                gold_keys.add(record_key(r["input"]))
        print(f"[splits] loaded {len(gold_keys):,} gold record keys from {args.gold}")
    else:
        print(f"[splits] WARNING: gold file not found at {args.gold}; "
              f"no gold protection in this draw")

    # 2) Stream input, assign splits, write TSV index
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    counts_by_relev = {s: Counter() for s in ("gold","train","val","test","fail")}
    counts_by_design = {s: Counter() for s in ("gold","train","val","test","fail")}
    counts_by_strat = {s: Counter() for s in ("gold","train","val","test","fail")}

    pmcid_seen_in_split: dict[str, str] = {}    # pmcid -> split  (consistency check)
    n_total = n_ok = n_fail = 0

    with open(args.input) as f, open(args.out, "w") as out:
        out.write("record_key\tciting_pmcid\tcited_pmid\tsplit\trelevance_check\tcited_study_design\tsupport_strength\n")
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            inp = r["input"]
            key = record_key(inp)
            citing_pmcid = inp["citing"].get("pmcid")
            cited_pmid = inp["cited"].get("pmid")

            if not r.get("ok"):
                # Fails are reported but not assigned a usable split
                split = "fail"
                n_fail += 1
                rc = ds = ss = "FAIL"
            else:
                n_ok += 1
                if key in gold_keys:
                    split = "gold"
                else:
                    split = split_of(citing_pmcid, args.seed)
                e = r["extracted"]
                rc = e.get("relevance_check") or ""
                ds = e.get("cited_study_design") or ""
                ss = e.get("support_strength") or ""

            # Sanity: same citing_pmcid should always land in same split
            # (gold can override, so we only check non-gold)
            if split not in ("gold", "fail") and citing_pmcid is not None:
                prev = pmcid_seen_in_split.get(citing_pmcid)
                if prev is None:
                    pmcid_seen_in_split[citing_pmcid] = split
                elif prev != split:
                    raise RuntimeError(
                        f"split inconsistency: pmcid={citing_pmcid} previously "
                        f"in {prev}, now {split}"
                    )

            counts[split] += 1
            counts_by_relev[split][rc] += 1
            counts_by_design[split][ds] += 1
            # support_strength stratum, conditional on relevance==match
            stratum = (
                f"match_{ss}" if rc == "match" and ss in ("strong","moderate","weak")
                else rc if rc in ("ambiguous","mismatch")
                else "other"
            )
            counts_by_strat[split][stratum] += 1

            out.write(f"{key}\t{citing_pmcid or ''}\t{cited_pmid or ''}\t"
                      f"{split}\t{rc}\t{ds}\t{ss}\n")

    print()
    print(f"=== input ===")
    print(f"  total records: {n_total:,}  ok={n_ok:,}  fail={n_fail}")
    print(f"  unique citing pmcids in non-gold pool: {len(pmcid_seen_in_split):,}")
    print()
    print(f"=== split sizes ===")
    for s in ("gold","train","val","test","fail"):
        c = counts[s]
        pct = (100.0 * c / n_total) if n_total else 0
        print(f"  {s:6s}  {c:>10,}  ({pct:.2f} %)")
    print()
    target_total = n_total - counts["fail"] - counts["gold"]
    print(f"  target total (ok minus gold) : {target_total:,}")
    print(f"  train share of target        : {100.0*counts['train']/target_total:.2f} %  (target 85 %)")
    print(f"  val   share of target        : {100.0*counts['val']  /target_total:.2f} %  (target  5 %)")
    print(f"  test  share of target        : {100.0*counts['test'] /target_total:.2f} %  (target 10 %)")

    # Distributional sanity checks: relevance / support / design distributions
    # should be near-identical across splits.
    print()
    print(f"=== relevance_check distribution per split (% of split) ===")
    relev_keys = ["match", "ambiguous", "mismatch"]
    print(f"  {'split':6s}  " + "  ".join(f"{k:>12s}" for k in relev_keys))
    for s in ("train","val","test","gold"):
        denom = counts[s] or 1
        row = "  ".join(f"{100.0*counts_by_relev[s].get(k,0)/denom:>11.2f}%" for k in relev_keys)
        print(f"  {s:6s}  {row}")

    print()
    print(f"=== adversarial pool size in each split ===")
    for s in ("train","val","test","gold"):
        n_adv = counts_by_relev[s].get("mismatch", 0)
        print(f"  {s:6s}  mismatch records: {n_adv:>8,}")

    # Stratum (5-class) breakdown — useful for evaluating model on test slices
    print()
    print(f"=== test split stratum breakdown ===")
    for k, v in counts_by_strat["test"].most_common():
        print(f"  {k:18s}  {v:>8,}")

    print()
    print(f"=== written: {args.out}")
    print(f"    sha256(seed) prefix used : {hashlib.sha256(args.seed.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
