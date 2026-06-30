#!/usr/bin/env python3
"""Stratified sampler for the v0.3 gold validation set (~1,000 records).

Per docs/VALIDATION.md, produces 200 records each across 5 main strata:
  match × strong / match × moderate / match × weak / ambiguous / mismatch

Within each stratum, samples are spread across (study_design × clinical_specialty)
cells using largest-remainder proportional allocation, so the gold set captures
the design pyramid AND the dataset's specialty mix.

Output schema = original record + a `gold_meta` block:
  gold_meta = {
    record_id        : 16-char sha256 prefix of the stable record key
    stratum          : "match_strong" | "match_moderate" | "match_weak"
                       | "ambiguous"   | "mismatch"
    design_bucket    : grouped study_design (rct / non-rand-trial /
                       cohort-observational / case-control / cross-sectional /
                       guideline / review / case / preclinical / other)
    specialty_bucket : top-N MeSH heuristic on cited paper
                       (Neoplasms / Cardiovascular / Infectious / Endocrine /
                        Pediatric / Mental / Nervous / Respiratory / Other)
    seed             : RNG seed for reproducibility
    input_version    : the GLM prompt version used to produce extracted (v0.3)
  }

The 1 schema-fail residual record is excluded (only ok=True records sampled).

Usage:
  python sample_gold_1000.py \
    --input  .../full_qa_clinical_v03.jsonl \
    --output .../data/processed/gold_validation_unannotated.jsonl \
    --seed 20260501

Determinism: same input + same seed → bit-identical output (records sorted by
record_id within each stratum).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults

DEFAULT_INPUT = "./data/interim/pilot/full_qa_clinical_v03.jsonl"
DEFAULT_OUTPUT = "./data/processed/gold_validation_unannotated.jsonl"
DEFAULT_SEED = 20260501
DEFAULT_PER_STRATUM = 200

# ---------------------------------------------------------------------------
# Bucketing

# Specialty patterns are matched in order; first match wins.
# Patterns are case-insensitive and substring-style on MeSH descriptor text.
SPECIALTY_PATTERNS = [
    ("Neoplasms",      r"\b(neoplasm|cancer|carcinoma|tumor|lymphoma|leukemia|sarcoma|melanoma|adenocarcinoma|metastas)"),
    ("Cardiovascular", r"\b(cardiovascular|heart|cardiac|coronary|myocardial|stroke|hypertension|atrial|vascular|aort)"),
    ("Infectious",     r"\b(infection|bacteria|viral|hiv|tuberculosis|pneumonia|sepsis|hepatitis|covid|influenza|malaria|antimicrobial)"),
    ("Mental",         r"\b(mental|depression|anxiety|schizophren|psychiatric|bipolar|ptsd|suicide)"),
    ("Endocrine",      r"\b(diabetes|thyroid|endocrine|insulin|obesity|metabolic syndrome)"),
    ("Respiratory",    r"\b(respiratory|lung|pulmonary|asthma|bronchitis|copd)"),
    ("Nervous",        r"\b(nervous system|brain|neurol|cognitive|alzheimer|parkinson|epilepsy|multiple sclerosis|dementia)"),
    ("Pediatric",      r"\b(infant|pediatric|child|newborn|neonate)"),
]
SPECIALTY_RES = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in SPECIALTY_PATTERNS]


def specialty_bucket(mesh_list) -> str:
    if not mesh_list:
        return "Other"
    txt = " | ".join(mesh_list)
    for name, rx in SPECIALTY_RES:
        if rx.search(txt):
            return name
    return "Other"


DESIGN_GROUP = {
    "rct": "rct",
    "non-randomized-trial": "non-rand-trial",
    "prospective-cohort": "cohort-observational",
    "retrospective-cohort": "cohort-observational",
    "observational": "cohort-observational",
    "case-control": "case-control",
    "cross-sectional": "cross-sectional",
    "guideline": "guideline",
    "systematic-review": "review",
    "meta-analysis": "review",
    "case-series": "case",
    "case-report": "case",
    "preclinical": "preclinical",
    "other": "other",
}
def design_bucket(d: str) -> str:
    return DESIGN_GROUP.get(d, "other")


def main_stratum(extracted: dict) -> str | None:
    rc = extracted.get("relevance_check")
    if rc == "mismatch":
        return "mismatch"
    if rc == "ambiguous":
        return "ambiguous"
    if rc == "match":
        s = extracted.get("support_strength")
        if s in ("strong", "moderate", "weak"):
            return f"match_{s}"
        # 'insufficient-context' on a 'match' record is a spec-edge-case;
        # collapse into 'ambiguous' for sampling purposes.
        return "ambiguous"
    return None


def stable_record_id(inp: dict) -> str:
    """Same key shape as the resume key, hashed for compactness."""
    c = inp["citing"]; ctx = inp["context"]; cd = inp["cited"]
    raw = f"{c.get('pmcid')}|{cd.get('pmid')}|{ctx.get('paragraph_idx')}|{(ctx.get('evidence_sentence') or '')[:80]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Largest-remainder allocation

def allocate(cell_pops: dict, target: int) -> dict:
    """Distribute `target` slots across cells proportionally to cell_pops,
    using largest-remainder method, capped at cell_pops[k].

    cell_pops: {cell_key: population}
    Returns: {cell_key: n_to_sample}
    """
    if not cell_pops:
        return {}
    total_pop = sum(cell_pops.values())
    if total_pop <= target:
        # Take all available; cannot fill target
        return dict(cell_pops)

    # Stage 1: floor allocation
    raw = {k: target * v / total_pop for k, v in cell_pops.items()}
    floor_alloc = {k: min(int(math.floor(r)), cell_pops[k]) for k, r in raw.items()}
    used = sum(floor_alloc.values())
    remain = target - used

    # Stage 2: distribute remainder to cells with largest fractional part
    # AND that haven't been capped at population.
    by_frac = sorted(
        cell_pops.keys(),
        key=lambda k: (raw[k] - math.floor(raw[k])),
        reverse=True,
    )
    i = 0
    while remain > 0 and i < len(by_frac) * 4:  # bounded loop
        for k in by_frac:
            if remain <= 0:
                break
            if floor_alloc[k] < cell_pops[k]:
                floor_alloc[k] += 1
                remain -= 1
        i += 1
        if all(floor_alloc[k] >= cell_pops[k] for k in cell_pops):
            break  # everything capped — cannot fill further

    return floor_alloc


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"[gold-sampler] input  = {args.input}", flush=True)
    print(f"[gold-sampler] output = {args.output}", flush=True)
    print(f"[gold-sampler] per-stratum = {args.per_stratum}, seed = {args.seed}",
          flush=True)

    # Pass 1: bucket every ok record into (stratum, design_bucket, specialty)
    # We keep just the line index per cell — the file is too big to hold whole
    # records in memory.
    print("[gold-sampler] pass 1: scanning + bucketing...", flush=True)
    cell_lines: dict = defaultdict(list)   # (stratum, design, specialty) -> [line_idx, ...]
    n_total = n_ok = n_no_stratum = 0
    with open(args.input) as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            n_total += 1
            r = json.loads(line)
            if not r.get("ok"):
                continue
            n_ok += 1
            e = r["extracted"]
            ms = main_stratum(e)
            if ms is None:
                n_no_stratum += 1
                continue
            cited = (r.get("input") or {}).get("cited") or {}
            sp = specialty_bucket(cited.get("mesh"))
            db = design_bucket(e.get("cited_study_design"))
            cell_lines[(ms, db, sp)].append(line_idx)
    print(f"[gold-sampler] scanned: {n_total:,}, ok: {n_ok:,}, "
          f"unbucketed: {n_no_stratum}", flush=True)

    # Per-stratum stats
    strata_pop = Counter()
    cells_per_stratum: dict = defaultdict(dict)  # stratum -> {(design, specialty): pop}
    for (st, db, sp), lines in cell_lines.items():
        strata_pop[st] += len(lines)
        cells_per_stratum[st][(db, sp)] = len(lines)
    print(f"[gold-sampler] strata populations: {dict(strata_pop)}",
          flush=True)

    # Allocate per-cell sample counts
    print("[gold-sampler] allocating per-cell sample counts (largest-remainder)...",
          flush=True)
    plan: dict = {}                      # (stratum, design, specialty) -> n
    actual_per_stratum = Counter()
    for st in ("match_strong", "match_moderate", "match_weak", "ambiguous", "mismatch"):
        cell_pops = cells_per_stratum.get(st, {})
        alloc = allocate(cell_pops, args.per_stratum)
        for (db, sp), n in alloc.items():
            if n > 0:
                plan[(st, db, sp)] = n
                actual_per_stratum[st] += n
    print(f"[gold-sampler] target per stratum = {args.per_stratum}", flush=True)
    print(f"[gold-sampler] actual per stratum = {dict(actual_per_stratum)}",
          flush=True)

    # Pick line indices per cell using seeded RNG
    print("[gold-sampler] picking line indices...", flush=True)
    selected: dict = {}                  # line_idx -> (stratum, design, specialty)
    for cell_key, n in plan.items():
        lines = cell_lines[cell_key]
        chosen = rng.sample(lines, min(n, len(lines)))
        for li in chosen:
            selected[li] = cell_key
    print(f"[gold-sampler] total selected lines: {len(selected):,}", flush=True)

    # Pass 2: stream input again, emit selected records with gold_meta
    print("[gold-sampler] pass 2: writing output...", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_records = []
    with open(args.input) as f:
        for line_idx, line in enumerate(f):
            if line_idx not in selected:
                continue
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (st, db, sp) = selected[line_idx]
            r["gold_meta"] = {
                "record_id": stable_record_id(r["input"]),
                "stratum": st,
                "design_bucket": db,
                "specialty_bucket": sp,
                "seed": args.seed,
                "input_version": (r.get("generator") or {}).get("prompt_version", ""),
            }
            out_records.append(r)

    # Sort by (stratum, record_id) for stable, reproducible output ordering
    stratum_order = {
        "match_strong": 0, "match_moderate": 1, "match_weak": 2,
        "ambiguous": 3, "mismatch": 4,
    }
    out_records.sort(key=lambda r: (
        stratum_order.get(r["gold_meta"]["stratum"], 99),
        r["gold_meta"]["record_id"],
    ))

    with open(args.output, "w") as out_f:
        for r in out_records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gold-sampler] wrote {len(out_records):,} records → {args.output}",
          flush=True)

    # Final summary
    out_strat = Counter(r["gold_meta"]["stratum"] for r in out_records)
    out_design = Counter(r["gold_meta"]["design_bucket"] for r in out_records)
    out_spec = Counter(r["gold_meta"]["specialty_bucket"] for r in out_records)
    print()
    print("=== gold set summary ===")
    print(f"  records:   {len(out_records):,}")
    print(f"  strata:    {dict(out_strat.most_common())}")
    print(f"  designs:   {dict(out_design.most_common())}")
    print(f"  specialty: {dict(out_spec.most_common())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
