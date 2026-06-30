#!/usr/bin/env python3
"""GLM extractor v0.3 — minimal-diff successor to v0.2 fixing the two
schema-fail patterns observed at 1.58 M production scale (234 fails total):

  * 195/234 — cited_study_design: LLM emitted 'observational' / 'observational-study'
    / 'cohort' / 'multicenter-*' for cited papers whose abstracts didn't disambiguate
    prospective vs retrospective ascertainment. v0.2 enum had no slot for this.
    -> v0.3 adds 'observational' as a residual observational bucket and prompt
       clarifies it is NOT to be used when prospective/retrospective is clear.

  * 39/234 — evidence_type: LLM emitted 'result-support' / 'contrast' (which are
    citation_purpose values) into evidence_type. Slot confusion.
    -> v0.3 prompt adds an explicit WHAT-vs-WHY contrast block so the two enums
       cannot be mistaken for each other.

Same input format as v0.2; separate output/cache so v0.2 vs v0.3 are comparable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Literal, Optional

# Local helper module (sibling file): a thin async vLLM/OpenAI client wrapper.
from llm_client import LLMClient, make_vllm_config  # noqa: E402

from pydantic import BaseModel, Field, field_validator  # noqa: E402

PROMPT_VERSION = "v0.3"

DEFAULT_IN = "./data/interim/pilot/enriched_clinical.jsonl"
DEFAULT_OUT = "./data/interim/pilot/pilot_qa_clinical_v03.jsonl"
DEFAULT_CACHE = "./data/interim/pilot/glm_cache_v03.sqlite"

# ---------------------------------------------------------------------------
# Schema

CitationPurpose = Literal[
    "background", "method", "result-support",
    "comparison", "contrast", "negative", "other",
]
EvidenceType = Literal[
    "direct-quote", "paraphrase", "statistic",
    "method-citation", "generic-background", "other",
]
SupportStrength = Literal["strong", "moderate", "weak", "insufficient-context"]
GroundingSource = Literal["evidence_sentence", "cited_abstract", "both"]
# v0.3: added 'observational' as residual observational bucket
StudyDesign = Literal[
    "rct", "non-randomized-trial",
    "prospective-cohort", "retrospective-cohort",
    "case-control", "cross-sectional",
    "observational",
    "case-series", "case-report",
    "systematic-review", "meta-analysis",
    "guideline", "preclinical", "other",
]


class StatisticExtraction(BaseModel):
    metric: str = Field(
        description="Name of the statistic, e.g. 'median LOS', 'OR for stroke', "
                    "'sensitivity', 'in-hospital mortality'."
    )
    value: str = Field(
        description="Raw value as written, including units. e.g. '2.53 days', "
                    "'93 kcal/kg', '0.42'."
    )
    ci_or_range: str = Field(
        default="",
        description="Confidence interval or range, e.g. '95% CI 1.78-2.98' or "
                    "'range 43-142'. Empty string if none reported."
    )
    p_value: str = Field(
        default="",
        description="p-value as written, e.g. '0.04', '<0.001'. Empty if not reported."
    )
    n_subjects: Optional[int] = Field(
        default=None,
        description="Sample size if explicitly mentioned, otherwise null."
    )


class CitationQA(BaseModel):
    """v0.3 — clinical-grade citation-evidence Q&A record.

    Differs from v0.2 only in:
      - StudyDesign adds 'observational' (residual observational bucket)
      - field/prompt wording clarifies it is a fallback, not a default
      - prompt adds WHAT-vs-WHY contrast for evidence_type vs citation_purpose
    """

    relevance_check: Literal["match", "ambiguous", "mismatch"] = Field(
        description=(
            "Does the cited paper's title/abstract topically match what the "
            "evidence_sentence is citing it for? 'match'=clearly the right ref; "
            "'ambiguous'=hard to tell; 'mismatch'=cited paper is unrelated to the "
            "claim. If 'mismatch', set support_strength='insufficient-context', "
            "leave question/answer/evidence_quote_from_cited empty, and explain "
            "in mismatch_reason."
        )
    )
    mismatch_reason: str = Field(
        default="",
        description=(
            "If relevance_check='mismatch' or 'ambiguous', a 1-sentence "
            "explanation of WHY the cited paper does not (or may not) ground the "
            "claim — e.g., 'cited paper is about X, but citing context discusses Y'. "
            "Empty string when relevance_check='match'."
        )
    )

    attributed_claim: str = Field(
        description=(
            "What does paper A claim, by virtue of citing paper B at this sentence? "
            "This is the proposition A presents as supported by B — NOT a summary "
            "of B itself. Be specific. 1-3 sentences. Empty when mismatch."
        )
    )
    citation_purpose: CitationPurpose = Field(
        description=(
            "WHY paper A cites paper B. "
            "background=general motivating fact; method=A is using B's technique; "
            "result-support=B's finding directly supports A's claim/result; "
            "comparison=A compares to B; contrast=A's findings differ from B's; "
            "negative=A claims B is wrong/limited."
        )
    )
    evidence_type: EvidenceType = Field(
        description=(
            "HOW the cited content is presented. "
            "direct-quote=A quotes B verbatim; "
            "paraphrase=A summarizes B's claim; statistic=A cites a number from B "
            "(if so, fill statistic_extraction); method-citation=A invokes B's "
            "protocol/instrument; generic-background=B is one of several refs "
            "supporting a generic statement. "
            "NEVER use a citation_purpose value (background/result-support/"
            "comparison/contrast/negative) here — those describe WHY, not HOW."
        )
    )
    support_strength: SupportStrength = Field(
        description=(
            "How well does B's abstract actually support A's attributed_claim? "
            "strong=abstract clearly states the claim; moderate=consistent but "
            "not the main result; weak=tangential; insufficient-context=cannot "
            "tell from inputs alone (use whenever relevance_check != 'match')."
        )
    )

    cited_study_design: StudyDesign = Field(
        description=(
            "The study design of the CITED paper, inferred from its abstract. "
            "Prefer the most specific applicable label: 'rct' / 'non-randomized-trial' "
            "/ 'prospective-cohort' / 'retrospective-cohort' / 'case-control' / "
            "'cross-sectional' / 'case-series' / 'case-report' / 'systematic-review' "
            "/ 'meta-analysis' / 'guideline' / 'preclinical'. "
            "Use 'observational' ONLY when the cited paper is clearly an observational "
            "design (not RCT, not preclinical) AND the abstract does not provide "
            "enough information to disambiguate prospective-cohort vs "
            "retrospective-cohort vs case-control vs cross-sectional. "
            "If the abstract clearly indicates prospective or retrospective "
            "ascertainment, USE THE SPECIFIC LABEL — do not default to "
            "'observational'. Use 'preclinical' for animal/in-vitro/translational "
            "mechanism studies. Use 'other' only if truly unidentifiable."
        )
    )

    question: str = Field(
        description=(
            "An open-ended biomedical question that A's citing context implicitly "
            "answers using B as evidence. NOT yes/no. Empty when mismatch."
        )
    )
    answer: str = Field(
        description=(
            "2-4 sentence factual answer grounded in evidence_sentence and/or "
            "cited abstract. Cite as [PMID:<pmid>] when stating B's contribution. "
            "Empty when mismatch."
        )
    )
    answer_grounded_in: list[GroundingSource] = Field(
        default_factory=list,
        description="Which input(s) the answer is grounded in. Empty when mismatch.",
    )

    evidence_quote_from_cited: str = Field(
        default="",
        description=(
            "VERBATIM span copied from the CITED paper's abstract that grounds the "
            "answer. MUST be an exact substring of the abstract (we will verify). "
            "Pick the single most relevant sentence (or contiguous fragment up to "
            "~50 words). Empty string if relevance_check != 'match' or if "
            "evidence_type='generic-background' (no specific quote)."
        )
    )

    statistic_extraction: Optional[StatisticExtraction] = Field(
        default=None,
        description=(
            "Required when evidence_type='statistic'. Otherwise null. Extract "
            "only what is actually stated in evidence_sentence or cited abstract; "
            "do not infer."
        )
    )

    @field_validator("attributed_claim", "question", "answer",
                     "evidence_quote_from_cited", "mismatch_reason")
    @classmethod
    def not_too_long(cls, v: str) -> str:
        if len(v) > 1500:
            raise ValueError("field too long")
        return v


# ---------------------------------------------------------------------------
# Prompt

EXAMPLE_OUTPUT = {
    "relevance_check": "match",
    "mismatch_reason": "",
    "attributed_claim": "Standardized caloric requirements for infants with congenital heart disease have not been established, and prior reports show actual enteral intake often falls short of postoperative goals.",
    "citation_purpose": "background",
    "evidence_type": "statistic",
    "support_strength": "strong",
    "cited_study_design": "retrospective-cohort",
    "question": "Why did the authors adopt the Indian RDA caloric target instead of a CHD-specific standard for postoperative infants?",
    "answer": "Because no consensus standard for caloric intake exists for infants with congenital heart disease (CHD); a retrospective cohort of 100 neonates after cardiac surgery [PMID:19116405] found median enteral intake of 93 kcal/kg, with the 100 kcal/kg goal met for only 48.4% of patient-days, indicating that enteral feeding alone is often suboptimal in this population. The authors therefore defaulted to the 2010 Indian RDA target of 92 kcal/kg/day.",
    "answer_grounded_in": ["both"],
    "evidence_quote_from_cited": "The median caloric intake per day was 93 kcal/kg (range, 43-142). A goal of 100 kcal/kg was achieved for 48.4% of patient days",
    "statistic_extraction": {
        "metric": "median caloric intake per day",
        "value": "93 kcal/kg",
        "ci_or_range": "range 43-142",
        "p_value": "",
        "n_subjects": 100
    }
}

EXAMPLE_MISMATCH = {
    "relevance_check": "mismatch",
    "mismatch_reason": "Cited paper reports global plastic production and waste statistics, but the citing context discusses interfacial bonding mechanics between microplastics and cement hydration products — topically unrelated.",
    "attributed_claim": "",
    "citation_purpose": "background",
    "evidence_type": "generic-background",
    "support_strength": "insufficient-context",
    "cited_study_design": "other",
    "question": "",
    "answer": "",
    "answer_grounded_in": [],
    "evidence_quote_from_cited": "",
    "statistic_extraction": None
}


def _schema_str() -> str:
    return json.dumps(CitationQA.model_json_schema(), ensure_ascii=False, indent=2)


SYSTEM = """You are a careful biomedical/clinical research assistant. You read ONE \
citation context from paper A (the citing paper) where it cites paper B, and you \
output a strict JSON record describing what A is asserting by citing B.

Critical distinctions:
- 'attributed_claim' is what paper A asserts BY CITING B in this specific sentence. \
It is NOT a summary of paper B itself.
- The cited paper's abstract is given so you can SANITY-CHECK plausibility. If B's \
abstract is unrelated to the citing context, set relevance_check='mismatch', \
support_strength='insufficient-context', explain in mismatch_reason, and write empty \
strings for attributed_claim / question / answer / evidence_quote_from_cited.
- For multi-citation sentences (n_co_cited_refs > 1), focus only on what B specifically \
supports; do not attribute the other refs' contributions to B.

WHAT vs WHY — do not confuse these two enums:
- 'citation_purpose' is WHY A cites B. Allowed values: \
background / method / result-support / comparison / contrast / negative / other.
- 'evidence_type'    is HOW the citation is presented. Allowed values: \
direct-quote / paraphrase / statistic / method-citation / generic-background / other.
NEVER use a citation_purpose value (e.g. 'result-support', 'contrast', 'background') \
inside evidence_type. They are different slots.

Hard requirements for v0.3:
1. evidence_quote_from_cited MUST be a verbatim substring of the cited abstract (we \
will programmatically verify). Pick the single most relevant sentence or fragment \
(up to ~50 words). When evidence_type='generic-background' or relevance_check is not \
'match', leave it empty.
2. cited_study_design reflects the design of paper B inferred from its abstract. \
Prefer the most specific label: rct / non-randomized-trial / prospective-cohort / \
retrospective-cohort / case-control / cross-sectional / case-series / case-report / \
systematic-review / meta-analysis / guideline / preclinical. Use 'observational' \
ONLY as a residual bucket when the abstract makes it clear the study is observational \
but does NOT specify prospective vs retrospective ascertainment AND the design is \
not clearly cohort/case-control/cross-sectional. If the abstract clearly says \
'prospective cohort' or 'retrospective review', USE THE SPECIFIC LABEL — do not \
default to 'observational'. Use 'preclinical' for animal/in-vitro/translational \
mechanism papers. Use 'other' only if truly unidentifiable.
3. When evidence_type='statistic', fill statistic_extraction with the metric, value, \
CI/range, p-value, and n_subjects exactly as written. Do not infer or compute. \
Otherwise statistic_extraction must be null.

Output requirements:
- Return ONE JSON object — no prose, no code fences, no extra fields.
- Field names and enum values MUST match the schema exactly.
- Cite the cited paper as [PMID:<pmid>] inside the answer.
- Do not invent facts that are not in evidence_sentence or cited abstract.

Schema (output must validate against this):
""" + _schema_str() + """

Example #1 — well-formed MATCH (template only, do NOT copy contents):
""" + json.dumps(EXAMPLE_OUTPUT, ensure_ascii=False, indent=2) + """

Example #2 — well-formed MISMATCH (template only):
""" + json.dumps(EXAMPLE_MISMATCH, ensure_ascii=False, indent=2)


def build_user_prompt(rec: dict) -> str:
    citing = rec["citing"]
    ctx = rec["context"]
    cited = rec["cited"]
    icite = cited.get("icite") or {}
    abs_text = (cited.get("abstract") or "")[:3500]
    mesh_cited = (cited.get("mesh") or [])[:15]

    parts = [
        "## CITING PAPER (A)",
        f"PMCID: {citing.get('pmcid')}  PMID: {citing.get('pmid')}",
        f"Title: {citing.get('title') or ''}",
        f"Journal: {citing.get('journal') or ''}  Year: {citing.get('year') or ''}",
        "",
        "## CITATION CONTEXT (where A cites B)",
        f"Section: {ctx.get('section') or ''}  ({ctx.get('section_bucket')})",
        f"Co-cited refs in this sentence: {ctx.get('n_co_cited_refs')}",
        "Evidence sentence:",
        f"  {ctx.get('evidence_sentence') or ''}",
        "",
        "## CITED PAPER (B)",
        f"PMID: {cited.get('pmid')}  DOI: {cited.get('doi') or ''}",
        f"Title: {cited.get('title') or ''}",
        f"Journal: {cited.get('journal') or ''}  Year: {cited.get('year') or ''}",
    ]
    pub_types = cited.get("pub_types") or []
    if pub_types:
        parts.append(f"Publication types: {', '.join(pub_types[:8])}")
    if icite:
        parts.append(
            f"Impact (iCite): citation_count={icite.get('citation_count')}, "
            f"RCR={icite.get('relative_citation_ratio')}, "
            f"is_clinical={icite.get('is_clinical')}"
        )
    if mesh_cited:
        parts.append(f"MeSH (cited): {', '.join(mesh_cited)}")
    parts.append("")
    parts.append("Abstract (cited):")
    parts.append(abs_text or "[no abstract available]")
    parts.append("")
    parts.append("## TASK")
    parts.append(
        "Extract the structured citation-evidence record per the v0.3 CitationQA "
        "schema. Output a single JSON object. Remember: evidence_quote_from_cited "
        "must be a verbatim substring of the abstract above."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Runner

async def process_record(client: LLMClient, rec: dict) -> dict:
    user = build_user_prompt(rec)
    res = await client.chat_json(SYSTEM, user)
    out = {
        "input": rec,
        "ok": res.ok,
        "raw_text": res.raw_text,
        "usage_in": res.usage_in,
        "usage_out": res.usage_out,
        "latency_s": round(res.latency_s, 3),
        "attempts": res.attempts,
        "generator": {
            "model": "glm-5.1-fp8",
            "prompt_version": PROMPT_VERSION,
            "ts": time.time(),
        },
    }
    if res.ok:
        out["extracted"] = res.data
    else:
        out["error_kind"] = res.error_kind
        out["error"] = str(res.data)[:500]
    return out


def _record_key(rec_in: dict) -> str:
    """Stable key for resuming. Identical to v0.2 keying."""
    citing = rec_in["citing"]; ctx = rec_in["context"]; cited = rec_in["cited"]
    return f"{citing.get('pmcid')}|{cited.get('pmid')}|{ctx.get('paragraph_idx')}|{(ctx.get('evidence_sentence') or '')[:80]}"


def _load_done_keys(path: str) -> set[str]:
    """Read existing output (if any) and return the set of input keys already
    processed, so we can skip them on resume."""
    done = set()
    if not Path(path).exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(_record_key(rec["input"]))
            except Exception:
                continue
    return done


async def main_async(args):
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    if len(urls) > 1:
        cfg = make_vllm_config(model=args.model, base_url=urls[0], base_urls=urls)
        print(f"[glm v03] LB across {len(urls)} endpoints: {urls}", flush=True)
    else:
        cfg = make_vllm_config(model=args.model, base_url=urls[0])
    client = LLMClient(
        cfg, concurrency=args.concurrency, cache_db=args.cache,
        validate_with=CitationQA, max_retry=3,
    )

    in_records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                in_records.append(json.loads(line))
    if args.limit:
        in_records = in_records[: args.limit]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    done_keys = _load_done_keys(args.output) if args.resume else set()
    if done_keys:
        before = len(in_records)
        in_records = [r for r in in_records if _record_key(r) not in done_keys]
        print(f"[glm v03] resume: skipping {before - len(in_records)} already-done; "
              f"{len(in_records)} remain", flush=True)

    print(f"[glm v03] {len(in_records)} input records to process, "
          f"conc={args.concurrency}", flush=True)
    if not in_records:
        print("[glm v03] nothing to do.", flush=True)
        return 0

    t0 = time.time()
    n_done = n_ok = n_fail = 0
    sem = asyncio.Semaphore(args.concurrency)

    async def one(rec):
        async with sem:
            return await process_record(client, rec)

    open_mode = "a" if args.resume else "w"
    with open(args.output, open_mode) as out_f:
        tasks = [asyncio.create_task(one(r)) for r in in_records]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            out_f.write(json.dumps(res, ensure_ascii=False))
            out_f.write("\n")
            n_done += 1
            if res["ok"]:
                n_ok += 1
            else:
                n_fail += 1
            if n_done % 100 == 0:
                out_f.flush()
            if n_done % args.report_every == 0 or n_done == len(in_records):
                el = time.time() - t0
                stat = client.stats
                rate = n_done / max(el, 1)
                eta_s = (len(in_records) - n_done) / max(rate, 0.001)
                print(f"[glm v03] {n_done}/{len(in_records)}  ok={n_ok} fail={n_fail}  "
                      f"elapsed={el:.0f}s rate={rate:.2f}/s eta={eta_s/3600:.1f}h  "
                      f"in={stat['input_tokens']:,} out={stat['output_tokens']:,}  "
                      f"fail_kinds={dict(stat['fail_kinds'])}",
                      flush=True)
    print(f"[glm v03] DONE ok={n_ok} fail={n_fail} elapsed={time.time()-t0:.0f}s",
          flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--model", default="glm-5.1-fp8")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="single URL or comma-separated list for round-robin LB")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="append-mode; skip records already present in output")
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
