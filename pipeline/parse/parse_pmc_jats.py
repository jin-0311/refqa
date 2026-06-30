#!/usr/bin/env python3
"""Stream PMC OA JATS XML tarballs into citation-context JSONL.

For each article we emit:
  {
    "pmcid": "PMC176545",
    "pmid": 12929205,                 # may be null
    "doi": "...",
    "title": "...",
    "journal": "Exp Parasitol",
    "year": 2003,
    "license": "...",
    "subset": "oa_comm",
    "n_refs": 67,
    "n_citations": 90,
    "references": [
      {"ref_id":"pbio-...","pmid":12427465,"doi":null,
       "title":"...","journal":"...","year":2002,"raw":"..."}
    ],
    "citations": [
      {"ref_ids":["pbio-..."], "sentence":"...",
       "section":"Introduction", "paragraph_idx":3, "in_table":false}
    ]
  }

Parallelism: one process per input tar.gz; ProcessPoolExecutor across files.
Output: one .jsonl.gz per input tar.gz (compressed). Resumable: if the output
file exists and is non-empty, the tar is skipped.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from lxml import etree

DEFAULT_PMC_ROOT = "./data/raw/pmc"
DEFAULT_OUT_ROOT = "./data/interim/citations"

# ---------------------------------------------------------------------------
# XML helpers

PLACEHOLDER_FMT = "␟XREF{}␟"   # use unit-separator-ish chars unlikely in text
PLACEHOLDER_RE = re.compile(r"␟XREF(\d+)␟")

# Sentence boundary: end of sentence punctuation followed by whitespace+capital,
# or end-of-string. We conservatively split on . ! ? followed by space + uppercase
# or digit. Tolerates abbreviations imperfectly but is good enough for citation
# context extraction (we always include the full sentence containing the marker).
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])")


def text_with_placeholders(elem, xref_to_idx: dict) -> str:
    """Render an element's text content, replacing each <xref ref-type="bibr">
    with a unique placeholder so we can locate it after sentence splitting."""
    out = []

    def walk(e):
        # element opening: emit its leading text
        if e.tag == "xref" and e.get("ref-type") == "bibr":
            idx = len(xref_to_idx)
            xref_to_idx[idx] = e
            out.append(PLACEHOLDER_FMT.format(idx))
        else:
            if e.text:
                out.append(e.text)
            for child in e:
                walk(child)
        # element closing: emit tail (text after the close tag)
        if e.tail:
            out.append(e.tail)

    walk(elem)
    return "".join(out)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def first_text(elem, xpath: str) -> str | None:
    nodes = elem.xpath(xpath)
    if not nodes:
        return None
    n = nodes[0]
    if isinstance(n, str):
        return n.strip() or None
    txt = "".join(n.itertext()).strip()
    return txt or None


def maybe_int(s):
    if s is None:
        return None
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Reference parsing

def parse_reference(ref_elem) -> dict:
    """Extract structured fields from a <ref> element. Falls back to mixed-citation
    text when element-citation isn't present."""
    rid = ref_elem.get("id") or ""
    cit = ref_elem.find("element-citation")
    mixed = ref_elem.find("mixed-citation")
    target = cit if cit is not None else mixed

    pmid = None
    doi = None
    title = None
    journal = None
    year = None

    if target is not None:
        for pid in target.findall(".//pub-id"):
            t = (pid.get("pub-id-type") or "").lower()
            v = (pid.text or "").strip()
            if not v:
                continue
            if t == "pmid" and pmid is None:
                pmid = maybe_int(v)
            elif t == "doi" and doi is None:
                doi = v
        title = first_text(target, "./article-title") \
                or first_text(target, "./chapter-title")
        journal = first_text(target, "./source")
        year = maybe_int(first_text(target, "./year"))

    raw = normalize_ws("".join(ref_elem.itertext()))
    # As a last resort, look for inline pmid/doi in mixed citations.
    if pmid is None:
        m = re.search(r"pmid[:\s]*([0-9]{4,9})", raw, re.I)
        if m:
            pmid = int(m.group(1))
    if doi is None:
        m = re.search(r"\b10\.\d{4,9}/[^\s\"<>]+", raw)
        if m:
            doi = m.group(0).rstrip(".,;)")

    return {
        "ref_id": rid,
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "year": year,
        "raw": raw[:1200],  # cap length
    }


# ---------------------------------------------------------------------------
# Article parsing

def parse_article(root) -> dict | None:
    """Return per-article record, or None if no usable content."""
    art = root if root.tag == "article" else root.find(".//article")
    if art is None:
        return None

    front = art.find("./front")
    body = art.find("./body")
    back = art.find("./back")

    # IDs
    pmcid = None
    pmid = None
    doi = None
    if front is not None:
        for aid in front.iter("article-id"):
            t = (aid.get("pub-id-type") or "").lower()
            v = (aid.text or "").strip()
            if not v:
                continue
            if t in ("pmc", "pmcid") and pmcid is None:
                pmcid = v if v.startswith("PMC") else f"PMC{v}"
            elif t == "pmid" and pmid is None:
                pmid = maybe_int(v)
            elif t == "doi" and doi is None:
                doi = v

    title = first_text(art, ".//front//title-group/article-title") \
            or first_text(art, ".//front//article-title")
    journal = first_text(art, ".//front//journal-title")
    year = (maybe_int(first_text(art, ".//front//pub-date[@pub-type='epub']/year"))
            or maybe_int(first_text(art, ".//front//pub-date[@pub-type='ppub']/year"))
            or maybe_int(first_text(art, ".//front//pub-date/year")))

    license_ = first_text(art, ".//front//license/@license-type")
    if license_ is None:
        license_ = first_text(art, ".//front//license/license-p")

    # References: id -> parsed dict
    references = []
    ref_id_index = {}
    if back is not None:
        for ref in back.iter("ref"):
            rec = parse_reference(ref)
            if not rec["ref_id"]:
                continue
            ref_id_index[rec["ref_id"]] = len(references)
            references.append(rec)

    # Citations: walk every paragraph in body (and back/notes) for xref bibr
    citations = []
    if body is not None:
        for citation in iter_citations(body, ref_id_index):
            citations.append(citation)
    # Some articles put discussion in back/notes; usually skip.

    return {
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "year": year,
        "license": license_,
        "n_refs": len(references),
        "n_citations": len(citations),
        "references": references,
        "citations": citations,
    }


def iter_citations(body, ref_id_index: dict):
    """Yield citation context dicts from every <p> in body."""
    # Walk by section so we can attribute section titles.
    # Section nesting: traverse and keep track of nearest enclosing <sec> title.
    para_idx = 0
    for p in body.iter("p"):
        # Skip <p> inside table-wrap/fig caption? include them but flag.
        in_table = any(a.tag == "table-wrap" for a in p.iterancestors())
        in_fig = any(a.tag == "fig" for a in p.iterancestors())

        # Find enclosing section title
        section = None
        for anc in p.iterancestors():
            if anc.tag == "sec":
                t = first_text(anc, "./title")
                if t:
                    section = t
                    break

        xref_map = {}
        text = text_with_placeholders(p, xref_map)
        if not xref_map:
            para_idx += 1
            continue
        text = normalize_ws(text)
        # Split into sentences. Then for each sentence, collect placeholder ids.
        sentences = SENT_SPLIT.split(text)
        for sent in sentences:
            ids = [int(m.group(1)) for m in PLACEHOLDER_RE.finditer(sent)]
            if not ids:
                continue
            # Resolve ref ids; group consecutive into a single citation event.
            rids = []
            for i in ids:
                el = xref_map.get(i)
                if el is None:
                    continue
                rid = el.get("rid")
                if not rid:
                    continue
                # rid may be space-separated multiple ids
                for r in rid.split():
                    if r in ref_id_index:
                        rids.append(r)
            if not rids:
                continue
            # Strip placeholders from sentence text for output.
            clean = PLACEHOLDER_RE.sub("", sent).strip()
            clean = re.sub(r"\s{2,}", " ", clean)
            yield {
                "ref_ids": rids,
                "sentence": clean,
                "section": section,
                "paragraph_idx": para_idx,
                "in_table": in_table,
                "in_fig": in_fig,
            }
        para_idx += 1


# ---------------------------------------------------------------------------
# Tarball driver

def process_tar(tar_path: str, out_path: str, subset: str) -> tuple[str, int, int, float]:
    """Process one tar.gz, write one .jsonl.gz, return stats."""
    t0 = time.time()
    n_articles = 0
    n_citations = 0
    parser = etree.XMLParser(huge_tree=True, recover=True, resolve_entities=False)
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path + ".tmp"

    with tarfile.open(tar_path, "r:gz") as tf, gzip.open(tmp_path, "wt", encoding="utf-8") as out:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                root = etree.fromstring(data, parser=parser)
                rec = parse_article(root)
                if rec is None:
                    continue
                rec["subset"] = subset
                rec["source_xml"] = member.name
                out.write(json.dumps(rec, ensure_ascii=False))
                out.write("\n")
                n_articles += 1
                n_citations += rec["n_citations"]
            except Exception:
                # never fail the whole tar for one bad article
                continue
    os.replace(tmp_path, out_path)
    return tar_path, n_articles, n_citations, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmc-root", default=DEFAULT_PMC_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--subsets", nargs="+",
                    default=["oa_comm", "oa_noncomm", "oa_other"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="for smoke test: only process N tarballs total")
    args = ap.parse_args()

    tasks = []
    for sub in args.subsets:
        in_dir = Path(args.pmc_root, sub, "xml")
        out_dir = Path(args.out_root, sub)
        for p in sorted(in_dir.glob("*.tar.gz")):
            out_p = out_dir / (p.stem.replace(".tar", "") + ".jsonl.gz")
            if out_p.exists() and out_p.stat().st_size > 0:
                continue
            tasks.append((str(p), str(out_p), sub))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[jats] tasks={len(tasks)} workers={args.workers}", flush=True)

    t0 = time.time()
    done = 0
    art_total = 0
    cit_total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_tar, *t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                tar, na, nc, el = fut.result()
                done += 1
                art_total += na
                cit_total += nc
                if done % 5 == 0 or done <= 3 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate_articles = art_total / max(elapsed, 1)
                    print(f"[jats] {done}/{len(tasks)} tars  "
                          f"articles={art_total:,} citations={cit_total:,}  "
                          f"elapsed={elapsed:.0f}s  rate={rate_articles:,.0f} art/s  "
                          f"last={Path(tar).name}({na}art,{nc}cit,{el:.1f}s)",
                          flush=True)
            except Exception as e:
                t = futs[fut]
                print(f"[jats] FAIL {t[0]}: {e}", flush=True)
                traceback.print_exc()

    print(f"[jats] DONE tars={done} articles={art_total:,} citations={cit_total:,} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
