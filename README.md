# RefQA: a million-scale cross-document clinical citation-evidence Q&A dataset

Code to reproduce **RefQA**, a million-scale cross-document clinical citation-evidence Q&A dataset built from PubMed Central. This repository contains the **data-construction pipeline only**. The dataset itself is deposited on Zenodo (DOI: 10.5281/zenodo.20805692) and described in the companion *Scientific Data* Data Descriptor.

## Using the released dataset

If you want to **use** the dataset (rather than reproduce it from scratch), start with [`using_refqa_dataset.ipynb`](using_refqa_dataset.ipynb): downloading the files from Zenodo, loading the permissive/full Parquet, and the main use cases (citation-grounded QA, claim-evidence faithfulness, hallucination detection, and SFT).

## Pipeline stages

| Stage | Script | Purpose |
|---|---|---|
| Acquire | `pipeline/download/fetch_pmc_oa.sh` | Fetch PMC OA bulk JATS archives |
| Parse | `pipeline/parse/parse_pmc_jats.py` | Extract citations + IMRaD structure from JATS XML |
| Index | `pipeline/index/build_pubmed_lookup.py`, `build_icite_lookup.py` | Build slim PubMed + iCite lookup tables |
| Enrich | `pipeline/pilot/enrich_full_clinical.py` | Double-clinical filter + join cited abstract/metadata |
| Extract | `pipeline/pilot/extract_qa_glm_v03.py` | GLM CitationQA extraction (system prompt + Pydantic schema; authoritative) |
| Dedupe | `pipeline/dedupe_v03.py` | Collapse to canonical records |
| Gold | `pipeline/sample_gold_1000.py` | Stratified 1,000-record gold sample |
| Split | `pipeline/make_splits.py` | Deterministic train/val/test/gold splits |

## Setup & run

```bash
export REFQA_ROOT=/path/to/your/workdir   # all default I/O paths resolve under here
python -m pip install lxml pyarrow pandas pydantic openai
# then run stages in the order above; see each script's --help for arguments
```
## Dependencies note

The semantic-extraction stage (`pipeline/pilot/extract_qa_glm_v03.py`) requires:
1. An OpenAI-compatible vLLM endpoint serving GLM-5.1-FP8 (pass `--base-url`).
2. The helper module **`llm_client.py`** (a thin async vLLM/OpenAI client wrapper exposing `LLMClient` and `make_vllm_config`) is **bundled alongside the script** at `pipeline/pilot/llm_client.py`. It talks to any OpenAI-compatible `/v1` endpoint via the `openai` package and reads the API key from `OPENAI_API_KEY` (defaulting to `EMPTY`, which vLLM accepts). No endpoint address or key is hard-coded.

All other stages depend only on the packages listed above.

## Reproducibility
Deterministic seeds + rule-based filtering reproduce the corpus statistically from the pinned 2026 snapshots (PMC OA 2026-01-23 + incrementals to 2026-04-28; PubMed 2026 baseline + updates to 2026-04; iCite 2026). Bit-identical reproduction is not guaranteed (vLLM under concurrent dispatch).

## License
MIT. The dataset files follow per-record licenses inherited from each citing PMC article (see the Data Descriptor).
