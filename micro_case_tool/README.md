# Micro Case Tools

Tools for fine-grained case analysis in multilingual information retrieval: mining outlier and top-performing query-setting combinations, and inspecting document-level changes between endpoints.

This is a two-stage workflow:
- **Stage A (Mining):** `case_miner.py` — Select interesting cases based on performance variance
- **Stage B (Inspection):** `case_inspector.py` — Deep-dive into one case: which queries drive the delta, what changed in top docs

## Prerequisites

The tool works with results from your retrieval pipeline:
1. Results CSVs from batch evaluation (e.g., `collect_results.py`)
2. External evaluation resources:
   - **qrels** (relevance judgments)
   - **query texts** and **query embeddings** (cached)
   - **run files** (`.trec` format, rank lists)

## Quick Example

```bash
# Step 1: Mine cases (outliers or top performers)
python case_miner.py \
  --raw-results /path/to/full_mmarco_results.csv \
  --processed-results /path/to/full_mmarco_processed.csv \
  --results-root /path/to/results/mmarco_full \
  --run-root /path/to/runs \
  --out-dir artifacts/micro_cases \
  --top-n 30

# Step 2: Inspect one case
python case_inspector.py \
  --cases artifacts/micro_cases/cases.csv \
  --case-id CASE_0001 \
  --qrels /path/to/qrels.tsv \
  --queries-dir /path/to/queries \
  --query-cache-root /path/to/query-embeddings \
  --out-dir artifacts/micro_reports
```

## Files

### Core Scripts

- **`case_miner.py`** — Mine cases from result CSVs
  - Selects outlier settings (underperforming) or top performers (overperforming)
  - Outputs compact case lists with run/eval paths resolved

- **`case_inspector.py`** — Inspect one case in detail
  - Analyzes per-query metrics (nDCG, Recall, doc ranking changes)
  - Classifies failures (ranking drop, recall drop, translation divergence, etc.)
  - Generates per-query diagnostics and markdown report

- **`build_doc_meta_subset.py`** — Build document metadata cache (optional)
  - Fetches doc titles and snippets for top-k docs in each case
  - Enables richer doc-level analysis in inspector
  - Can merge incremental runs for efficiency

- **`micro_case_common.py`** — Shared utilities and helpers
  - CSV I/O, metrics computation, failure classification, markdown generation

- **`micro_case_tool.py`** — Backward-compatible combined dispatcher (optional)

- **`analyze_purity_effect.py`** — Language purity analysis (Stage C, optional)
  - Tests whether mixed-query gains correlate with linguistically impure docs
  - Requires `stanza` and `langid` for NLP features

- **`build_doc_purity_features.py`** — Build language purity features (Stage C, optional)
  - Pre-computes ASCII/English-mix metrics on doc snippets or full text

### Configuration

- **`README_PUBLIC.md`** (this file)
- **`rerun_latin_purity_parallel.sh`** — Example batch script for purity analysis

## Stage A: Mining Cases

### Purpose

Select a ranked list of interesting cases (settings + queries) from your results.

**Outlier selection (mixed underperforms):**
- Triggers: `delta < threshold` or `ci_low < ci_threshold` (default: `ci_low < 0`)
- Use case: Find failure patterns

**Top-performing selection (mixed overperforms):**
- Triggers: `delta >= threshold` and `ci_low >= ci_threshold` (default: `delta >= 0` and `ci_low >= 0`)
- Use case: Find success patterns

### Required Inputs

1. **Raw results CSV** — `pair`, `doc_mix`, `method`, `model`, `mix_ratio`, `ndcg10`, `source_file`
2. **Processed results CSV** — `pair`, `doc_mix`, and CI columns (e.g., `delta_ndcg_ci90_low`, `delta_ndcg_ci90_high`)
3. **Results root** — Directory containing per-setting subdirectories with evaluation files
4. **Run root** — Directory containing `.trec` run files

### Command

```bash
python case_miner.py \
  --raw-results /path/to/full_mmarco_results.csv \
  --processed-results /path/to/full_mmarco_processed.csv \
  --results-root /path/to/results/mmarco_full \
  --run-root /path/to/runs \
  --out-dir artifacts/micro_cases \
  --top-n 30 \
  --outlier-indicator ci \
  --ci-threshold 0.0
```

### Output Files

Generated in `--out-dir`:

- **`cases.csv`** — Top-N outlier cases (sorted by CI low)
- **`cases_all_summary.csv`** — All mined settings with outlier flag
- **`cases_best.csv`** (optional) — Top-N top-performing settings (if `--best-top-n > 0`)
- **`method_comparison.csv`** (optional) — Method (embed vs word) gap analysis (if `--method-comparison`)

**Key columns:**
- `case_id` — unique ID
- `pair`, `doc_mix` — language pair and document regime
- `endpoint_score`, `mixed_best_score`, `delta` — nDCG@10 values
- `ci_low`, `ci_high` — confidence bounds
- `endpoint_run_path`, `mixed_run_path` — resolved run file locations
- `endpoint_eval_path`, `mixed_eval_path` — resolved eval file locations

### Key Options

- `--method {embed,word,all}` — retrieval method filter
- `--doc-type {mono,bi,all}` — document language purity filter
- `--non-english-docs-only` / `--include-english-docs` — doc language filter
- `--outlier-indicator {ci,delta}` — primary outlier signal (default: `ci`)
- `--ci-threshold` (default: `0.0`) — used when `--outlier-indicator ci`
- `--delta-threshold` (default: `-0.2`) — used when `--outlier-indicator delta`
- `--best-top-n` — if `>0`, also output top performers
- `--best-indicator {ci,delta}` — sorting signal for top performers
- `--fill-with-lowest` — if outliers < top-n, fill with lowest-delta cases

## Stage B: Inspecting One Case

### Purpose

Explain why one case shows the observed delta through per-query and per-doc analysis.

Outputs:
1. **`case_report.md`** — Human-readable markdown report
2. **`selected_queries.csv`** — Per-query diagnostics (metrics, embedding geometry, failure labels)
3. **`top_docs_diff.csv`** — Top-10 doc lists for endpoint vs mixed
4. **`case_summary.csv`** — Case-level summary

### Required Inputs

1. **Cases list** — from Stage A (`cases.csv` or `cases_best.csv`)
2. **qrels** — relevance judgments (TREC format: `qid docid relevance`)
3. **Queries directory** — query text files (e.g., `queries.lang.tsv`)
4. **Query cache root** — directory with cached query embeddings (`.npz` files with `qids` and `vecs`)

### Command

```bash
python case_inspector.py \
  --cases artifacts/micro_cases/cases.csv \
  --case-id CASE_0001 \
  --qrels /path/to/qrels.tsv \
  --queries-dir /path/to/queries \
  --query-cache-root /path/to/query-embeddings \
  --query-focus worst \
  --out-dir artifacts/micro_reports
```

### Output Explanation

**`selected_queries.csv`** includes:
- `ndcg10_end`, `ndcg10_mix`, `delta_ndcg10` — per-query nDCG (0-100 scale)
- `recall10_end`, `recall10_mix`, `delta_recall10` — per-query Recall (0-100 scale)
- `overlap10`, `overlap50` — doc list overlap between endpoints
- `endpoint_cos`, `r`, `delta_perp` — query embedding geometry
- `label` — automated failure classification

**Failure labels (deterministic by priority):**
1. `IndexLeakage` — mixed top-10 contains docs in unexpected language
2. `TranslationDivergence` — query variants are semantically/structurally inconsistent
3. `RecallDrop` — mixed loses relevant docs in top-10
4. `RankDrop` — relevant docs present but ranked worse
5. `Unclassified` — none of the above

**`top_docs_diff.csv`** shows top-10 changes per query:
- `rank`, `docid`, `retrieval_score_raw` (model ranking score, not nDCG scale)
- `rel` (qrels label for this doc)

### Key Options

- `--query-focus {worst,best}` — analyze worst-performing or best-performing queries
- `--worst-n` — how many queries to analyze (default: 100)
- `--control-n` — control group size for sanity checks (default: 20)
- `--report-top-worst` — how many focused queries to detail in report (default: 20)
- `--report-top-diff-blocks` — how many top-diff blocks in report (default: 20)
- `--k` — rank depth for metrics (default: 10)
- `--split` — which split to use (`train`, `val`, or `test`)

### Label Thresholds (all tunable)

Customize failure classification with:
- `--label-mismatch-rate-gt` — doc language mismatch rate for IndexLeakage (default: 0.1)
- `--label-endpoint-cos-lt` — cosine threshold for TranslationDivergence (default: 0.9)
- `--label-len-ratio-min`, `--label-len-ratio-max` — query length ratio bounds (default: 0.5–2.0)
- `--label-delta-recall-lt` — recall drop threshold for RecallDrop (default: -5)
- `--label-rankdrop-ndcg-lt` — nDCG drop threshold for RankDrop (default: -2)
- `--label-rankdrop-recall-ge` — recall must not drop for RankDrop (default: 0)

## Optional: Add Document Metadata

For richer doc-level diagnostics, pre-build document metadata:

```bash
python build_doc_meta_subset.py \
  --cases artifacts/micro_cases/cases.csv \
  --all-cases \
  --out artifacts/micro_cases/doc_meta_subset.csv
```

Then pass it to inspector:

```bash
python case_inspector.py \
  --cases artifacts/micro_cases/cases.csv \
  --case-id CASE_0001 \
  --doc-meta artifacts/micro_cases/doc_meta_subset.csv \
  --qrels /path/to/qrels.tsv \
  --queries-dir /path/to/queries \
  --query-cache-root /path/to/query-embeddings \
  --out-dir artifacts/micro_reports
```

### Building Metadata Incrementally

If you want to add best-case docs on top of worst-case docs already built:

```bash
python build_doc_meta_subset.py \
  --cases artifacts/micro_cases/cases_best.csv \
  --all-cases \
  --out artifacts/micro_cases/doc_meta_subset.csv \
  --merge-existing
```

## Optional: Stage C — Language Purity Analysis

Test whether mixed-query gains correlate with linguistically impure relevant docs.

### Requirements

- `stanza` — NLP tokenization
- `langid` — language identification

```bash
pip install stanza langid
```

### Quick Run

```bash
# Build purity features (from doc snippets, fast)
python build_doc_purity_features.py \
  --cases artifacts/micro_cases/cases_best.csv \
  --all-cases \
  --use-doc-meta-only \
  --out artifacts/micro_cases/doc_purity_features_best.csv

# Analyze purity correlation
python analyze_purity_effect.py \
  --cases artifacts/micro_cases/cases_best.csv \
  --all-cases \
  --doc-purity artifacts/micro_cases/doc_purity_features_best.csv \
  --out-dir artifacts/purity_reports

# Review results
cat artifacts/purity_reports/purity_summary_by_case.csv
```

## Tips & Troubleshooting

1. **No cases found?** Lower `--ci-threshold` or `--delta-threshold`, or check that your results CSV has the expected columns.
2. **Can't find query embeddings?** Ensure `--query-cache-root` has subdirs like `en/`, `ar/`, etc., each with `queries.npz`.
3. **Empty per-query metrics?** Ensure qrels and run files are present at the paths in `cases.csv`.
4. **Slow metadata build?** Use `--use-doc-meta-only` to cache from snippets instead of streaming the full collection.
5. **Memory issues with purity analysis?** Process one case at a time with `--case-id CASE_XXXX`.

## Requirements

```
pandas >= 1.2
numpy >= 1.20
```

Optional (for full functionality):
```
stanza >= 1.1  # for purity analysis
langid >= 1.1.6  # for purity analysis
```

Install:
```bash
pip install pandas numpy
pip install stanza langid  # optional, for Stage C
```

## Compatibility

- Python 3.8+
- Tested with pandas 1.3+, numpy 1.21+
- Outputs use UTF-8 encoding

## Getting Result CSVs

See `codemix-dense-retrieval` for example batch evaluation pipelines that produce the input CSVs.

The tool expects:
- **Raw results**: from `collect_results.py` or similar
- **Processed results**: from your statistical analysis pipeline (e.g., bootstrapping, CI estimation)
- **Runs**: from your retrieval system in TREC format
