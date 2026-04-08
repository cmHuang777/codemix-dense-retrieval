# Micro Case Tools (`case_miner.py` + `case_inspector.py`)

This tool implements the two-stage micro investigation workflow using the **existing artifacts** in this repo/workflow.

- Stage A: `mine` (case miner)
- Stage B: `inspect` (case inspector)

It is designed around the outputs produced by:
- [`run_all_vector_pairs.sh`](/home/hcming/test/run_all_vector_pairs.sh)
- [`collect_results.py`](/home/hcming/test/collect_results.py)
- compiled results under [`compiled_results/`](/home/hcming/test/compiled_results)

## Files

- Miner entrypoint: [`case_miner.py`](/home/hcming/test/micro_case_tool/case_miner.py)
- Inspector entrypoint: [`case_inspector.py`](/home/hcming/test/micro_case_tool/case_inspector.py)
- Batch doc metadata builder: [`build_doc_meta_subset.py`](/home/hcming/test/micro_case_tool/build_doc_meta_subset.py)
- Shared helpers (reused by both): [`micro_case_common.py`](/home/hcming/test/micro_case_tool/micro_case_common.py)
- Backward-compatible combined dispatcher: [`micro_case_tool.py`](/home/hcming/test/micro_case_tool/micro_case_tool.py)

## Defaults (Aligned With Current Setup)

`mine` defaults:
- `--raw-results`: `/home/hcming/test/compiled_results/full_mmarco_results_20260210.csv`
- `--processed-results`: `/home/hcming/test/compiled_results/full_mmarco_processed_results_20260210.csv`
- `--results-root`: `/home/hcming/test/results/mmarco_full`
- `--run-root`: `/home/hcming/data/runs`
- `--method`: `embed`
- `--doc-type`: `all`
- `--non-english-docs-only`: enabled
- `--outlier-indicator`: `ci` (default primary signal)

`inspect` defaults:
- `--qrels`: `/home/hcming/data/data/qrels_cache/BeIR_msmarco-qrels-default-validation.tsv`
- `--queries-dir`: `/home/hcming/data/data/mmarco_dev`
- `--query-cache-root`: `/home/hcming/data/enc-query-mmarco-bge-m3`

Named default/tuning constants are centralized in [`micro_case_common.py`](/home/hcming/test/micro_case_tool/micro_case_common.py).

## Stage A: Mine Cases

## Purpose
Mine mixed-query settings and output compact case lists. By default, outlier selection is CI-driven (`delta_ndcg_ci90_low < 0`), and you can optionally emit a top-performing case list.

## Command

```bash
python /home/hcming/test/micro_case_tool/case_miner.py \
  --raw-results /home/hcming/test/compiled_results/full_mmarco_results_20260210.csv \
  --processed-results /home/hcming/test/compiled_results/full_mmarco_processed_results_20260210.csv \
  --results-root /home/hcming/test/results/mmarco_full \
  --run-root /home/hcming/data/runs \
  --out-dir /home/hcming/test/micro_case_tool/micro_cases \
  --top-n 30
```

## Top-performing cases:

Top-performing means mixed beats endpoint with confidence support (`delta >= 0`, `ci_low >= 0`).

### 1) Produce top-performing cases from current defaults

```bash
python /home/hcming/test/micro_case_tool/case_miner.py \
  --raw-results /home/hcming/test/compiled_results/full_mmarco_results_20260210.csv \
  --processed-results /home/hcming/test/compiled_results/full_mmarco_processed_results_20260210.csv \
  --results-root /home/hcming/test/results/mmarco_full \
  --run-root /home/hcming/data/runs \
  --out-dir /home/hcming/test/micro_case_tool/micro_cases \
  --best-top-n 20 \
  --best-cases-file cases_best.csv \
  --best-indicator ci \
  --best-delta-min 0.0 \
  --best-ci-low-min 0.0
```

## Key options

- `--method {embed,word,all}`
- `--doc-type {mono,bi,all}`
- `--non-english-docs-only` / `--include-english-docs`
- `--outlier-indicator {ci,delta}` (default `ci`)
- `--ci-threshold` (default `0.0`, used when `--outlier-indicator ci`)
- `--delta-threshold` (default `-0.2`, used when `--outlier-indicator delta`; 0-100 scale)
- `--method-comparison` (writes `method_comparison.csv` when both methods exist)
- `--fill-with-lowest` (if outliers are fewer than `--top-n`, fills with lowest-delta non-outliers)
- `--best-top-n` (if `>0`, also writes top-performing cases)
- `--best-cases-file` (default `cases_best.csv`)
- `--best-indicator {ci,delta}` (primary sorting signal for top-performing output)
- `--best-delta-min` (minimum delta for top-performing output, default `0.0`)
- `--best-ci-low-min` (minimum CI low for top-performing output, default `0.0`)

## Inputs required

- Raw results CSV from `collect_results.py` containing at least:
  - `pair`, `doc_mix`, `method`, `model`, `mix_ratio`, `ndcg10`, `source_file`
- Processed results CSV containing at least:
  - `pair`, `doc_mix`, `delta_ndcg_ci90_low`, `delta_ndcg_ci90_high` (or your chosen CI columns)
- Result root and run root paths so the tool can resolve:
  - evaluation files (agg/perquery)
  - corresponding `.trec` run files

## Outputs

Under `--out-dir`:

- `cases.csv`
  - selected outlier cases (`top-n` after sorting by active outlier indicator: `ci_low` for `ci`, `delta` for `delta`)
- `cases_all_summary.csv`
  - all mined settings + `is_outlier`
- `cases_best.csv` (optional; filename configurable via `--best-cases-file`)
  - selected top-performing settings filtered by `--best-delta-min` and `--best-ci-low-min`, sorted by `--best-indicator`
- `method_comparison.csv` (optional)
  - only when `--method-comparison` and both word/embed rows exist

`cases.csv`/`cases_all_summary.csv` include:
- `case_id`
- `pair`, `doc_mix`, `doc_type`, `doc_lang`, `q_lang_a`, `q_lang_b`
- `model`, `method`, `doc_index_id`
- `endpoint_lambda`, `lambda_star`
- `endpoint_score`, `mixed_best_score`, `delta`
- `ci_low`, `ci_high`
- `endpoint_run_path`, `mixed_run_path`
- `endpoint_eval_path`, `mixed_eval_path`

## Stage B: Inspect One Case

## Purpose
For one mined case, explain which queries drive the drop or gain and what changed in top docs.

## Command

```bash
python /home/hcming/test/micro_case_tool/case_inspector.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases.csv \
  --case-id CASE_0001 \
  --query-focus worst \
  --qrels /home/hcming/data/data/qrels_cache/BeIR_msmarco-qrels-default-validation.tsv \
  --queries-dir /home/hcming/data/data/mmarco_dev \
  --query-cache-root /home/hcming/data/enc-query-mmarco-bge-m3 \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports
```

## Exact commands: inspector (worst vs top queries)

### A) Worst-query inspection (default behavior, explicit)

```bash
python /home/hcming/test/micro_case_tool/case_inspector.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases.csv \
  --case-id CASE_0007 \
  --query-focus worst \
  --worst-n 100 \
  --control-n 20 \
  --report-top-worst 20 \
  --report-top-diff-blocks 20 \
  --qrels /home/hcming/data/data/qrels_cache/BeIR_msmarco-qrels-default-validation.tsv \
  --queries-dir /home/hcming/data/data/mmarco_dev \
  --query-cache-root /home/hcming/data/enc-query-mmarco-bge-m3 \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports
```

### B) Top-query inspection (best gains)

```bash
python /home/hcming/test/micro_case_tool/case_inspector.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases_best.csv \
  --case-id CASE_0247 \
  --query-focus best \
  --worst-n 100 \
  --control-n 20 \
  --report-top-worst 20 \
  --report-top-diff-blocks 20 \
  --qrels /home/hcming/data/data/qrels_cache/BeIR_msmarco-qrels-default-validation.tsv \
  --queries-dir /home/hcming/data/data/mmarco_dev \
  --query-cache-root /home/hcming/data/enc-query-mmarco-bge-m3 \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports
```

Note: `--worst-n` is the focused-set size knob for both modes (`worst` and `best`).

### C) Top-query inspection with doc metadata

Add to the arguments:
```bash
--doc-meta /home/hcming/test/micro_case_tool/micro_cases/doc_meta_subset.csv 
```

## Key options

- `--query-focus {worst,best}` (default `worst`)
- `--worst-n`, `--control-n`
- `--k`, `--rank-depth`
- `--report-top-worst`, `--report-top-diff-blocks`
- Failure-label thresholds (all tunable now):
  - `--label-mismatch-rate-gt`
  - `--label-endpoint-cos-lt`
  - `--label-len-ratio-min`
  - `--label-len-ratio-max`
  - `--label-delta-recall-lt`
  - `--label-rankdrop-ndcg-lt`
  - `--label-rankdrop-recall-ge`

Optional doc metadata:

```bash
python /home/hcming/test/micro_case_tool/build_doc_meta_subset.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases.csv \
  --all-cases \
  --qid-list /home/hcming/data/data/mmarco_dev/queries_cm_5_bands_5-mini/qids-common.tsv \
  --out /home/hcming/test/micro_case_tool/micro_cases/doc_meta_subset.csv \
  --stats-out /home/hcming/test/micro_case_tool/micro_cases/doc_meta_subset_stats.csv
```

To incrementally extend an existing metadata file (for example, add best-case docs on top of already-built worst-case docs), add `--merge-existing`:

```bash
python /home/hcming/test/micro_case_tool/build_doc_meta_subset.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases_best.csv \
  --all-cases \
  --qid-list /home/hcming/data/data/mmarco_dev/queries_cm_5_bands_5-mini/qids-common.tsv \
  --out /home/hcming/test/micro_case_tool/micro_cases/doc_meta_subset.csv \
  --stats-out /home/hcming/test/micro_case_tool/micro_cases/doc_meta_subset_stats.csv \
  --merge-existing
```

Use `--case-id CASE_0001` to build metadata for one case only.

Then run inspector with the shared metadata file:

```bash
python /home/hcming/test/micro_case_tool/case_inspector.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases.csv \
  --case-id CASE_0001 \
  --doc-meta /home/hcming/test/micro_case_tool/micro_cases/doc_meta_subset.csv \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports
```

Optional qid restriction (recommended when you want the same evaluation universe used in run scripts, e.g. common qids, current setup ensures all runs use common qids already):

```bash
python /home/hcming/test/micro_case_tool/case_inspector.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases.csv \
  --case-id CASE_0001 \
  --qid-list /home/hcming/data/data/mmarco_dev/queries_cm_5_bands_5-mini/qids-common.tsv \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports
```

## Inputs required

- `cases.csv` from Stage A
- qrels TSV (`qid docid rel` or TREC 4-column compatible)
- query texts: `queries.<lang>.tsv` in `--queries-dir`
- query embedding caches: `<query-cache-root>/<lang>/queries.npz` containing `qids` and `vecs`
- optional qid universe file (`--qid-list`)

Optional:
- `doc_meta_subset.csv` with columns:
  - `case_id` (required)
  - `qid` (required)
  - `condition` (required; `endpoint` or `mixed`)
  - `rank` (required)
  - `docid` (required)
  - `lang` (optional)
  - `title` (optional)
  - `snippet` (optional)

## Outputs

Under `<out-dir>/<case_id>/`:

- `case_report.md`
- `selected_queries.csv`
- `top_docs_diff.csv`
- `case_summary.csv`

### `selected_queries.csv`
Includes focused (`best` or `worst` via `--query-focus`) + control query diagnostics:
- per-query nDCG/Recall deltas
- metric source per query (`evaluate_perquery` preferred, fallback `recomputed_from_run_qrels`)
- overlap@10/@50
- first relevant rank shift
- text length ratio
- endpoint cosine + mixing geometry (`r`, `delta_perp`, `cos_to_a`, `cos_to_b`)
- optional doc sanity metrics when `doc_meta` is provided
- failure label

### `top_docs_diff.csv`
Top-10 docs for `endpoint` and `mixed` per selected query:
- `case_id`, `qid`, `condition`, `rank`, `docid`, `retrieval_score_raw`, `score`, `rel`, `doc_lang`, `snippet`
- `retrieval_score_raw`/`score` are raw ranking scores from `.trec` files (not nDCG/Recall and not on 0-100 scale)

### `case_report.md`
Contains:
1. case header
2. distribution summary for ΔnDCG@10
3. failure-label breakdown (focused set)
4. top focused-query table with all measured diagnostics (`--report-top-worst`)
5. top per-query diff blocks for the focused set (`--report-top-diff-blocks`)

Per-query metrics scale:
- nDCG and Recall values are on the same 0-100 scale as [`evaluate.py`](/home/hcming/test/evaluate.py).
- Inspector first reads `endpoint_perquery_path` and `mixed_perquery_path` from `cases.csv`; if both have a query, those values are used directly.
- If per-query files are missing/incomplete for a query, inspector falls back to recomputing from run + qrels.

## Metric Glossary (What Each Measured Value Means)

All delta values use this sign convention:
- `delta = mixed - endpoint`
- Negative means mixed is worse than endpoint.
- Positive means mixed is better than endpoint.

Stage A (`case_miner.py`) core values:
- `endpoint_lambda`: best endpoint mixing weight among `{0, 1}`.
- `lambda_star`: best non-endpoint mixing weight among `(0, 1)`.
- `endpoint_score`: nDCG@10 score (0-100) of the chosen endpoint run.
- `mixed_best_score`: nDCG@10 score (0-100) of the chosen best mixed run.
- `delta`: `mixed_best_score - endpoint_score` (percentage-point change in nDCG@10).
- `ci_low`, `ci_high`: lower/upper CI bounds from processed results (typically `delta_ndcg_ci90_low/high`).
- `processed_delta_ndcg`: delta value from processed CSV for the matched setting.
- `is_outlier`: outlier flag decided by `--outlier-indicator`.
- `outlier-indicator=ci`: outlier if `ci_low < ci_threshold` (default threshold `0.0`).
- `outlier-indicator=delta`: outlier if `delta < delta_threshold` (default `-0.2`, 0-100 scale).
- top-performing output (`cases_best.csv`):
  - keep rows where `delta >= best_delta_min` and `ci_low >= best_ci_low_min`
  - sort descending by `--best-indicator` (ties broken by the secondary metric and identifiers)

Stage B (`case_inspector.py`) per-query values:
- `metric_source`: where per-query metrics came from.
- `evaluate_perquery`: read directly from `*-perquery.csv` produced by [`evaluate.py`](/home/hcming/test/evaluate.py).
- `recomputed_from_run_qrels`: fallback recomputation from run + qrels when needed.
- `ndcg10_end`, `ndcg10_mix`: per-query nDCG@10 on 0-100 scale.
- `delta_ndcg10`: `ndcg10_mix - ndcg10_end`.
- `recall10_end`, `recall10_mix`: per-query Recall@10 on 0-100 scale.
- `delta_recall10`: `recall10_mix - recall10_end`.
- `first_rel_rank50_end`, `first_rel_rank50_mix`: rank of first relevant hit within top-50 (`inf` if none found).
- `best_rel_rank_shift`: `first_rel_rank50_mix - first_rel_rank50_end`; positive means first relevant moved down (worse), negative means moved up (better).
- `overlap10`, `overlap50`: number of shared docids between endpoint and mixed top-k lists.
- `token_count_a`, `token_count_b`: word-token counts of query A/B (URL/email/handle-like pieces filtered; digit-containing tokens dropped).
  - Uses Stanza tokenization when available for the query language; otherwise uses a Unicode-word fallback (Han fallback approximates by character tokens).
- `len_ratio`: token-length ratio `token_count_a / token_count_b`.
- `endpoint_cos`: cosine similarity between endpoint query embeddings (`q_lang_a` vs `q_lang_b`).
- `r`: projection coefficient of mixed query embedding on the endpoint line (`qa -> qb`) at `lambda_star`.
- `delta_perp`: perpendicular distance of mixed embedding to that endpoint line.
- `cos_to_a`, `cos_to_b`: cosine similarity of mixed embedding to endpoint A/B embeddings.
- `doc_lang_mismatch_rate10_end`, `doc_lang_mismatch_rate10_mix`: among top-10 docs with known `doc_meta.lang`, fraction whose language is not expected for the case.
- `ascii_ratio10_end`, `ascii_ratio10_mix`: mean ASCII-letter ratio in top-10 snippets (only where snippet text exists).
- `label`: deterministic failure label (`IndexLeakage`, `TranslationDivergence`, `RecallDrop`, `RankDrop`, `Unclassified`) assigned by rule priority.
  - Rules are tunable through the `--label-*` CLI thresholds above.

Top-doc diff values (`top_docs_diff.csv` and report blocks):
- `rank`: rank position from run file.
- `docid`: retrieved document id.
- `rel`: qrels relevance label for `(qid, docid)` (0 means non-relevant/unjudged in this qrels table).
- `retrieval_score_raw` (and `score`): raw model ranking score from `.trec`.
- Raw retrieval scores are model-internal ranking values, not calibrated percentages, and are not comparable to nDCG/Recall scale.
- Raw retrieval score is primarily useful for within-run ordering, score ties, and confidence gaps between nearby ranked docs.

## Failure Labels (Priority Order)

Focused-query label assignment follows this deterministic order:
1. `IndexLeakage`
2. `TranslationDivergence`
3. `RecallDrop`
4. `RankDrop` (objective rule: `delta_ndcg10 < label_rankdrop_ndcg_lt` AND `delta_recall10 >= label_rankdrop_recall_ge`)
5. `Unclassified` (none of the above rules matched)

How to interpret each label:
1. `IndexLeakage`
Trigger: `doc_lang_mismatch_rate10_mix > label_mismatch_rate_gt`.
Meaning: mixed top-10 contains docs outside expected doc language(s), so degradation likely comes from language leakage/misaligned index hits.
2. `TranslationDivergence`
Trigger: `endpoint_cos < label_endpoint_cos_lt` OR `len_ratio < label_len_ratio_min` OR `len_ratio > label_len_ratio_max`.
Meaning: query-language variants look semantically/structurally inconsistent; mixing may move the query representation in a harmful direction.
3. `RecallDrop`
Trigger: `delta_recall10 < label_delta_recall_lt`.
Meaning: mixed run loses relevant docs in top-10; primarily a retrieval-coverage failure.
4. `RankDrop`
Trigger: `delta_ndcg10 < label_rankdrop_ndcg_lt` AND `delta_recall10 >= label_rankdrop_recall_ge`.
Meaning: relevant docs are still present (recall not worse), but are ordered worse; primarily a ranking-quality failure.
5. `Unclassified`
Trigger: none of the above rules matched.
Meaning: current signals are insufficient for confident assignment; inspect manually or tune `--label-*` thresholds.

Why order matters:
- Labels are first-match by priority, not multi-label.
- Example: a query can satisfy both `RankDrop` and `TranslationDivergence`; it will be labeled `TranslationDivergence` because that rule is earlier.

## Word-mix vs Embedding-mix Comparison

If your raw results CSV contains both methods, run miner with:

```bash
python /home/hcming/test/micro_case_tool/case_miner.py \
  --raw-results /path/to/full_results.csv \
  --processed-results /path/to/full_processed.csv \
  --method all \
  --doc-type all \
  --method-comparison \
  --out-dir /home/hcming/test/micro_case_tool/micro_cases
```

Then inspect method gaps in:
- `/home/hcming/test/micro_case_tool/micro_cases/method_comparison.csv`

## Stage C: Purity Analysis (Language Impurity Test)

## Purpose
Test whether mixed-query gains concentrate on queries whose relevant docs are linguistically impure (for example, non-English target-language docs that contain English spans or English-like tokens).

## Environment
Use the `ir-lab-3` conda environment:

```bash
source /home/hcming/miniconda3/etc/profile.d/conda.sh
conda activate ir-lab-3
```

## Files

- Doc purity builder: [`build_doc_purity_features.py`](/home/hcming/test/micro_case_tool/build_doc_purity_features.py)
- Query-level purity analyzer: [`analyze_purity_effect.py`](/home/hcming/test/micro_case_tool/analyze_purity_effect.py)

## Recommended whole-set run (fast first pass)

This mode uses the existing [`doc_meta_subset.csv`](/home/hcming/test/micro_case_tool/micro_cases/doc_meta_subset.csv) snippets instead of streaming full collection bodies.

Use this when you want to run the full query set for a case list quickly.

### A) Top-performing cases (`cases_best.csv`)

Build doc purity features for all docs touched by the selected cases and all qrels for those queries:

```bash
python /home/hcming/test/micro_case_tool/build_doc_purity_features.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases_best.csv \
  --all-cases \
  --use-doc-meta-only \
  --out /home/hcming/test/micro_case_tool/micro_cases/doc_purity_features_cases_best.csv
```

Then analyze all queries for those cases:

```bash
python /home/hcming/test/micro_case_tool/analyze_purity_effect.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases_best.csv \
  --all-cases \
  --doc-purity /home/hcming/test/micro_case_tool/micro_cases/doc_purity_features_cases_best.csv \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports_purity/best \
  --summary-out /home/hcming/test/micro_case_tool/micro_reports_purity/best/purity_summary_by_case.csv
```

### B) Outlier / bad cases (`cases.csv`)

```bash
python /home/hcming/test/micro_case_tool/build_doc_purity_features.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases.csv \
  --all-cases \
  --use-doc-meta-only \
  --out /home/hcming/test/micro_case_tool/micro_cases/doc_purity_features_cases.csv

python /home/hcming/test/micro_case_tool/analyze_purity_effect.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases.csv \
  --all-cases \
  --doc-purity /home/hcming/test/micro_case_tool/micro_cases/doc_purity_features_cases.csv \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports_purity/outliers \
  --summary-out /home/hcming/test/micro_case_tool/micro_reports_purity/outliers/purity_summary_by_case.csv
```

Note: there is no merged case manifest for `cases.csv` + `cases_best.csv` today, so if you want both good and bad sets, run the two passes above.

## Slower body-text run (higher fidelity)

If you want purity computed from full collection bodies instead of snippets, drop `--use-doc-meta-only`:

```bash
python /home/hcming/test/micro_case_tool/build_doc_purity_features.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases_best.csv \
  --all-cases \
  --out /home/hcming/test/micro_case_tool/micro_cases/doc_purity_features_cases_best.csv
```

This is heavier because it streams the collection and can take substantially longer.

## Single-case run

For one case only:

```bash
python /home/hcming/test/micro_case_tool/build_doc_purity_features.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases_best.csv \
  --case-id CASE_0247 \
  --use-doc-meta-only \
  --out /home/hcming/test/micro_case_tool/micro_cases/doc_purity_features_CASE_0247.csv

python /home/hcming/test/micro_case_tool/analyze_purity_effect.py \
  --cases /home/hcming/test/micro_case_tool/micro_cases/cases_best.csv \
  --case-id CASE_0247 \
  --doc-purity /home/hcming/test/micro_case_tool/micro_cases/doc_purity_features_CASE_0247.csv \
  --out-dir /home/hcming/test/micro_case_tool/micro_reports_purity/single \
  --summary-out /home/hcming/test/micro_case_tool/micro_reports_purity/single/purity_summary_by_case.csv
```

## What You Should Get

From `build_doc_purity_features.py`:
- one CSV with one row per `(case_id, docid)`
- key columns: `doc_lang`, `ascii_word_ratio`, `ascii_run_max`, `english_span_level`, `doc_purity_label`

From `analyze_purity_effect.py`:
- one per-case CSV at `<out-dir>/<CASE_ID>/query_purity_effect.csv`
- one per-case markdown report at `<out-dir>/<CASE_ID>/purity_analysis.md`
- one cross-case summary CSV at the `--summary-out` path

## How To Read The Outputs

`query_purity_effect.csv` tells you, for every query in the selected case set:
- whether the relevant docs are bucketed as `all_rel_pure`, `has_rel_mixed_light_only`, `has_rel_mixed_clear`, or `indeterminate_only`
- whether the query pair is labeled `clean_translation`, `suspect_translation`, or `unknown`
- `delta_ndcg10`, `delta_recall10`, and first relevant rank movement under mixing

`purity_summary_by_case.csv` tells you, for each case and bucket:
- how many queries fell into that bucket
- mean/median `delta_ndcg10`
- positive-delta rate
- mean first relevant rank gain
- bootstrap CI for mean `delta_ndcg10`

`purity_analysis.md` is the readable report:
- summary bucket table
- top gains with mixed relevant docs
- top gains with pure relevant docs
- top drops

## Recommended interpretation order

1. Start from `purity_summary_by_case.csv`.
2. Check whether `has_rel_mixed_clear` queries have larger positive deltas than `all_rel_pure`.
3. Then open the corresponding per-case `purity_analysis.md` to inspect the concrete queries driving that result.
4. Use `query_purity_effect.csv` when you need raw per-query rows for filtering or plotting.

## Practical Workflow

1. Regenerate/refresh compiled CSVs with `collect_results.py` if needed.
2. Run `case_miner.py` with your desired filters.
3. Pick a `case_id` from `cases.csv` (worst) or `cases_best.csv` (best).
4. Build/refresh doc metadata with `build_doc_meta_subset.py` (`--merge-existing` if you want incremental extension).
5. Run `case_inspector.py` with `--query-focus worst|best` for that case.
6. Review `case_report.md` + `selected_queries.csv` + `top_docs_diff.csv`.

## Changing Paths Quickly

All major directories are CLI flags and can be overridden per run:
- `--raw-results`
- `--processed-results`
- `--results-root`
- `--run-root`
- `--qrels`
- `--queries-dir`
- `--query-cache-root`
- `--doc-meta`
- `--out-dir`
