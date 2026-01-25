# mMARCO Code-Mixed Experiments

This repository reproduces three core experiment pipelines:
- `run_all_vector_pairs.sh` (vector-mix, all bilingual + monolingual pairs)
- `reproduce_en_zh.sh` (EN–ZH reproduction: word-mix + vector-mix)
- `run_ablation.sh` (vector-mix ablations across multiple encoders)

The repo name is arbitrary; all commands assume you are in the repo root.
Warning: full replication is time-consuming (hours to days) and resource-heavy
GPU/CPU, disk, and network. Each section below includes a "Subset tip" showing
how to run a smaller smoke test without changing the default full-replication
instructions.

## Path conventions (defaults)
All scripts use paths relative to the repo root unless you override them:
- `data/`        downloaded queries, qrels cache, query caches
- `indexes/`     FAISS indexes
- `runs/`        run artifacts (trec files, docids)
- `results/`     evaluation outputs

Override via environment variables when needed:
- `DATA_ROOT`, `INDEX_ROOT`, `INDEX_ROOT_BASE`, `RUN_ROOT`, `RESULT_ROOT`,
  `QUERY_CACHE_ROOT`, `QRELS_CACHE`

## Environment

### System requirements
- NVIDIA GPU(s) recommended (we used 2× RTX 3090)
- CUDA 11.8 compatible driver (matches torch 2.6.0+cu118)
- Disk: full mMARCO indexes are large (~30+ GB per language)
- Network: required for Hugging Face datasets/qrels and OpenAI (code-mix generation)

### Python + CUDA stack
- Python 3.11.13
- Torch 2.6.0+cu118
- FAISS GPU 1.8.0 (conda-forge; CUDA build varies by platform)

## Setup

### 1) Create the conda environment
```bash
conda create -n <env_name> python=3.11.13 -y
conda activate <env_name>
```

### 2) Install FAISS GPU via conda
```bash
conda install -c conda-forge faiss-gpu=1.8.0 -y
```
If `faiss.StandardGpuResources` is missing on your build, the scripts will
fall back to CPU FAISS automatically (slower but functionally correct).
Subset tip: none; this is the same for full or subset runs.

### 3) Install Python deps
```bash
pip install -r requirements.txt
```

### 4) Download Stanza models (needed for code-mix generation)
```bash
python - <<'PY'
import stanza
stanza.download('en')
stanza.download('zh')
PY
```

## Data preparation

### A) Download mMARCO query TSVs
```bash
python download_mmarco_queries.py \
  --out_dir ./data/mmarco_dev \
  --split dev \
  --languages english chinese french german italian spanish portuguese dutch russian japanese arabic hindi indonesian vietnamese
```
(choose the languages you want to work with)
Subset tip: for a minimal run, limit `--languages` to two (e.g., `english chinese`)
and keep the rest of the pipeline consistent with those choices.

### B) Generate EN–ZH code-mix bands
This step is required for the word-mix portions in `reproduce_en_zh.sh`.

1) Generate 5 bands (requires `OPENAI_API_KEY`):
```bash
export OPENAI_API_KEY=YOUR_KEY
python generate_cm_bands.py \
  --en ./data/mmarco_dev/queries.en.tsv \
  --zh ./data/mmarco_dev/queries.zh.tsv \
  --out_dir ./data/mmarco_dev/queries_cm_5_bands_5-mini \
  --bands 0-20 20-40 40-60 60-80 80-100 \
  --model gpt-5-mini \
  --workers 1
```

2) Add pure EN/ZH bands used by word-mix scripts:
```bash
cp ./data/mmarco_dev/queries.en.tsv \
  ./data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm0.tsv
cp ./data/mmarco_dev/queries.zh.tsv \
  ./data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm100.tsv
```

3) Recompute `qids-common.tsv` across all bands:
```bash
python - <<'PY'
from pathlib import Path
cm_dir = Path('./data/mmarco_dev/queries_cm_5_bands_5-mini')
qfiles = sorted(cm_dir.glob('queries-cm*.tsv'))
qid_sets = []
for p in qfiles:
    qids = set()
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line:
            continue
        qid = line.split('\t', 1)[0]
        if qid:
            qids.add(qid)
    qid_sets.append(qids)
common = set.intersection(*qid_sets) if qid_sets else set()
(cm_dir / 'qids-common.tsv').write_text('\n'.join(sorted(common)) + '\n', encoding='utf-8')
print(f"wrote {len(common)} qids to {cm_dir / 'qids-common.tsv'}")
PY
```

## Indexing

### A) Full BGE-M3 indexes (for `run_all_vector_pairs.sh`)
```bash
bash run_encode_index_groups.sh
```
Default output:
- `indexes/idx-mmarco-bge-m3-sub8841823`

This is the most expensive step. Adjust languages and GPU settings in
`run_encode_index_groups.sh` if you need to split the workload.
Subset tip: edit `run_encode_index_groups.sh` and set a smaller `SUBSET_CAP`
(e.g., `100000`) and trim `GROUP1` to just the languages you want
(e.g., `GROUP1="english,chinese"`), leaving other groups empty.

### B) Ablation indexes (for `run_ablation.sh`)
```bash
bash run_encode_index_ablation.sh
```
Default output:
- `indexes/idx-mmarco-<enc_tag>-sub100000`
Subset tip: in `run_encode_index_ablation.sh`, lower `SUBSET_NEG_CAP`, reduce
`LANG_CODES`/`LANG_CONFIG_MAP` to your chosen languages, and set `SKIP_*` flags
to skip models you do not want to test.

## Run experiments

### GPU scheduling notes
Defaults are single‑GPU for safety. Each script exposes environment variables
to change GPU IDs and per‑GPU concurrency:
- `run_all_vector_pairs.sh`: `GPUS`, `GPU0_SLOTS`, `GPU1_SLOTS`, `DEFAULT_GPU_SLOTS`
- `run_ablation.sh`: `GPUS`, `GPU0_SLOTS`, `GPU1_SLOTS`, `DEFAULT_GPU_SLOTS`
- `reproduce_en_zh.sh`: `GPU_EN`, `GPU_ZH`, `GPU_BI`, `GPU_INDEX_EN`, `GPU_INDEX_ZH`,
  plus `GPU0_SLOTS`, `GPU1_SLOTS`, `GPU2_SLOTS`

### 1) All vector pairs (bilingual + monolingual, vector-mix)
```bash
INDEX_ROOT=./indexes/idx-mmarco-bge-m3-sub8841823 \
QUERY_CACHE_ROOT=./data/enc-query-mmarco-bge-m3 \
bash run_all_vector_pairs.sh
```
Note: If disk space is limited, you can avoid keeping a full `INDEX_ROOT`.
`run_all_vector_pairs.sh` can be adjusted to build and use indexes on the fly
per job (slower, but much smaller disk footprint).
Subset tip: In `run_all_vector_pairs.sh` edit `BILINGUAL_PAIRS_DEFAULT` and `MONO_JOBS_DEFAULT` to decide the exact language pairs and documents settings to run. For an easy subset, create `failed_pairs.txt` and/or `failed_monolingual_jobs.txt` containing just a few pairs (the script reads
those files if they exist).

Outputs:
- Logs: `logs/vector_pairs`
- Runs: `runs/`
- Results: `results/mmarco_full`

### 2) EN–ZH reproduction (word-mix + vector-mix)
```bash
./reproduce_en_zh.sh
```
Outputs:
- Logs: `logs/repro_en_zh_<timestamp>`
- Runs: `runs/repro_en_zh_<timestamp>`
- Results: `results/repro_en_zh_<timestamp>`

### 3) Ablation runs
```bash
./run_ablation.sh
```
Outputs:
- Logs: `logs/ablation`
- Runs: `runs/` (subdir `ablation2`)
- Results: `results/mmarco_full/ablation2`
Subset tip: in `run_ablation.sh`, shrink `COMPOSITION_PAIRS`,
`HUB_MONO_JOBS`, and `SIZE_BILINGUAL_PAIRS`, and limit `CORE_MODELS` /
`SIZE_MODELS` to just one encoder.

## Summarize results (recommended)
Use these scripts after the runs finish to get compact CSV summaries.

### 1) Aggregate main (vector-pair) results
```bash
python collect_results.py \
  ./results/mmarco_full \
  --output ./full_mmarco_results.csv \
  --processed-out ./full_mmarco_processed_results.csv
```
Defaults (if you omit flags):
- `results/mmarco_full` as the root
- `full_mmarco_results.csv` and `full_mmarco_processed_results.csv` in repo root

### 2) Aggregate ablation results
```bash
python collect_ablation_results.py \
  ./results/mmarco_full/ablation2 \
  --output ./ablation_results.csv \
  --processed-out ./ablation_processed_results.csv
```
Defaults (if you omit flags):
- `results/mmarco_full/ablation2` as the root
- `ablation_results.csv` and `ablation_processed_results.csv` in repo root

## Existing results you can view directly

### Summarized CSVs (already generated)
- `full_mmarco_results.csv`
- `full_mmarco_processed_results.csv`
- `ablation_results.csv`
- `ablation_processed_results.csv`

### Raw evaluation outputs (per run)
Each run directory contains `*-agg.csv` and `*-agg.json` produced by `evaluate.py`.

Vector-pair runs:
- `results/mmarco_full/**/vector_mix/*-agg.json`
- `results/mmarco_full/**/vector_mix/*-agg.csv`

Ablation runs:
- `results/mmarco_full/ablation2/**/vector_mix/*-agg.json`
- `results/mmarco_full/ablation2/**/vector_mix/*-agg.csv`

EN–ZH reproduction:
- `results/repro_en_zh_example/**/**/*-agg.json`
- `results/repro_en_zh_example/**/**/*-agg.csv`

## Tables & Figures from the paper

### Paper table values (stdout)
```bash
python calculate_paper_values.py > paper_values.txt
```
Inputs:
- `full_mmarco_processed_results.csv`
- `full_mmarco_results.csv`

Existing output:
- `paper_values.txt`

### Figures for the paper
```bash
python plot_diagram_2.py
```
Outputs (examples):
- `diagrams_paper/*.png`
- `diagrams_paper/*.pdf`
- `diagrams_paper/ratio_curves/*`
- `diagrams_paper/embedding_projections/*`

### Embedding space analysis
Example run (writes a full report + plots into an output directory):
```bash
python cm_embedding_space_analysis.py \
  --en_file ./data/mmarco_dev/queries.en.tsv \
  --zh_file ./data/mmarco_dev/queries.zh.tsv \
  --cm_files ./data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm0-20.tsv \
             ./data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm20-40.tsv \
             ./data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm40-60.tsv \
             ./data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm60-80.tsv \
             ./data/mmarco_dev/queries_cm_5_bands_5-mini/queries-cm80-100.tsv \
  --qids_common_file ./data/mmarco_dev/queries_cm_5_bands_5-mini/qids-common.tsv \
  --output_dir cm_analysis_dev_5_bge-m3
```
Existing outputs (examples):
- `cm_analysis_dev_5_bge-m3/report.md`
- `cm_analysis_dev_5_bge-m3/band_summaries.csv`
- `cm_analysis_dev_5_bge-m3/viz_umap_by_band_interactive.html`
- `cm_analysis_dev_5_bge-m3/viz_tsne.png`
