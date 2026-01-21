#!/usr/bin/env bash
set -euo pipefail

# Reproduce EN-ZH results (monolingual + bilingual) for word-mix + embedding-mix.
# This script is intentionally explicit and mirrors the run/eval flow used elsewhere.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR"

DATASET=mmarco
REPO=unicamp-dl/mmarco
ENCODER=BAAI/bge-m3
ENC_TAG=${ENCODER##*/}
SIZE=100000
QRELS_SPLIT=validation
DTYPE=fp16
BATCH=128
ENC_BATCH=64
TOPK=100
NEG_PROB=0.02

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/repro_en_zh_${RUN_TAG}}"
INDEX_SAVE_ROOT="${INDEX_SAVE_ROOT:-$REPO_ROOT/indexes}"
ENCODE_RUN_NAME="idx-${DATASET}-${ENC_TAG}-sub${SIZE}-en-zh-${RUN_TAG}"
INDEX_ROOT="${INDEX_ROOT:-$INDEX_SAVE_ROOT/$ENCODE_RUN_NAME}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/repro_en_zh_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/repro_en_zh_${RUN_TAG}}"

QUERY_PAIR="en-zh"
Q_EN="${Q_EN:-$DATA_ROOT/mmarco_dev/queries.en.tsv}"
Q_ZH="${Q_ZH:-$DATA_ROOT/mmarco_dev/queries.zh.tsv}"
CM_DIR="${CM_DIR:-$DATA_ROOT/mmarco_dev/queries_cm_5_bands_5-mini}"
COMMON_QIDS="${COMMON_QIDS:-$CM_DIR/qids-common.tsv}"

# GPUs (defaults to a single GPU; override to spread across multiple GPUs)
GPU_EN=${GPU_EN:-0}
GPU_ZH=${GPU_ZH:-0}
GPU_BI=${GPU_BI:-0}
GPU_INDEX_EN=${GPU_INDEX_EN:-0}
GPU_INDEX_ZH=${GPU_INDEX_ZH:-0}

# Per-GPU job slots (set explicitly here)
GPU0_SLOTS=${GPU0_SLOTS:-2}
GPU1_SLOTS=${GPU1_SLOTS:-2}
GPU2_SLOTS=${GPU2_SLOTS:-0}

BANDS=(0 0-20 20-40 40-60 60-80 80-100 100)
ALPHAS=(0 0.1 0.3 0.5 0.7 0.9 1)

FORCE=${FORCE:-0}     # set to 1 to re-generate runs even if files exist
SKIP_EVAL=${SKIP_EVAL:-0}

ONEPASS_DENSE_RUN="$SCRIPT_DIR/onepass_dense_run.py"
ONEPASS_MIX_MONO="$SCRIPT_DIR/onepass_dense_mix_run_custom_lang.py"
ONEPASS_BI_WORD="$SCRIPT_DIR/onepass_bilingual_hub.py"
ONEPASS_BI_MIX="$SCRIPT_DIR/onepass_bilingual_mix_hub_custom_lang.py"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate.py"
ENCODE_SCRIPT="$SCRIPT_DIR/encode_multilingual_corpus.py"

mkdir -p "$RESULT_ROOT" "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*"
}

require_file() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing file: $path" >&2
    exit 1
  fi
}

run_with_log() {
  local log_file=$1
  shift
  mkdir -p "$(dirname "$log_file")"
  "$@" >>"$log_file" 2>&1
}

require_file "$ONEPASS_DENSE_RUN"
require_file "$ONEPASS_MIX_MONO"
require_file "$ONEPASS_BI_WORD"
require_file "$ONEPASS_BI_MIX"
require_file "$EVAL_SCRIPT"
require_file "$ENCODE_SCRIPT"
require_file "$Q_EN"
require_file "$Q_ZH"
require_file "$COMMON_QIDS"

# ------------------------------
# Encode EN/ZH corpus index
# ------------------------------
encode_index() {
  local lang=$1
  local gpu=$2
  local log_file="$LOG_DIR/encode-${lang}-${RUN_TAG}.log"
  mkdir -p "$INDEX_SAVE_ROOT"
  log "Encoding ${lang} corpus index into $INDEX_ROOT (GPU ${gpu})"
  run_with_log "$log_file" \
    python "$ENCODE_SCRIPT" \
      --repo "$REPO" \
      --split collection \
      --encoder "$ENCODER" \
      --device "cuda:${gpu}" \
      --batch "$BATCH" \
      --enc_batch "$ENC_BATCH" \
      --dtype "$DTYPE" \
      --gpu_faiss \
      --faiss_gpu_id "${gpu}" \
      --langs "${lang}" \
      --subset_neg_cap "$SIZE" \
      --neg_prob "$NEG_PROB" \
      --qrels_repo BeIR/msmarco-qrels \
      --qrels_split "$QRELS_SPLIT" \
      --qrels_docid corpus-id \
      --trust_remote \
      --save_root "$INDEX_SAVE_ROOT" \
      --run_name "$ENCODE_RUN_NAME"
}

# ------------------------------
# Monolingual word-mix (EN docs)
# ------------------------------
run_mono_wordmix() {
  local doc_lang=$1
  local gpu=$2
  local run_dir="$RUN_ROOT/${DATASET}-${SIZE}-${doc_lang}-${QUERY_PAIR}-5bands-${ENC_TAG}"
  local result_dir="$RESULT_ROOT/${DATASET}-${SIZE}-${doc_lang}-${QUERY_PAIR}-5bands-${ENC_TAG}"
  local docids_out="$run_dir/docids.txt"
  mkdir -p "$run_dir" "$result_dir"

  log "Monolingual word-mix for ${doc_lang} docs (GPU ${gpu})"

  local log_file="$LOG_DIR/mono-wordmix-${doc_lang}-$(date '+%Y%m%d_%H%M%S').log"

  if [[ ! -f "$run_dir/cm0.trec" || "$FORCE" -eq 1 ]]; then
    log "Running onepass_dense_run.py for ${doc_lang} (q_directory mode)"
    run_with_log "$log_file" \
      python "$ONEPASS_DENSE_RUN" \
        --repo "$REPO" \
        --config "collection-${doc_lang}" \
        --q_config "queries-${doc_lang}" \
        --q_split dev \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --run_out "$run_dir" \
        --docids_out "$docids_out" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --dtype "$DTYPE" \
        --q_directory "$CM_DIR" \
        --q_glob "queries-cm[0-9]*.tsv" \
        --max_docs "$SIZE"
  else
    log "Skipping existing runs under: $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for band in "${BANDS[@]}"; do
      local runfile="$run_dir/cm${band}.trec"
      log "Evaluating band ${band} for ${doc_lang}"
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$runfile" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_out" \
          --filter_qids "$COMMON_QIDS"
    done
  fi
}

# ------------------------------
# Monolingual embedding-mix
# ------------------------------
run_mono_vecmix() {
  local doc_lang=$1
  local gpu=$2
  local run_dir="$RUN_ROOT/${DATASET}-${SIZE}-${doc_lang}-${QUERY_PAIR}-5bands-${ENC_TAG}/vector_mix"
  local result_dir="$RESULT_ROOT/${DATASET}-${SIZE}-${doc_lang}-${QUERY_PAIR}-5bands-${ENC_TAG}/vector_mix"
  local docids_path="${run_dir}/docids-${QUERY_PAIR}.txt"
  mkdir -p "$run_dir" "$result_dir"

  log "Monolingual embedding-mix for ${doc_lang} docs (GPU ${gpu})"

  local log_file="$LOG_DIR/mono-vecmix-${doc_lang}-$(date '+%Y%m%d_%H%M%S').log"
  if [[ ! -f "$run_dir/cm-alpha-0.trec" || "$FORCE" -eq 1 ]]; then
    run_with_log "$log_file" \
      python "$ONEPASS_MIX_MONO" \
        --repo "$REPO" \
        --config "collection-${doc_lang}" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --run_out "$run_dir" \
        --docids_out "$docids_path" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --dtype "$DTYPE" \
        --max_docs "$SIZE" \
        --query_tsv "en=${Q_EN}" \
        --query_tsv "zh=${Q_ZH}" \
        --cm_alphas "0,0.1,0.3,0.5,0.7,0.9,1"
  else
    log "Skipping existing vector-mix runs in $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for alpha in "${ALPHAS[@]}"; do
      log "Evaluating embedding alpha ${alpha} for ${doc_lang}"
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$run_dir/cm-alpha-${alpha}.trec" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_path" \
          --filter_qids "$COMMON_QIDS"
    done
  fi
}

# ------------------------------
# Bilingual word-mix (EN+ZH docs)
# ------------------------------
run_bi_wordmix() {
  local gpu=$1
  local run_dir="$RUN_ROOT/${DATASET}-${SIZE}-bilingual-${QUERY_PAIR}-5bands-${ENC_TAG}"
  local result_dir="$RESULT_ROOT/${DATASET}-${SIZE}-bilingual-${QUERY_PAIR}-5bands-${ENC_TAG}"
  local docids_path="${run_dir}/docids.txt"
  mkdir -p "$run_dir" "$result_dir"

  log "Bilingual word-mix for EN+ZH docs (GPU ${gpu})"

  local log_file="$LOG_DIR/bilingual-wordmix-$(date '+%Y%m%d_%H%M%S').log"
  if [[ ! -f "$run_dir/cm0_base.trec" || "$FORCE" -eq 1 ]]; then
    run_with_log "$log_file" \
      python "$ONEPASS_BI_WORD" \
        --repo "$REPO" \
        --qrels_repo "BeIR/msmarco-qrels" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --docids_out "$docids_path" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --enc_batch "$ENC_BATCH" \
        --dtype "$DTYPE" \
        --max_docs "$SIZE" \
        --langs "english,chinese" \
        --q_directory "$CM_DIR" \
        --outdir "$run_dir"
  else
    log "Skipping existing bilingual word-mix runs in $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for band in "${BANDS[@]}"; do
      log "Evaluating bilingual word-mix band ${band}"
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$run_dir/cm${band}_base.trec" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_path" \
          --filter_qids "$COMMON_QIDS"
    done
  fi
}

# ------------------------------
# Bilingual embedding-mix
# ------------------------------
run_bi_vecmix() {
  local gpu=$1
  local run_dir="$RUN_ROOT/${DATASET}-${SIZE}-bilingual-${QUERY_PAIR}-5bands-${ENC_TAG}/vector_mix"
  local result_dir="$RESULT_ROOT/${DATASET}-${SIZE}-bilingual-${QUERY_PAIR}-5bands-${ENC_TAG}/vector_mix"
  local docids_path="${run_dir}/docids-${QUERY_PAIR}.txt"
  mkdir -p "$run_dir" "$result_dir"

  log "Bilingual embedding-mix for EN+ZH docs (GPU ${gpu})"

  local log_file="$LOG_DIR/bilingual-vecmix-$(date '+%Y%m%d_%H%M%S').log"
  if [[ ! -f "$run_dir/cm-alpha-0.trec" || "$FORCE" -eq 1 ]]; then
    run_with_log "$log_file" \
      python "$ONEPASS_BI_MIX" \
        --repo "$REPO" \
        --qrels_repo "BeIR/msmarco-qrels" \
        --langs "english,chinese" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --docids_out "$docids_path" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --dtype "$DTYPE" \
        --max_docs "$SIZE" \
        --query_tsv "en=${Q_EN}" \
        --query_tsv "zh=${Q_ZH}" \
        --outdir "$run_dir" \
        --cm_alphas "0,0.1,0.3,0.5,0.7,0.9,1"
  else
    log "Skipping existing bilingual vector-mix runs in $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for alpha in "${ALPHAS[@]}"; do
      log "Evaluating bilingual embedding alpha ${alpha}"
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$run_dir/cm-alpha-${alpha}.trec" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_path" \
          --filter_qids "$COMMON_QIDS"
    done
  fi
}

declare -A GPU_CAPACITY=()
declare -A GPU_SLOT_USAGE=()
declare -A PID_TO_GPU=()
declare -A PID_TO_DESC=()
RUNNING_JOBS=0
FAILED=0

unique_gpus=()
for g in "$GPU_EN" "$GPU_ZH" "$GPU_BI"; do
  if [[ ! " ${unique_gpus[*]} " =~ " ${g} " ]]; then
    unique_gpus+=("$g")
  fi
done

for gpu in "${unique_gpus[@]}"; do
  cap_var="GPU${gpu}_SLOTS"
  if [[ -z "${!cap_var:-}" ]]; then
    echo "[ERROR] ${cap_var} is not set. Define slots per GPU in this script." >&2
    exit 1
  fi
  GPU_CAPACITY["$gpu"]="${!cap_var}"
  GPU_SLOT_USAGE["$gpu"]=0
done

reap_finished_jobs() {
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      local status=0
      wait "$pid" || status=$?
      local gpu=${PID_TO_GPU[$pid]}
      local desc=${PID_TO_DESC[$pid]}
      unset PID_TO_GPU["$pid"] PID_TO_DESC["$pid"]
      if [[ ${GPU_SLOT_USAGE[$gpu]:-0} -gt 0 ]]; then
        GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] - 1))
      fi
      RUNNING_JOBS=$((RUNNING_JOBS - 1))
      if [[ $status -ne 0 ]]; then
        FAILED=1
        log "Job failed: ${desc} (status ${status})"
      else
        log "Job finished: ${desc}"
      fi
    fi
  done
}

wait_for_gpu_slot() {
  local gpu=$1
  while true; do
    reap_finished_jobs
    local cap=${GPU_CAPACITY[$gpu]:-0}
    if (( GPU_SLOT_USAGE[$gpu] < cap )); then
      return
    fi
    sleep 5
  done
}

start_job() {
  local gpu=$1
  local desc=$2
  shift 2
  wait_for_gpu_slot "$gpu"
  log "Starting job: ${desc} (GPU ${gpu})"
  (
    set -euo pipefail
    "$@"
  ) &
  local pid=$!
  PID_TO_GPU[$pid]=$gpu
  PID_TO_DESC[$pid]=$desc
  GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] + 1))
  RUNNING_JOBS=$((RUNNING_JOBS + 1))
}

log "=== EN-ZH reproduction start ==="
log "Encoding EN/ZH indexes in parallel."
if [[ ! -d "$INDEX_ROOT" || "$FORCE" -eq 1 ]]; then
  log "Index root $INDEX_ROOT does not exist or FORCE=1; proceeding with encoding."
  encode_fail=0
  encode_index "english" "$GPU_INDEX_EN" &
  encode_en_pid=$!
  encode_index "chinese" "$GPU_INDEX_ZH" &
  encode_zh_pid=$!

  if ! wait "$encode_en_pid"; then
    encode_fail=1
  fi
  if ! wait "$encode_zh_pid"; then
    encode_fail=1
  fi
  if [[ "$encode_fail" -ne 0 ]]; then
    echo "[ERROR] Index encoding failed. Check logs under $LOG_DIR." >&2
    exit 1
  fi
else
  log "Index root $INDEX_ROOT already exists; skipping encoding."
fi

log "Launching jobs with per-GPU slot limits."

start_job "$GPU_EN" "mono-english (word mix)" run_mono_wordmix "english" "$GPU_EN"
start_job "$GPU_EN" "mono-english (vec mix)" run_mono_vecmix "english" "$GPU_EN"
start_job "$GPU_ZH" "mono-chinese (word mix)" run_mono_wordmix "chinese" "$GPU_ZH"
start_job "$GPU_ZH" "mono-chinese (vec mix)" run_mono_vecmix "chinese" "$GPU_ZH"
start_job "$GPU_BI" "bilingual (word mix)" run_bi_wordmix "$GPU_BI"
start_job "$GPU_BI" "bilingual (vec mix)" run_bi_vecmix "$GPU_BI"

while (( RUNNING_JOBS > 0 )); do
  reap_finished_jobs
  sleep 5
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "[ERROR] One or more jobs failed. Check logs under $LOG_DIR." >&2
  exit 1
fi

log "=== EN-ZH reproduction done ==="
