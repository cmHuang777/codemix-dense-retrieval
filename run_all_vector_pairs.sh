#!/usr/bin/env bash
set -euo pipefail

# Run bilingual + monolingual vector experiments for a list of language pairs,
# distributing work across two GPUs and queueing jobs when devices are busy.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET=mmarco
REPO=unicamp-dl/mmarco
ENCODER=BAAI/bge-m3
ENC_TAG=${ENCODER##*/}
SIZE=8841823
QRELS_SPLIT=validation
CM_ALPHAS="0,0.1,0.3,0.5,0.7,0.9,1"
BANDS=(0 0.1 0.3 0.5 0.7 0.9 1)

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/mmarco_full}"
QUERY_DIR="${QUERY_DIR:-$DATA_ROOT/mmarco_dev}"
COMMON_QIDS="${COMMON_QIDS:-$QUERY_DIR/queries_cm_5_bands_5-mini/qids-common.tsv}"
INDEX_ROOT="${INDEX_ROOT:-$REPO_ROOT/indexes/idx-${DATASET}-${ENC_TAG}-sub${SIZE}}"
QUERY_CACHE_ROOT="${QUERY_CACHE_ROOT:-$DATA_ROOT/enc-query-${DATASET}-${ENC_TAG}}"
QRELS_CACHE="${QRELS_CACHE:-$DATA_ROOT/qrels_cache}"

BILINGUAL_SCRIPT="${SCRIPT_DIR}/onepass_bilingual_mix_hub_custom_lang.py"
MONO_SCRIPT="${SCRIPT_DIR}/onepass_dense_mix_run_custom_lang.py"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate.py"
CACHE_SCRIPT="${SCRIPT_DIR}/cache_queries_for_mix.py"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/vector_pairs}"
mkdir -p "$LOG_DIR" "$RESULT_ROOT"

GPUS=(${GPUS:-0})
SLEEP_INTERVAL=${SLEEP_INTERVAL:-60}   # seconds between scheduler polls
BETWEEN_LAUNCH_SLEEP=${BETWEEN_LAUNCH_SLEEP:-5}   # fixed pause between launches
DEFAULT_GPU_SLOTS=1   # default concurrent jobs per GPU
BILINGUAL_PAIRS_FILE=${BILINGUAL_PAIRS_FILE:-${SCRIPT_DIR}/failed_pairs.txt}
MONO_JOBS_FILE=${MONO_JOBS_FILE:-${SCRIPT_DIR}/failed_monolingual_jobs.txt}

declare -A LANG_NAME_MAP=(
    [ar]=arabic
    [de]=german
    [en]=english
    [es]=spanish
    [fr]=french
    [hi]=hindi
    [id]=indonesian
    [it]=italian
    [ja]=japanese
    [nl]=dutch
    [pt]=portuguese
    [ru]=russian
    [vi]=vietnamese
    [zh]=chinese
)
LANG_CODES=(ar de en es fr hi id it ja nl pt ru vi zh)

declare -A QUERY_FILE_MAP
for code in "${!LANG_NAME_MAP[@]}"; do
    QUERY_FILE_MAP[$code]="${QUERY_DIR}/queries.${code}.tsv"
done

BILINGUAL_PAIRS_DEFAULT=(
    "en:fr"
    "en:it"
    "en:pt"
    "en:nl"
    "es:fr"
    "es:it"
    "fr:pt"
    "it:pt"
    "de:fr"
    "de:it"
    "nl:fr"
    "nl:it"
    "nl:es"
    "ja:hi"
    "ja:ru"
    "ar:zh"
    "hi:zh"
    "es:pt"
    "de:nl" 
    "en:de"
    "en:es" 
    "es:de"
    "en:id"
    "id:vi"
    "en:vi"
    "en:ru"
    "en:hi"
    "en:ar"
    "en:zh"
    "id:zh"
    "en:ja"
    "hi:ar"
    "fr:it"
    "zh:ja"
    "zh:ru"
)
BILINGUAL_PAIRS=("${BILINGUAL_PAIRS_DEFAULT[@]}")
if [[ -f "$BILINGUAL_PAIRS_FILE" ]]; then
    mapfile -t _pairs_from_file < <(grep -v '^\s*#' "$BILINGUAL_PAIRS_FILE" | sed '/^\s*$/d' | tr '[:upper:]' '[:lower:]')
    if (( ${#_pairs_from_file[@]} > 0 )); then
        BILINGUAL_PAIRS=("${_pairs_from_file[@]}")
        echo "[INFO] Loaded ${#BILINGUAL_PAIRS[@]} bilingual pairs from ${BILINGUAL_PAIRS_FILE}" >&2
    else
        echo "[WARN] ${BILINGUAL_PAIRS_FILE} exists but contained no pairs; using built-in bilingual defaults." >&2
    fi
fi

MONO_JOBS_DEFAULT=(
    "en:en:fr"  # docLang:queryLangA:queryLangB
    "en:en:it"
    "en:en:pt"
    "en:en:nl"
    "fr:en:fr"
    "it:en:it"
    "pt:en:pt"
    "nl:en:nl"
    "es:es:fr"
    "fr:es:fr"
    "es:es:it"
    "it:es:it"
    "fr:fr:pt"
    "pt:fr:pt"
    "it:it:pt"
    "pt:it:pt"
    "de:de:fr"
    "fr:de:fr"
    "de:de:it"
    "it:de:it"
    "nl:nl:fr"
    "fr:nl:fr"
    "nl:nl:it"
    "it:nl:it"
    "nl:nl:es"
    "es:nl:es"
    "ja:ja:hi"
    "hi:ja:hi"
    "ja:ja:ru"
    "ru:ja:ru"
    "ar:ar:zh"
    "zh:ar:zh"
    "hi:hi:zh"
    "zh:hi:zh"
    "pt:es:pt"
    "nl:de:nl"
    "de:en:de"
    "es:en:es"
    "vi:id:vi"
    "zh:en:zh"
    "zh:id:zh"
    "ar:hi:ar"
    "ar:en:ar" 
    "de:es:de" 
    "de:de:nl" 
    "en:en:de" 
    "en:en:ar" 
    "en:en:es"
    "en:en:hi"
    "en:en:id"
    "en:en:zh"
    "en:en:ja"
    "en:en:ru"
    "en:en:vi"
    "es:es:de"
    "es:es:pt"
    "hi:en:hi"
    "hi:hi:ar"
    "id:en:id"
    "id:id:vi"
    "id:id:zh"
    "it:fr:it"
    "ja:en:ja"
    "ja:zh:ja"
    "ru:en:ru"
    "ru:zh:ru"
    "vi:en:vi"
    "zh:zh:ja"
    "zh:zh:ru"
    "fr:fr:it"
)
MONO_JOBS=("${MONO_JOBS_DEFAULT[@]}")
if [[ -f "$MONO_JOBS_FILE" ]]; then
    mapfile -t _mono_from_file < <(grep -v '^\s*#' "$MONO_JOBS_FILE" | sed '/^\s*$/d' | tr '[:upper:]' '[:lower:]')
    if (( ${#_mono_from_file[@]} > 0 )); then
        MONO_JOBS=("${_mono_from_file[@]}")
        echo "[INFO] Loaded ${#MONO_JOBS[@]} monolingual jobs from ${MONO_JOBS_FILE}" >&2
    else
        echo "[WARN] ${MONO_JOBS_FILE} exists but contained no jobs; using built-in monolingual defaults." >&2
    fi
fi

require_file() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] Missing required file: $path" >&2
        echo "        Use download_mmarco_queries.py to pull the needed queries." >&2
        exit 1
    fi
}

for lang in "${LANG_CODES[@]}"; do
    require_file "${QUERY_FILE_MAP[$lang]}"
done

require_file "$COMMON_QIDS"
require_file "$BILINGUAL_SCRIPT"
require_file "$MONO_SCRIPT"
require_file "$EVAL_SCRIPT"
require_file "$CACHE_SCRIPT"

declare -a normalize_pairs=()
for pair in "${BILINGUAL_PAIRS[@]}"; do
    IFS=':' read -r raw_a raw_b <<< "$pair"
    lang_a=${raw_a,,}
    lang_b=${raw_b,,}
    if [[ -z "${LANG_NAME_MAP[$lang_a]:-}" || -z "${LANG_NAME_MAP[$lang_b]:-}" ]]; then
        echo "[ERROR] Unsupported language code in pair '${pair}'." >&2
        exit 1
    fi
    normalize_pairs+=("${lang_a}:${lang_b}")
    require_file "${QUERY_FILE_MAP[$lang_a]}"
    require_file "${QUERY_FILE_MAP[$lang_b]}"
done
declare -a BILINGUAL_PAIRS=("${normalize_pairs[@]}")

declare -a normalized_mono_jobs=()
for job in "${MONO_JOBS[@]}"; do
    IFS=':' read -r raw_doc raw_a raw_b <<< "$job"
    doc_lang=${raw_doc,,}
    lang_a=${raw_a,,}
    lang_b=${raw_b,,}
    if [[ -z "$doc_lang" || -z "$lang_a" || -z "$lang_b" ]]; then
        echo "[ERROR] Bad monolingual job spec '${job}' (expected doc:langA:langB)" >&2
        exit 1
    fi
    if [[ -z "${LANG_NAME_MAP[$doc_lang]:-}" ]]; then
        echo "[ERROR] Unsupported document language code '${doc_lang}' in monolingual job '${job}'." >&2
        exit 1
    fi
    if [[ -z "${QUERY_FILE_MAP[$lang_a]:-}" || -z "${QUERY_FILE_MAP[$lang_b]:-}" ]]; then
        echo "[ERROR] Missing query TSV mapping for '${lang_a}' or '${lang_b}' in monolingual job '${job}'." >&2
        exit 1
    fi
    require_file "${QUERY_FILE_MAP[$lang_a]}"
    require_file "${QUERY_FILE_MAP[$lang_b]}"
    normalized_mono_jobs+=("${doc_lang}:${lang_a}:${lang_b}")
done
MONO_JOBS=("${normalized_mono_jobs[@]}")

declare -a JOB_QUEUE=()
for pair in "${BILINGUAL_PAIRS[@]}"; do
    IFS=':' read -r lang_a lang_b <<< "$pair"
    JOB_QUEUE+=("bilingual,${lang_a},${lang_b}")
done

for job in "${MONO_JOBS[@]}"; do
    IFS=':' read -r doc_lang lang_a lang_b <<< "$job"
    JOB_QUEUE+=("monolingual,${lang_a},${lang_b},${doc_lang}")
done

preencode_queries() {
    local device=${PREENCODE_DEVICE:-"cuda:0"}
    local batch=${PREENCODE_BATCH:-128}

    # pick a non-en partner to seed the first cache when english is missing
    local partner_for_en=""
    for code in "${LANG_CODES[@]}"; do
        if [[ "$code" != "en" ]]; then
            partner_for_en=$code
            break
        fi
    done
    if [[ -z "$partner_for_en" ]]; then
        echo "[ERROR] Could not find a partner language for English when pre-encoding queries." >&2
        exit 1
    fi

    echo "[INFO] Pre-encoding queries for all languages into ${QUERY_CACHE_ROOT}"
    for lang in "${LANG_CODES[@]}"; do
        local cache_file="${QUERY_CACHE_ROOT}/${lang}/queries.npz"
        if [[ -f "$cache_file" ]]; then
            echo "[INFO] Cache exists for ${lang} (${cache_file}); skipping."
            continue
        fi
        local partner="en"
        if [[ "$lang" == "en" ]]; then
            partner="$partner_for_en"
        fi
        echo "[INFO] Caching queries for ${lang} (paired with ${partner})"
        "$PYTHON_BIN" "$CACHE_SCRIPT" \
            --repo "$REPO" \
            --encoder "$ENCODER" \
            --device "$device" \
            --enc_batch "$batch" \
            --cache_root "$QUERY_CACHE_ROOT" \
            --query_tsv "${partner}=${QUERY_FILE_MAP[$partner]}" \
            --query_tsv "${lang}=${QUERY_FILE_MAP[$lang]}"
    done
}

declare -A GPU_CAPACITY=(
    [0]=${GPU0_SLOTS:-3}
    [1]=${GPU1_SLOTS:-3}
)
declare -A GPU_SLOT_USAGE=()
for gpu in "${GPUS[@]}"; do
    if [[ -z "${GPU_CAPACITY[$gpu]:-}" ]]; then
        GPU_CAPACITY[$gpu]=$DEFAULT_GPU_SLOTS
    fi
    GPU_SLOT_USAGE[$gpu]=0
done

declare -A PID_TO_GPU=()
declare -A PID_TO_DESC=()
declare -A PID_TO_LOG=()
RUNNING_JOBS=0

terminate_jobs() {
    for pid in "${!PID_TO_GPU[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap terminate_jobs SIGINT SIGTERM

gpu_has_capacity() {
    local gpu=$1
    local capacity=${GPU_CAPACITY[$gpu]:-$DEFAULT_GPU_SLOTS}
    local ours=${GPU_SLOT_USAGE[$gpu]:-0}
    if (( ours < capacity )); then
        return 0
    fi
    return 1
}

reap_finished_jobs() {
    for pid in "${!PID_TO_GPU[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        local status=0
        if ! wait "$pid"; then
            status=$?
        fi
        local desc=${PID_TO_DESC[$pid]}
        local log=${PID_TO_LOG[$pid]}
        local gpu=${PID_TO_GPU[$pid]}
        unset PID_TO_DESC["$pid"] PID_TO_LOG["$pid"] PID_TO_GPU["$pid"]
        if [[ ${GPU_SLOT_USAGE[$gpu]:-0} -gt 0 ]]; then
            GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] - 1))
        fi
        RUNNING_JOBS=$((RUNNING_JOBS - 1))
        if [[ $status -ne 0 ]]; then
            echo "[ERROR] Job '${desc}' failed with status ${status}. See ${log}" >&2
            terminate_jobs
            exit $status
        fi
        echo "[$(date '+%F %T')] Completed ${desc} (logs: ${log})" >&2
    done
}

wait_for_gpu_slot() {
    while true; do
        reap_finished_jobs
        for gpu in "${GPUS[@]}"; do
            if ! gpu_has_capacity "$gpu"; then
                continue
            fi
            echo "$gpu"
            return 0
        done
        echo "[$(date '+%F %T')] All GPU slots busy, retrying in ${SLEEP_INTERVAL}s..." >&2
        sleep "$SLEEP_INTERVAL"
    done
}

start_job() {
    local gpu=$1
    local desc=$2
    local log_file=$3
    shift 3
    (
        set -euo pipefail
        "$@"
    ) &> "$log_file" &
    local pid=$!
    PID_TO_GPU[$pid]=$gpu
    PID_TO_DESC[$pid]=$desc
    PID_TO_LOG[$pid]=$log_file
    GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] + 1))
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
    echo "[$(date '+%F %T')] Launched ${desc} on GPU ${gpu} (log: ${log_file})" >&2
}

run_bilingual_job() {
    local gpu=$1
    local lang_a=$2
    local lang_b=$3

    local doc_lang_a=${LANG_NAME_MAP[$lang_a]}
    local doc_lang_b=${LANG_NAME_MAP[$lang_b]}
    local doc_langs="${doc_lang_a},${doc_lang_b}"
    local lang_pair="${lang_a}-${lang_b}"
    local q_primary=${QUERY_FILE_MAP[$lang_a]}
    local q_secondary=${QUERY_FILE_MAP[$lang_b]}
    local exp_tag="bilingual-${lang_pair}"
    local rundir="${RUN_ROOT}/${DATASET}-${SIZE}-${exp_tag}-5bands-${ENC_TAG}/vector_mix"
    local docids_path="${rundir}/docids-${lang_pair}.txt"
    local result_dir="${RESULT_ROOT}/${DATASET}-${SIZE}-${exp_tag}-5bands-${ENC_TAG}/vector_mix"
    mkdir -p "$rundir" "$result_dir"

    echo "[$(date '+%F %T')] [GPU ${gpu}] Starting bilingual mix for ${lang_pair} (${doc_langs})"

    "$PYTHON_BIN" "$BILINGUAL_SCRIPT" \
        --repo "$REPO" \
        --qrels_repo "BeIR/msmarco-qrels" \
        --langs "$doc_langs" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --docids_out "$docids_path" \
        --trust_remote \
        --batch 1024 \
        --dtype fp16 \
        --max_docs "$SIZE" \
        --index_root "$INDEX_ROOT" \
        --query_tsv "${lang_a}=${q_primary}" \
        --query_tsv "${lang_b}=${q_secondary}" \
        --outdir "$rundir" \
        --cm_alphas "$CM_ALPHAS" \
        --cache_queries \
        --query_cache_dir "$QUERY_CACHE_ROOT"

    for band in "${BANDS[@]}"; do
        "$PYTHON_BIN" "$EVAL_SCRIPT" \
            --dataset "$DATASET" \
            --run "${rundir}/cm-alpha-${band}.trec" \
            --qrels_repo "BeIR/msmarco-qrels" \
            --qrels_split "$QRELS_SPLIT" \
            --outdir "$result_dir" \
            --trust_remote \
            --filter_docids "$docids_path" \
            --qrels_cache "$QRELS_CACHE" \
            --filter_qids "$COMMON_QIDS"
    done
}

run_monolingual_job() {
    local gpu=$1
    local lang_a=$2
    local lang_b=$3
    local doc_code=$4

    local doc_lang=${LANG_NAME_MAP[$doc_code]}
    local query_pair="${lang_a}-${lang_b}"
    local q_primary=${QUERY_FILE_MAP[$lang_a]}
    local q_secondary=${QUERY_FILE_MAP[$lang_b]}
    local rundir="${RUN_ROOT}/${DATASET}-${SIZE}-${doc_lang}-${query_pair}-5bands-${ENC_TAG}/vector_mix"
    local docids_path="${rundir}/docids-${query_pair}.txt"
    local result_dir="${RESULT_ROOT}/${DATASET}-${SIZE}-${doc_lang}-${query_pair}-5bands-${ENC_TAG}/vector_mix"
    mkdir -p "$rundir" "$result_dir"

    echo "[$(date '+%F %T')] [GPU ${gpu}] Starting monolingual mix for ${doc_lang} docs (${query_pair} queries)"

    "$PYTHON_BIN" "$MONO_SCRIPT" \
        --repo "$REPO" \
        --config "collection-${doc_lang}" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --run_out "$rundir" \
        --docids_out "$docids_path" \
        --trust_remote \
        --batch 1024 \
        --qblock 1024 \
        --dtype fp16 \
        --max_docs "$SIZE" \
        --index_root "$INDEX_ROOT" \
        --query_tsv "${lang_a}=${q_primary}" \
        --query_tsv "${lang_b}=${q_secondary}" \
        --cm_alphas "$CM_ALPHAS" \
        --cache_queries \
        --query_cache_dir "$QUERY_CACHE_ROOT"

    for band in "${BANDS[@]}"; do
        "$PYTHON_BIN" "$EVAL_SCRIPT" \
            --dataset "$DATASET" \
            --run "${rundir}/cm-alpha-${band}.trec" \
            --qrels_repo "BeIR/msmarco-qrels" \
            --qrels_split "$QRELS_SPLIT" \
            --outdir "$result_dir" \
            --trust_remote \
            --filter_docids "$docids_path" \
            --qrels_cache "$QRELS_CACHE" \
            --filter_qids "$COMMON_QIDS"
    done
}

launch_jobs() {
    for job in "${JOB_QUEUE[@]}"; do
        IFS=',' read -r job_type lang_a lang_b doc_code <<< "$job"
        local pair_slug="${lang_a}-${lang_b}"
        local log_tag
        local desc
        case "$job_type" in
            bilingual)
                log_tag="bilingual-${pair_slug}"
                desc="bilingual ${pair_slug}"
                ;;
            monolingual)
                log_tag="monolingual-${doc_code}-${pair_slug}"
                desc="monolingual ${doc_code} (${pair_slug})"
                ;;
            *)
                echo "[ERROR] Unknown job type '${job_type}'." >&2
                exit 1
                ;;
        esac
        local gpu
        gpu=$(wait_for_gpu_slot)
        local log_file="${LOG_DIR}/${log_tag}-$(date '+%Y%m%d_%H%M%S')-$$.log"
        if [[ "$job_type" == "bilingual" ]]; then
            start_job "$gpu" "$desc" "$log_file" run_bilingual_job "$gpu" "$lang_a" "$lang_b"
        else
            start_job "$gpu" "$desc" "$log_file" run_monolingual_job "$gpu" "$lang_a" "$lang_b" "$doc_code"
        fi
        if (( BETWEEN_LAUNCH_SLEEP > 0 )); then
            sleep "$BETWEEN_LAUNCH_SLEEP"
        fi
    done
}

# echo "[$(date '+%F %T')] Starting pre-encoding of queries..."
# preencode_queries
# echo "[$(date '+%F %T')] Finished pre-encoding queries."

launch_jobs
while (( RUNNING_JOBS > 0 )); do
    reap_finished_jobs
    if (( RUNNING_JOBS > 0 )); then
        sleep 60s
    fi
done

echo "[$(date '+%F %T')] All language-pair experiments completed."
