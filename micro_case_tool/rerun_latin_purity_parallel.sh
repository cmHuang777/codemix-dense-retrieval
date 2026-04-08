#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_SH=${CONDA_SH:-/home/hcming/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-ir-lab-3}

CASES_CSV=${1:-${SCRIPT_DIR}/micro_cases/cases_best.csv}
DOC_PURITY_OUT=${2:-${SCRIPT_DIR}/micro_cases/doc_purity_features_cases_best.csv}
REPORT_OUT_DIR=${3:-${SCRIPT_DIR}/micro_reports_purity/best}
SUMMARY_OUT=${4:-${REPORT_OUT_DIR}/purity_summary_by_case.csv}

MAX_JOBS=${MAX_JOBS:-4}
TMP_ROOT=${TMP_ROOT:-logs/latin_purity_rerun_$$}
KEEP_TMP=${KEEP_TMP:-1}

if [[ ! -f "${CASES_CSV}" ]]; then
    echo "[ERROR] Cases CSV not found: ${CASES_CSV}" >&2
    exit 1
fi

mkdir -p "${TMP_ROOT}/subsets" "${TMP_ROOT}/build" "${TMP_ROOT}/logs"

cleanup() {
    if [[ "${KEEP_TMP}" == "1" ]]; then
        echo "[INFO] Keeping temp directory: ${TMP_ROOT}" >&2
        return 0
    fi
    rm -rf "${TMP_ROOT}"
}
trap cleanup EXIT

echo "[INFO] Using temp directory: ${TMP_ROOT}" >&2

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

python - "${CASES_CSV}" "${TMP_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

latin = {"de", "en", "es", "fr", "id", "it", "nl", "pt", "sl", "sw", "vi"}
cases_csv = Path(sys.argv[1])
tmp_root = Path(sys.argv[2])
subset_dir = tmp_root / "subsets"
subset_dir.mkdir(parents=True, exist_ok=True)

with cases_csv.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    raise SystemExit(f"Cases CSV is empty: {cases_csv}")

fieldnames = list(rows[0].keys())
latin_rows = []
latin_items = []
for row in rows:
    case_id = (row.get("case_id") or "").strip()
    doc_lang = (row.get("doc_lang") or "").strip().lower()
    parts = tuple(sorted({p.strip() for p in doc_lang.split("+") if p.strip()}))
    if not parts or any(p not in latin for p in parts):
        continue
    latin_rows.append(row)
    latin_items.append((row, parts))

if not latin_rows:
    raise SystemExit("No Latin-script document-language cases found in the selected cases CSV.")

latin_cases_path = tmp_root / "cases_latin_subset.csv"
with latin_cases_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(latin_rows)

manifest_path = tmp_root / "groups.tsv"
groups = []
remaining = list(latin_items)
while remaining:
    seed_row, seed_parts = remaining.pop(0)
    group_rows = [seed_row]
    group_langs = set(seed_parts)
    changed = True
    while changed:
        changed = False
        next_remaining = []
        for row, parts in remaining:
            part_set = set(parts)
            if group_langs & part_set:
                group_rows.append(row)
                group_langs.update(part_set)
                changed = True
            else:
                next_remaining.append((row, parts))
        remaining = next_remaining
    groups.append((tuple(sorted(group_langs)), group_rows))

with manifest_path.open("w", encoding="utf-8") as fh:
    for langs, group_rows in sorted(groups, key=lambda item: item[0]):
        safe_name = "_".join(langs)
        doc_lang = "+".join(langs)
        subset_path = subset_dir / f"cases_{safe_name}.csv"
        with subset_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_rows)
        case_ids = ",".join((row.get("case_id") or "").strip() for row in group_rows)
        fh.write(f"{safe_name}\t{doc_lang}\t{subset_path}\t{case_ids}\n")

print(f"[INFO] Wrote Latin subset cases: {latin_cases_path}")
print(f"[INFO] Wrote group manifest: {manifest_path}")
for langs, group_rows in sorted(groups, key=lambda item: item[0]):
    doc_lang = "+".join(langs)
    case_ids = ",".join((row.get('case_id') or '').strip() for row in group_rows)
    print(f"[INFO] Group {doc_lang}: {case_ids}")
PY

declare -a ACTIVE_PIDS=()
declare -A PID_TO_GROUP=()

reap_one() {
    local pid
    pid=$(wait -n)
}

while IFS=$'\t' read -r safe_name doc_lang subset_path case_ids; do
    [[ -n "${safe_name}" ]] || continue
    log_path="${TMP_ROOT}/logs/build_${safe_name}.log"
    out_path="${TMP_ROOT}/build/doc_purity_${safe_name}.csv"
    echo "[INFO] Launching ${doc_lang}: ${case_ids}" >&2
    (
        set -euo pipefail
        python "${SCRIPT_DIR}/build_doc_purity_features.py" \
            --cases "${subset_path}" \
            --all-cases \
            --out "${out_path}"
    ) > "${log_path}" 2>&1 &
    pid=$!
    ACTIVE_PIDS+=("${pid}")
    PID_TO_GROUP["${pid}"]="${doc_lang}"

    while (( ${#ACTIVE_PIDS[@]} >= MAX_JOBS )); do
        finished_pid=$(wait -n) || {
            status=$?
            echo "[ERROR] A build job failed (status=${status})." >&2
            for p in "${ACTIVE_PIDS[@]}"; do
                if kill -0 "${p}" 2>/dev/null; then
                    kill "${p}" 2>/dev/null || true
                fi
            done
            exit "${status}"
        }
        next_active=()
        for p in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "${p}" 2>/dev/null; then
                next_active+=("${p}")
            fi
        done
        ACTIVE_PIDS=("${next_active[@]}")
    done
done < "${TMP_ROOT}/groups.tsv"

for pid in "${ACTIVE_PIDS[@]}"; do
    wait "${pid}"
done

python - "${DOC_PURITY_OUT}" "${TMP_ROOT}/build" <<'PY'
import sys
from pathlib import Path

from build_doc_purity_features import merge_purity_rows
from micro_case_common import DOC_PURITY_FIELDNAMES, read_csv_rows, write_csv_rows

out_path = Path(sys.argv[1])
build_dir = Path(sys.argv[2])
rows = read_csv_rows(out_path) if out_path.exists() else []
for path in sorted(build_dir.glob("doc_purity_*.csv")):
    incoming = read_csv_rows(path)
    rows = merge_purity_rows(rows, incoming)
write_csv_rows(out_path, rows, fieldnames=DOC_PURITY_FIELDNAMES)
print(f"[INFO] Wrote merged doc purity CSV: {out_path} (rows={len(rows)})")
PY

mkdir -p "${REPORT_OUT_DIR}"

LATIN_CASES_CSV="${TMP_ROOT}/cases_latin_subset.csv"
LATIN_SUMMARY_TMP="${TMP_ROOT}/purity_summary_latin.csv"

python "${SCRIPT_DIR}/analyze_purity_effect.py" \
    --cases "${LATIN_CASES_CSV}" \
    --all-cases \
    --doc-purity "${DOC_PURITY_OUT}" \
    --out-dir "${REPORT_OUT_DIR}" \
    --summary-out "${LATIN_SUMMARY_TMP}"

python - "${SUMMARY_OUT}" "${LATIN_SUMMARY_TMP}" <<'PY'
import sys
from pathlib import Path

from analyze_purity_effect import SUMMARY_FIELDNAMES
from micro_case_common import read_csv_rows, write_csv_rows

summary_out = Path(sys.argv[1])
latin_summary = Path(sys.argv[2])

def key(row):
    return (
        str(row.get("case_id", "")).strip(),
        str(row.get("pair", "")).strip(),
        str(row.get("doc_lang", "")).strip(),
        str(row.get("rel_bucket", "")).strip(),
        str(row.get("query_quality_label", "")).strip(),
    )

merged = {}
if summary_out.exists():
    for row in read_csv_rows(summary_out):
        merged[key(row)] = {k: row.get(k, "") for k in SUMMARY_FIELDNAMES}
for row in read_csv_rows(latin_summary):
    merged[key(row)] = {k: row.get(k, "") for k in SUMMARY_FIELDNAMES}

rows = sorted(
    merged.values(),
    key=lambda row: (
        row.get("case_id", ""),
        row.get("rel_bucket", ""),
        row.get("query_quality_label", ""),
    ),
)
write_csv_rows(summary_out, rows, fieldnames=SUMMARY_FIELDNAMES)
print(f"[INFO] Wrote merged summary CSV: {summary_out} (rows={len(rows)})")
PY

echo "[INFO] Latin-only purity rerun complete." >&2
echo "[INFO] Updated doc purity CSV: ${DOC_PURITY_OUT}" >&2
echo "[INFO] Updated report dir: ${REPORT_OUT_DIR}" >&2
echo "[INFO] Updated summary CSV: ${SUMMARY_OUT}" >&2
