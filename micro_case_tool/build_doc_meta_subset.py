#!/usr/bin/env python3
"""Build hit-level doc metadata for micro-case inspector runs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from micro_case_common import (
    DEFAULT_MINE_OUT,
    LANG_NAME_TO_CODE,
    load_qid_list,
    log,
    parse_doc_mix_codes,
    read_csv_rows,
    warn,
    write_csv_rows,
)

DEFAULT_CASES = DEFAULT_MINE_OUT / "cases.csv"
DEFAULT_OUTPUT = DEFAULT_MINE_OUT / "doc_meta_subset.csv"
DEFAULT_STATS_OUTPUT = DEFAULT_MINE_OUT / "doc_meta_subset_stats.csv"
DEFAULT_TOP_K = 50
DEFAULT_RAW_RANK_DEPTH = 500
DEFAULT_SNIPPET_CHARS = 300
DEFAULT_HF_REPO = "unicamp-dl/mmarco"
DEFAULT_HF_SPLIT = "collection"
DOC_META_FIELDNAMES = (
    "case_id",
    "qid",
    "condition",
    "rank",
    "docid",
    "retrieval_score_raw",
    "lang",
    "lang_source",
    "title",
    "snippet",
)

CODE_TO_HF_LANG = {
    "am": "amharic",
    "ar": "arabic",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "km": "khmer",
    "ku": "kurdish",
    "my": "burmese",
    "ne": "nepali",
    "nl": "dutch",
    "pt": "portuguese",
    "ru": "russian",
    "shn": "shan",
    "si": "sinhala",
    "sl": "slovene",
    "sw": "swahili",
    "vi": "vietnamese",
    "zh": "chinese",
}

ID_FIELD_CANDIDATES = ("id", "docid", "doc_id", "_id")
TEXT_FIELD_CANDIDATES = ("text", "contents", "content", "body")
TITLE_FIELD_CANDIDATES = ("title", "headline")


@dataclass
class HitRow:
    case_id: str
    qid: str
    condition: str
    rank: int
    docid: str
    retrieval_score_raw: float
    lang: str = ""
    lang_source: str = ""
    title: str = ""
    snippet: str = ""


def normalize_lang_token(value: str) -> str:
    token = (value or "").strip().lower()
    if not token:
        return ""
    code = LANG_NAME_TO_CODE.get(token, token)
    return code if code in CODE_TO_HF_LANG else ""


def parse_codes_from_text(value: str) -> List[str]:
    out: List[str] = []
    for token in re.split(r"[+,&/\s-]+", value or ""):
        code = normalize_lang_token(token)
        if code and code not in out:
            out.append(code)
    return out


def parse_case_doc_codes(row: Mapping[str, str]) -> List[str]:
    codes = parse_doc_mix_codes(row.get("doc_mix", ""))
    if codes:
        return codes

    from_doc_lang = parse_codes_from_text(row.get("doc_lang", ""))
    if from_doc_lang:
        return from_doc_lang

    doc_index_id = (row.get("doc_index_id") or "").strip()
    parts = [p.strip().lower() for p in doc_index_id.split("-") if p.strip()]
    if len(parts) >= 3 and parts[2] != "bilingual":
        code = normalize_lang_token(parts[2])
        if code:
            return [code]
    return []


def infer_mono_case_lang(row: Mapping[str, str], case_doc_codes: Sequence[str]) -> str:
    if len(case_doc_codes) == 1:
        return case_doc_codes[0]

    from_doc_lang = parse_codes_from_text(row.get("doc_lang", ""))
    if len(from_doc_lang) == 1:
        return from_doc_lang[0]

    doc_index_id = (row.get("doc_index_id") or "").strip()
    parts = [p.strip().lower() for p in doc_index_id.split("-") if p.strip()]
    if len(parts) >= 3 and parts[2] != "bilingual":
        return normalize_lang_token(parts[2])
    return ""


def is_bilingual_case(row: Mapping[str, str], case_doc_codes: Sequence[str]) -> bool:
    doc_type = (row.get("doc_type") or "").strip().lower()
    if doc_type in {"bi", "bilingual"}:
        return True
    if doc_type in {"mono", "monolingual"}:
        return False
    if len(case_doc_codes) > 1:
        return True
    doc_mix = (row.get("doc_mix") or "").strip()
    doc_lang = (row.get("doc_lang") or "").strip()
    return ("+" in doc_mix) or ("+" in doc_lang)


def qid_sort_key(qid: str) -> Tuple[int, object]:
    text = (qid or "").strip()
    return (0, int(text)) if text.isdigit() else (1, text)


def condition_sort_key(cond: str) -> int:
    c = (cond or "").strip().lower()
    if c == "endpoint":
        return 0
    if c == "mixed":
        return 1
    return 2


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def make_snippet(value: object, snippet_chars: int) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if snippet_chars <= 0:
        return text
    return text[:snippet_chars]


def split_raw_docid(raw_docid: str) -> Tuple[str, str]:
    text = (raw_docid or "").strip()
    if not text:
        return "", ""
    if "#" in text:
        base, suffix = text.rsplit("#", 1)
        return base.strip(), normalize_lang_token(suffix)
    return text, ""


def raw_sibling_path(run_path: Path) -> Path:
    return run_path.with_name(f"{run_path.stem}_raw.trec")


def load_run_hits(
    case_id: str,
    condition: str,
    run_path: Path,
    top_k: int,
    allowed_qids: Optional[Set[str]] = None,
) -> List[HitRow]:
    out: List[HitRow] = []
    if not run_path.exists():
        warn(f"Run file not found: {run_path}")
        return out

    with run_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if len(parts) < 6:
                continue
            qid = parts[0].strip()
            docid = parts[2].strip()
            try:
                rank = int(parts[3])
                score = float(parts[4])
            except Exception:
                continue
            if rank > top_k:
                continue
            if not qid or not docid:
                continue
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            out.append(
                HitRow(
                    case_id=case_id,
                    qid=qid,
                    condition=condition,
                    rank=rank,
                    docid=docid,
                    retrieval_score_raw=score,
                )
            )
    return out


def build_raw_lang_lookup(
    raw_path: Path,
    raw_rank_depth: int,
) -> Tuple[Dict[Tuple[str, int, str], str], Dict[Tuple[str, str], Counter]]:
    rank_lang: Dict[Tuple[str, int, str], str] = {}
    qdoc_lang_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    seen_by_qid: Dict[str, Set[str]] = defaultdict(set)
    collapsed_rank_by_qid: Dict[str, int] = defaultdict(int)

    with raw_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if len(parts) < 6:
                continue

            qid = parts[0].strip()
            raw_docid = parts[2].strip()
            try:
                raw_rank = int(parts[3])
            except Exception:
                continue
            if raw_rank_depth > 0 and raw_rank > raw_rank_depth:
                continue

            base_docid, lang = split_raw_docid(raw_docid)
            if not qid or not base_docid or not lang:
                continue

            qdoc_lang_counts[(qid, base_docid)][lang] += 1
            if base_docid in seen_by_qid[qid]:
                continue

            seen_by_qid[qid].add(base_docid)
            collapsed_rank_by_qid[qid] += 1
            c_rank = collapsed_rank_by_qid[qid]
            rank_lang[(qid, c_rank, base_docid)] = lang

    return rank_lang, qdoc_lang_counts


def pick_lang_from_counter(counter: Counter) -> Tuple[str, str]:
    if not counter:
        return "", ""
    ranked = counter.most_common()
    if not ranked:
        return "", ""
    top_count = ranked[0][1]
    tied = [lang for lang, cnt in ranked if cnt == top_count]
    if len(tied) != 1:
        return "", "tie"
    return tied[0], "majority"


def load_docid_map_langs(docid_map_path: Path) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    if not docid_map_path.exists():
        return out

    with docid_map_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames:
            return out
        lower = {f.strip().lower(): f for f in reader.fieldnames}
        base_col = lower.get("base_id") or lower.get("base")
        lang_col = lower.get("lang") or lower.get("language")
        if not base_col or not lang_col:
            return out

        for row in reader:
            base_docid = (row.get(base_col) or "").strip()
            if not base_docid:
                continue
            lang = normalize_lang_token(row.get(lang_col, ""))
            if not lang:
                continue
            out[base_docid].add(lang)
    return out


def assign_langs_for_bilingual_hits(
    *,
    hits: List[HitRow],
    run_path: Path,
    raw_rank_depth: int,
) -> Counter:
    source_counts: Counter = Counter()
    raw_path = raw_sibling_path(run_path)
    rank_lang: Dict[Tuple[str, int, str], str] = {}
    qdoc_lang_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    if raw_path.exists():
        rank_lang, qdoc_lang_counts = build_raw_lang_lookup(raw_path, raw_rank_depth)
        source_counts["raw_found"] += 1
    else:
        source_counts["raw_missing"] += 1

    docid_map_path = run_path.parent / "docid_map.tsv"
    docid_map_langs: Optional[Dict[str, Set[str]]] = None

    for hit in hits:
        key_rank = (hit.qid, hit.rank, hit.docid)
        lang = rank_lang.get(key_rank, "")
        if lang:
            hit.lang = lang
            hit.lang_source = "raw_rank"
            source_counts["raw_rank"] += 1
            continue

        lang_counter = qdoc_lang_counts.get((hit.qid, hit.docid), Counter())
        picked, reason = pick_lang_from_counter(lang_counter)
        if picked:
            hit.lang = picked
            hit.lang_source = "raw_qid_majority"
            source_counts["raw_qid_majority"] += 1
            continue
        if reason == "tie":
            source_counts["raw_qid_tie"] += 1

        if docid_map_langs is None:
            docid_map_langs = load_docid_map_langs(docid_map_path)
            source_counts["docid_map_loaded"] += 1
        map_langs = sorted(docid_map_langs.get(hit.docid, set()))
        if len(map_langs) == 1:
            hit.lang = map_langs[0]
            hit.lang_source = "docid_map_unique"
            source_counts["docid_map_unique"] += 1
            continue
        if len(map_langs) > 1:
            hit.lang = ""
            hit.lang_source = "docid_map_ambiguous"
            source_counts["docid_map_ambiguous"] += 1
            continue

        hit.lang = ""
        hit.lang_source = "unresolved"
        source_counts["unresolved"] += 1

    return source_counts


def pick_row_field(row: Mapping[str, object], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(key).lower(): str(key) for key in row.keys()}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def fetch_corpus_pairs(
    *,
    lang_to_docids: Mapping[str, Set[str]],
    repo: str,
    split: str,
    snippet_chars: int,
    trust_remote_code: bool,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it first (e.g. `pip install datasets`) "
            "or run with --skip-corpus to generate language-only metadata."
        ) from exc

    out: Dict[Tuple[str, str], Dict[str, str]] = {}

    for lang_code in sorted(lang_to_docids.keys()):
        target = set(lang_to_docids.get(lang_code, set()))
        if not target:
            continue

        lang_name = CODE_TO_HF_LANG.get(lang_code)
        if not lang_name:
            warn(f"Unsupported language code for HF collection lookup: {lang_code}")
            continue

        cfg = f"collection-{lang_name}"
        log(f"Streaming {repo}/{cfg} for {len(target)} docids")
        try:
            stream = load_dataset(
                repo,
                cfg,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            warn(f"Failed to load {repo}/{cfg}: {exc}")
            continue

        remaining = set(target)
        scanned = 0
        found = 0
        id_field: Optional[str] = None
        text_field: Optional[str] = None
        title_field: Optional[str] = None

        for row in stream:
            scanned += 1
            if id_field is None:
                id_field = pick_row_field(row, ID_FIELD_CANDIDATES)
                text_field = pick_row_field(row, TEXT_FIELD_CANDIDATES)
                title_field = pick_row_field(row, TITLE_FIELD_CANDIDATES)
                if not id_field:
                    warn(f"Could not find id field in {repo}/{cfg}; skipping")
                    break

            docid = normalize_text(row.get(id_field, ""))
            if not docid or docid not in remaining:
                continue

            title = normalize_text(row.get(title_field, "")) if title_field else ""
            snippet = make_snippet(row.get(text_field, "") if text_field else "", snippet_chars)
            out[(docid, lang_code)] = {"title": title, "snippet": snippet}
            remaining.remove(docid)
            found += 1
            if not remaining:
                break

        log(f"  {cfg}: found {found}/{len(target)} (scanned={scanned})")
        if remaining:
            warn(f"  {cfg}: {len(remaining)} target docids still missing")

    return out


def summarize_hit_coverage(hits: Sequence[HitRow]) -> Dict[str, int]:
    total_hits = len(hits)
    lang_filled = sum(1 for h in hits if (h.lang or "").strip())
    snippet_filled = sum(1 for h in hits if (h.snippet or "").strip())
    resolved = sum(1 for h in hits if (h.lang or "").strip() and (h.snippet or "").strip())

    by_docid: Dict[str, Dict[str, bool]] = {}
    for h in hits:
        x = by_docid.setdefault(h.docid, {"lang": False, "snippet": False})
        x["lang"] = x["lang"] or bool((h.lang or "").strip())
        x["snippet"] = x["snippet"] or bool((h.snippet or "").strip())

    unique_docids = len(by_docid)
    unresolved_docids = sum(1 for v in by_docid.values() if not (v["lang"] and v["snippet"]))

    return {
        "total_hits": total_hits,
        "unique_docids": unique_docids,
        "lang_filled": lang_filled,
        "snippet_filled": snippet_filled,
        "resolved": resolved,
        "unresolved": total_hits - resolved,
        "unresolved_docids": unresolved_docids,
    }


def metadata_row_key(row: Mapping[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("case_id", "")).strip(),
        str(row.get("qid", "")).strip(),
        str(row.get("condition", "")).strip().lower(),
        str(row.get("rank", "")).strip(),
        str(row.get("docid", "")).strip(),
    )


def metadata_row_quality(row: Mapping[str, object]) -> int:
    return sum(
        1
        for field in ("lang", "title", "snippet")
        if str(row.get(field, "")).strip()
    )


def merge_metadata_rows(
    existing_rows: Sequence[Mapping[str, object]],
    new_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    for row in existing_rows:
        merged[metadata_row_key(row)] = {k: row.get(k, "") for k in DOC_META_FIELDNAMES}

    for row in new_rows:
        key = metadata_row_key(row)
        incoming = {k: row.get(k, "") for k in DOC_META_FIELDNAMES}
        current = merged.get(key)
        if current is None:
            merged[key] = incoming
            continue

        # Keep richer row when available, but still allow incoming non-empty fields to fill blanks.
        if metadata_row_quality(incoming) > metadata_row_quality(current):
            base = dict(incoming)
            other = current
        else:
            base = dict(current)
            other = incoming

        for field in DOC_META_FIELDNAMES:
            if str(base.get(field, "")).strip():
                continue
            candidate = other.get(field, "")
            if str(candidate).strip():
                base[field] = candidate
        merged[key] = base

    def sort_key(row: Mapping[str, object]) -> Tuple[str, Tuple[int, object], int, int, str]:
        rank_text = str(row.get("rank", "")).strip()
        try:
            rank_value = int(rank_text)
        except Exception:
            rank_value = 10**9
        return (
            str(row.get("case_id", "")),
            qid_sort_key(str(row.get("qid", ""))),
            condition_sort_key(str(row.get("condition", ""))),
            rank_value,
            str(row.get("docid", "")),
        )

    return sorted(merged.values(), key=sort_key)


def summarize_rows_coverage(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    total_hits = len(rows)
    lang_filled = sum(1 for row in rows if str(row.get("lang", "")).strip())
    snippet_filled = sum(1 for row in rows if str(row.get("snippet", "")).strip())
    resolved = sum(
        1
        for row in rows
        if str(row.get("lang", "")).strip() and str(row.get("snippet", "")).strip()
    )

    by_docid: Dict[str, Dict[str, bool]] = {}
    for row in rows:
        docid = str(row.get("docid", "")).strip()
        x = by_docid.setdefault(docid, {"lang": False, "snippet": False})
        x["lang"] = x["lang"] or bool(str(row.get("lang", "")).strip())
        x["snippet"] = x["snippet"] or bool(str(row.get("snippet", "")).strip())

    unique_docids = len(by_docid)
    unresolved_docids = sum(1 for v in by_docid.values() if not (v["lang"] and v["snippet"]))

    return {
        "total_hits": total_hits,
        "unique_docids": unique_docids,
        "lang_filled": lang_filled,
        "snippet_filled": snippet_filled,
        "resolved": resolved,
        "unresolved": total_hits - resolved,
        "unresolved_docids": unresolved_docids,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build hit-level doc metadata rows for inspector (all cases by default, or one --case-id)."
        )
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to cases.csv")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case-id", help="Optional single case_id to process")
    group.add_argument("--all-cases", action="store_true", help="Process all rows in --cases (default behavior)")
    parser.add_argument(
        "--qid-list",
        default="",
        help="Optional qid list (one qid per line) to restrict hit collection",
    )

    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k docs per query from each run")
    parser.add_argument(
        "--raw-rank-depth",
        type=int,
        default=DEFAULT_RAW_RANK_DEPTH,
        help="Max raw rank scanned from *_raw.trec for bilingual language recovery (<=0 scans all)",
    )
    parser.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS, help="Snippet length cap")
    parser.add_argument("--repo", default=DEFAULT_HF_REPO, help="HF dataset repo")
    parser.add_argument("--split", default=DEFAULT_HF_SPLIT, help="HF dataset split")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV (case_id,qid,condition,rank,docid,retrieval_score_raw,lang,lang_source,title,snippet)",
    )
    parser.add_argument("--stats-out", default=str(DEFAULT_STATS_OUTPUT), help="Coverage stats CSV")
    parser.add_argument("--skip-corpus", action="store_true", help="Skip HF text/title fetch")
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable datasets trust_remote_code",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge new rows into existing --out file (if present) instead of replacing it",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise SystemExit(f"Cases CSV not found: {cases_path}")

    rows = read_csv_rows(cases_path)
    if not rows:
        raise SystemExit(f"Cases CSV is empty: {cases_path}")

    if args.case_id:
        selected_rows = [r for r in rows if (r.get("case_id") or "").strip() == args.case_id]
        if not selected_rows:
            raise SystemExit(f"case_id not found in {cases_path}: {args.case_id}")
    else:
        selected_rows = rows

    allowed_qids: Optional[Set[str]] = None
    qid_list_text = str(args.qid_list).strip()
    if qid_list_text:
        allowed_qids = load_qid_list(Path(qid_list_text))
        if not allowed_qids:
            raise SystemExit(f"QID list is empty: {qid_list_text}")
        log(f"Applying qid filter: {len(allowed_qids)} qids from {qid_list_text}")

    case_hits: Dict[str, List[HitRow]] = {}
    case_doc_codes: Dict[str, List[str]] = {}
    case_is_bilingual: Dict[str, bool] = {}

    for row in selected_rows:
        case_id = (row.get("case_id") or "").strip()
        if not case_id:
            continue

        endpoint_path = Path((row.get("endpoint_run_path") or "").strip())
        mixed_path = Path((row.get("mixed_run_path") or "").strip())
        hits_endpoint = load_run_hits(
            case_id,
            "endpoint",
            endpoint_path,
            args.top_k,
            allowed_qids=allowed_qids,
        )
        hits_mixed = load_run_hits(
            case_id,
            "mixed",
            mixed_path,
            args.top_k,
            allowed_qids=allowed_qids,
        )
        hits = hits_endpoint + hits_mixed
        case_hits[case_id] = hits

        codes = parse_case_doc_codes(row)
        case_doc_codes[case_id] = codes
        case_is_bilingual[case_id] = is_bilingual_case(row, codes)

        log(
            f"{case_id}: collected hits endpoint={len(hits_endpoint)}, mixed={len(hits_mixed)}, "
            f"total={len(hits)}, unique_docids={len({h.docid for h in hits})}"
        )

        if case_is_bilingual[case_id]:
            src_endpoint = assign_langs_for_bilingual_hits(
                hits=hits_endpoint,
                run_path=endpoint_path,
                raw_rank_depth=args.raw_rank_depth,
            )
            src_mixed = assign_langs_for_bilingual_hits(
                hits=hits_mixed,
                run_path=mixed_path,
                raw_rank_depth=args.raw_rank_depth,
            )
            total_src = src_endpoint + src_mixed
            log(
                f"{case_id}: bilingual lang sources -> "
                + ", ".join(f"{k}:{v}" for k, v in sorted(total_src.items()))
            )
        else:
            mono_lang = infer_mono_case_lang(row, codes)
            if not mono_lang:
                warn(f"{case_id}: mono language could not be inferred")
            for hit in hits:
                hit.lang = mono_lang
                hit.lang_source = "mono_case" if mono_lang else "unresolved"
            log(f"{case_id}: mono lang assignment -> lang={mono_lang or 'blank'} hits={len(hits)}")

        lang_counts = Counter(h.lang for h in hits if h.lang)
        blank = sum(1 for h in hits if not h.lang)
        log(
            f"{case_id}: picked languages -> "
            + (", ".join(f"{k}:{v}" for k, v in sorted(lang_counts.items())) if lang_counts else "none")
            + f"; blank={blank}"
        )

    all_hits: List[HitRow] = [h for case_id in sorted(case_hits.keys()) for h in case_hits[case_id]]
    if not all_hits:
        raise SystemExit("No hits collected from selected cases.")

    lang_to_docids: Dict[str, Set[str]] = defaultdict(set)
    for case_id, hits in case_hits.items():
        codes = case_doc_codes.get(case_id, [])
        for hit in hits:
            if hit.lang:
                lang_to_docids[hit.lang].add(hit.docid)
            else:
                for code in codes:
                    lang_to_docids[code].add(hit.docid)

    corpus_pairs: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not args.skip_corpus:
        corpus_pairs = fetch_corpus_pairs(
            lang_to_docids=lang_to_docids,
            repo=args.repo,
            split=args.split,
            snippet_chars=args.snippet_chars,
            trust_remote_code=(not args.no_trust_remote_code),
        )
    else:
        warn("Skipping corpus fetch (--skip-corpus); snippets/titles remain empty")

    for case_id, hits in case_hits.items():
        codes = case_doc_codes.get(case_id, [])
        for hit in hits:
            if hit.lang:
                payload = corpus_pairs.get((hit.docid, hit.lang), {})
                hit.title = payload.get("title", "")
                hit.snippet = payload.get("snippet", "")
                continue

            chosen: Dict[str, str] = {}
            for code in codes:
                cand = corpus_pairs.get((hit.docid, code), {})
                if not cand:
                    continue
                if (cand.get("snippet") or "").strip():
                    chosen = cand
                    break
                if not chosen:
                    chosen = cand
            if chosen:
                hit.title = chosen.get("title", "")
                hit.snippet = chosen.get("snippet", "")

    rows_out: List[Dict[str, object]] = []
    for hit in sorted(
        all_hits,
        key=lambda h: (
            h.case_id,
            qid_sort_key(h.qid),
            condition_sort_key(h.condition),
            h.rank,
            h.docid,
        ),
    ):
        rows_out.append(
            {
                "case_id": hit.case_id,
                "qid": hit.qid,
                "condition": hit.condition,
                "rank": hit.rank,
                "docid": hit.docid,
                "retrieval_score_raw": hit.retrieval_score_raw,
                "lang": hit.lang,
                "lang_source": hit.lang_source,
                "title": hit.title,
                "snippet": hit.snippet,
            }
        )

    out_path = Path(args.out)
    rows_to_write = rows_out
    if args.merge_existing:
        if out_path.exists():
            existing_rows = read_csv_rows(out_path)
            rows_to_write = merge_metadata_rows(existing_rows, rows_out)
            log(
                f"Merged existing metadata from {out_path}: old={len(existing_rows)}, "
                f"new={len(rows_out)}, merged={len(rows_to_write)}"
            )
        else:
            log(f"--merge-existing enabled but no existing file at {out_path}; writing new rows only")

    write_csv_rows(
        out_path,
        rows_to_write,
        fieldnames=DOC_META_FIELDNAMES,
    )
    log(f"Wrote metadata subset: {out_path} (rows={len(rows_to_write)})")

    stats_rows: List[Dict[str, object]] = []
    global_stats = summarize_rows_coverage(rows_to_write)
    stats_rows.append({"scope": "global", "case_id": "ALL", **global_stats})

    rows_by_case: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows_to_write:
        case_id = str(row.get("case_id", "")).strip()
        rows_by_case[case_id].append(row)

    for case_id in sorted(rows_by_case.keys()):
        stats = summarize_rows_coverage(rows_by_case[case_id])
        stats_rows.append({"scope": "case", "case_id": case_id, **stats})

    stats_path = Path(args.stats_out)
    write_csv_rows(
        stats_path,
        stats_rows,
        fieldnames=(
            "scope",
            "case_id",
            "total_hits",
            "unique_docids",
            "lang_filled",
            "snippet_filled",
            "resolved",
            "unresolved",
            "unresolved_docids",
        ),
    )
    log(f"Wrote coverage stats: {stats_path} (rows={len(stats_rows)})")
    log(
        "Global unresolved docids: "
        f"{int(global_stats['unresolved_docids'])}/{int(global_stats['unique_docids'])}"
    )
    script_dir = Path(__file__).resolve().parent
    log(
        "Inspector usage: "
        f"python {script_dir}/case_inspector.py --cases {cases_path} "
        f"--case-id <CASE_ID> --doc-meta {out_path}"
    )


if __name__ == "__main__":
    main()
