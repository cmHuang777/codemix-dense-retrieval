#!/usr/bin/env python3
"""Build doc-level purity features for micro-case analysis."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from build_doc_meta_subset import (
    CODE_TO_HF_LANG,
    DEFAULT_CASES,
    DEFAULT_HF_REPO,
    DEFAULT_HF_SPLIT,
    DEFAULT_SNIPPET_CHARS,
    ID_FIELD_CANDIDATES,
    TEXT_FIELD_CANDIDATES,
    TITLE_FIELD_CANDIDATES,
    assign_langs_for_bilingual_hits,
    infer_mono_case_lang,
    is_bilingual_case,
    load_run_hits,
    make_snippet,
    normalize_text,
    parse_case_doc_codes,
    pick_row_field,
)
from micro_case_common import (
    DEFAULT_MINE_OUT,
    DEFAULT_DOC_PURITY_OUT,
    DEFAULT_QRELS,
    DOC_PURITY_FIELDNAMES,
    classify_doc_purity,
    load_qid_list,
    load_qrels,
    log,
    read_csv_rows,
    warn,
    write_csv_rows,
)

DEFAULT_DOC_META_FALLBACK = DEFAULT_MINE_OUT / "doc_meta_subset.csv"


def _add_doc_lang_candidates(
    table: MutableMapping[str, MutableMapping[str, Set[str]]],
    case_id: str,
    docid: str,
    langs: Sequence[str],
) -> None:
    if not case_id or not docid:
        return
    bucket = table[case_id][docid]
    for lang in langs:
        lang_text = (lang or "").strip().lower()
        if lang_text:
            bucket.add(lang_text)



def fetch_corpus_payloads(
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
            "to build doc purity features."
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
        id_field = None
        text_field = None
        title_field = None

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
            body = normalize_text(row.get(text_field, "")) if text_field else ""
            snippet = make_snippet(body, snippet_chars)
            out[(docid, lang_code)] = {
                "title": title,
                "body": body,
                "snippet": snippet,
                "text_source": "title+body" if body else ("title_only" if title else "missing"),
            }
            remaining.remove(docid)
            found += 1
            if not remaining:
                break

        log(f"  {cfg}: found {found}/{len(target)} (scanned={scanned})")
        if remaining:
            warn(f"  {cfg}: {len(remaining)} target docids still missing")

    return out



def doc_meta_row_quality(payload: Mapping[str, str]) -> int:
    return len((payload.get('title') or '').strip()) + len((payload.get('snippet') or '').strip())


def load_doc_meta_payloads(
    path: Path,
    allowed_case_ids: Optional[Set[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.exists():
        warn(f"Doc metadata fallback file not found: {path}")
        return {}

    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in read_csv_rows(path):
        case_id = str(row.get('case_id', '')).strip()
        docid = str(row.get('docid', '')).strip()
        if not case_id or not docid:
            continue
        if allowed_case_ids is not None and case_id not in allowed_case_ids:
            continue
        payload = {
            'lang': str(row.get('lang', '')).strip().lower(),
            'title': str(row.get('title', '')).strip(),
            'body': '',
            'snippet': str(row.get('snippet', '')).strip(),
            'text_source': 'title+snippet',
        }
        key = (case_id, docid)
        current = out.get(key)
        if current is None or doc_meta_row_quality(payload) >= doc_meta_row_quality(current):
            out[key] = payload
    log(f"Loaded doc-meta fallback payloads: {len(out)} docs from {path}")
    return out


def pick_payload_for_doc(
    docid: str,
    lang_candidates: Sequence[str],
    payloads: Mapping[Tuple[str, str], Mapping[str, str]],
) -> Tuple[str, Mapping[str, str]]:
    options = []
    for lang in lang_candidates:
        payload = payloads.get((docid, lang), {})
        if payload:
            score = (
                1 if (payload.get("body") or "").strip() else 0,
                1 if (payload.get("snippet") or "").strip() else 0,
                1 if (payload.get("title") or "").strip() else 0,
            )
            options.append((score, lang, payload))
    if options:
        options.sort(key=lambda item: (item[0], item[1]))
        _, lang, payload = options[-1]
        return lang, payload
    if len(lang_candidates) == 1:
        return lang_candidates[0], {}
    return "", {}



def purity_row_key(row: Mapping[str, object]) -> Tuple[str, str]:
    return (str(row.get("case_id", "")).strip(), str(row.get("docid", "")).strip())



def purity_row_quality(row: Mapping[str, object]) -> int:
    try:
        return int(float(row.get("text_chars", 0) or 0))
    except Exception:
        return 0



def merge_purity_rows(existing_rows, new_rows):
    merged = {
        purity_row_key(row): {k: row.get(k, "") for k in DOC_PURITY_FIELDNAMES}
        for row in existing_rows
    }
    for row in new_rows:
        key = purity_row_key(row)
        incoming = {k: row.get(k, "") for k in DOC_PURITY_FIELDNAMES}
        current = merged.get(key)
        if current is None or purity_row_quality(incoming) >= purity_row_quality(current):
            merged[key] = incoming
    return sorted(merged.values(), key=lambda row: (row.get("case_id", ""), row.get("docid", "")))



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build doc-level purity features for selected micro cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to cases.csv or cases_best.csv")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case-id", help="Optional single case_id to process")
    group.add_argument("--all-cases", action="store_true", help="Process all rows in --cases (default behavior)")
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS), help="Qrels TSV used to include relevant docs")
    parser.add_argument(
        "--qid-list",
        default="",
        help="Optional qid list (one qid per line) to restrict run hits and qrel expansion",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k docs per query from each run")
    parser.add_argument("--repo", default=DEFAULT_HF_REPO, help="HF dataset repo")
    parser.add_argument("--split", default=DEFAULT_HF_SPLIT, help="HF dataset split")
    parser.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS, help="Snippet length cap")
    parser.add_argument(
        "--hits-only",
        action="store_true",
        help="Only build purity features for run-hit docids; do not expand with qrel docids",
    )
    parser.add_argument(
        "--doc-meta",
        default=str(DEFAULT_DOC_META_FALLBACK),
        help="Optional existing doc_meta_subset.csv used as a snippet fallback",
    )
    parser.add_argument(
        "--use-doc-meta-only",
        action="store_true",
        help="Skip streamed collection fetch and build purity features from doc_meta snippets only",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_DOC_PURITY_OUT),
        help="Output CSV for doc purity features",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge into an existing --out file instead of replacing it",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable datasets trust_remote_code",
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
    if str(args.qid_list).strip():
        allowed_qids = load_qid_list(Path(args.qid_list))
        if not allowed_qids:
            raise SystemExit(f"QID list is empty: {args.qid_list}")
        log(f"Applying qid filter: {len(allowed_qids)} qids from {args.qid_list}")

    qrels = {} if args.hits_only else load_qrels(Path(args.qrels))
    selected_case_ids = {(row.get('case_id') or '').strip() for row in selected_rows if (row.get('case_id') or '').strip()}
    doc_meta_payloads = (
        load_doc_meta_payloads(Path(args.doc_meta), allowed_case_ids=selected_case_ids)
        if str(args.doc_meta).strip()
        else {}
    )
    case_doc_langs: Dict[str, MutableMapping[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for row in selected_rows:
        case_id = (row.get("case_id") or "").strip()
        if not case_id:
            continue

        endpoint_path = Path((row.get("endpoint_run_path") or "").strip())
        mixed_path = Path((row.get("mixed_run_path") or "").strip())
        hits_endpoint = load_run_hits(case_id, "endpoint", endpoint_path, args.top_k, allowed_qids=allowed_qids)
        hits_mixed = load_run_hits(case_id, "mixed", mixed_path, args.top_k, allowed_qids=allowed_qids)
        hits = hits_endpoint + hits_mixed

        codes = parse_case_doc_codes(row)
        bilingual = is_bilingual_case(row, codes)
        mono_lang = infer_mono_case_lang(row, codes)
        if bilingual:
            assign_langs_for_bilingual_hits(hits=hits_endpoint, run_path=endpoint_path, raw_rank_depth=500)
            assign_langs_for_bilingual_hits(hits=hits_mixed, run_path=mixed_path, raw_rank_depth=500)
        else:
            for hit in hits:
                hit.lang = mono_lang

        run_docids = set()
        qids = set()
        for hit in hits:
            run_docids.add(hit.docid)
            qids.add(hit.qid)
            if hit.lang:
                _add_doc_lang_candidates(case_doc_langs, case_id, hit.docid, [hit.lang])
            else:
                _add_doc_lang_candidates(case_doc_langs, case_id, hit.docid, codes)

        qrel_docids = set()
        if not args.hits_only:
            for qid in sorted(qids):
                for docid, rel in qrels.get(qid, {}).items():
                    if rel <= 0:
                        continue
                    qrel_docids.add(docid)
                    if mono_lang:
                        _add_doc_lang_candidates(case_doc_langs, case_id, docid, [mono_lang])
                    else:
                        _add_doc_lang_candidates(case_doc_langs, case_id, docid, codes)

        log(
            f"{case_id}: docids from runs={len(run_docids)}, qrel_expansion={len(qrel_docids)}, "
            f"unique_total={len(case_doc_langs[case_id])}"
        )

    lang_to_docids: Dict[str, Set[str]] = defaultdict(set)
    for docs in case_doc_langs.values():
        for docid, langs in docs.items():
            for lang in langs:
                if lang:
                    lang_to_docids[lang].add(docid)

    payloads: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not args.use_doc_meta_only:
        try:
            payloads = fetch_corpus_payloads(
                lang_to_docids=lang_to_docids,
                repo=args.repo,
                split=args.split,
                snippet_chars=args.snippet_chars,
                trust_remote_code=(not args.no_trust_remote_code),
            )
        except SystemExit as exc:
            if doc_meta_payloads:
                warn(f"{exc} Falling back to --doc-meta payloads only.")
                payloads = {}
            else:
                raise

    rows_out = []
    label_counter = Counter()
    for case_id in sorted(case_doc_langs.keys()):
        for docid in sorted(case_doc_langs[case_id].keys()):
            lang_candidates = sorted(case_doc_langs[case_id][docid])
            lang, payload = pick_payload_for_doc(docid, lang_candidates, payloads)
            fallback_payload = doc_meta_payloads.get((case_id, docid), {})
            if fallback_payload and (not payload or not ((payload.get("body") or "").strip() or (payload.get("snippet") or "").strip())):
                payload = fallback_payload
            if fallback_payload and not lang:
                lang = str(fallback_payload.get("lang", "")).strip().lower()
            title = str(payload.get("title", ""))
            body = str(payload.get("body", "") or payload.get("snippet", ""))
            features = classify_doc_purity(
                title=title,
                body=body,
                lang=lang,
                text_source=str(payload.get("text_source", "missing")),
            )
            row = {
                "case_id": case_id,
                "docid": docid,
                "doc_lang": lang,
                **features,
            }
            rows_out.append(row)
            label_counter[str(row.get("doc_purity_label", ""))] += 1

    out_path = Path(args.out)
    rows_to_write = rows_out
    if args.merge_existing and out_path.exists():
        existing_rows = read_csv_rows(out_path)
        rows_to_write = merge_purity_rows(existing_rows, rows_out)
        log(
            f"Merged existing purity rows from {out_path}: old={len(existing_rows)}, "
            f"new={len(rows_out)}, merged={len(rows_to_write)}"
        )

    write_csv_rows(out_path, rows_to_write, fieldnames=DOC_PURITY_FIELDNAMES)
    log(f"Wrote doc purity features: {out_path} (rows={len(rows_to_write)})")
    if label_counter:
        log("Doc purity label counts: " + ", ".join(f"{k}:{v}" for k, v in sorted(label_counter.items())))


if __name__ == "__main__":
    main()
