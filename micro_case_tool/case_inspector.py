#!/usr/bin/env python3
"""Stage B: inspect one mined case at per-query and top-doc level."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from micro_case_common import (
    DEFAULT_FAILURE_LABEL_CONFIG,
    DEFAULT_INSPECT_CONTROL_N,
    DEFAULT_INSPECT_K,
    DEFAULT_INSPECT_RANK_DEPTH,
    DEFAULT_INSPECT_OUT,
    DEFAULT_INSPECT_WORST_N,
    DEFAULT_QRELS,
    DEFAULT_QUERIES_DIR,
    DEFAULT_QUERY_CACHE_ROOT,
    DEFAULT_REPORT_DIFF_BLOCKS,
    DEFAULT_REPORT_TOP_WORST,
    FAILURE_LABEL_ORDER,
    FailureLabelConfig,
    REPORT_DELTA_QUANTILES,
    REPORT_DOC_LANG_CELL_LIMIT,
    REPORT_QUERY_TEXT_CELL_LIMIT,
    assign_failure_label,
    count_word_tokens,
    fmt,
    geometry_features,
    is_finite,
    load_embeddings,
    load_qid_list,
    load_qrels,
    load_queries_tsv,
    load_trec_run,
    log,
    ndcg_at_k,
    normalize_doc_lang,
    parse_doc_mix_codes,
    parse_float,
    percentile,
    rank_text,
    read_csv_rows,
    read_doc_meta_subset,
    recall_at_k,
    shift_with_inf,
    snippet_ascii_ratio,
    to_lambda,
    to_markdown_cell,
    top_entries,
    write_csv_rows,
    first_rel_rank,
    FAILURE_LABEL_UNCLASSIFIED,
    LANG_NAME_TO_CODE,
)


def _normalize_condition(value: str) -> str:
    text = (value or "").strip().lower()
    if text in {"end", "endpoint"}:
        return "endpoint"
    if text in {"mix", "mixed"}:
        return "mixed"
    return text


def _lookup_doc_meta(
    *,
    qid: str,
    condition: str,
    rank: int,
    docid: str,
    by_hit: Mapping[Tuple[str, str, int, str], Mapping[str, str]],
    by_doc: Mapping[Tuple[str, str, str], Mapping[str, str]],
) -> Mapping[str, str]:
    cond = _normalize_condition(condition)
    payload = by_hit.get((qid, cond, int(rank), docid))
    if payload:
        return payload
    return by_doc.get((qid, cond, docid), {})


def _mismatch_rate_for_entries(
    entries: Sequence[Tuple[int, str, float]],
    *,
    qid: str,
    condition: str,
    by_hit: Mapping[Tuple[str, str, int, str], Mapping[str, str]],
    by_doc: Mapping[Tuple[str, str, str], Mapping[str, str]],
    expected: Sequence[str],
) -> float:
    expected_set = {e for e in expected if e}
    if not expected_set:
        return float("nan")
    langs: List[str] = []
    for rank, docid, _ in entries:
        meta = _lookup_doc_meta(
            qid=qid,
            condition=condition,
            rank=rank,
            docid=docid,
            by_hit=by_hit,
            by_doc=by_doc,
        )
        lang = normalize_doc_lang(meta.get("lang", ""))
        if lang:
            langs.append(lang)
    if not langs:
        return float("nan")
    mismatches = sum(1 for lang in langs if lang not in expected_set)
    return mismatches / len(langs)


def _mean_ascii_ratio_for_entries(
    entries: Sequence[Tuple[int, str, float]],
    *,
    qid: str,
    condition: str,
    by_hit: Mapping[Tuple[str, str, int, str], Mapping[str, str]],
    by_doc: Mapping[Tuple[str, str, str], Mapping[str, str]],
) -> float:
    values: List[float] = []
    for rank, docid, _ in entries:
        meta = _lookup_doc_meta(
            qid=qid,
            condition=condition,
            rank=rank,
            docid=docid,
            by_hit=by_hit,
            by_doc=by_doc,
        )
        ratio = snippet_ascii_ratio(meta.get("snippet", ""))
        if is_finite(ratio):
            values.append(float(ratio))
    if not values:
        return float("nan")
    return sum(values) / len(values)


def load_perquery_eval_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    """Load nDCG@10 and R@10 from evaluate.py per-query CSV output."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return {}

        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        qid_col = field_map.get("qid") or field_map.get("query-id") or field_map.get("query_id")
        ndcg_col = field_map.get("ndcg@10")
        recall_col = field_map.get("r@10") or field_map.get("recall@10")
        if not qid_col or not ndcg_col or not recall_col:
            return {}

        out: Dict[str, Dict[str, float]] = {}
        for row in reader:
            qid = str(row.get(qid_col, "")).strip()
            if not qid:
                continue
            ndcg = parse_float(row.get(ndcg_col))
            recall = parse_float(row.get(recall_col))
            if not is_finite(ndcg) or not is_finite(recall):
                continue
            out[qid] = {"ndcg10": float(ndcg), "recall10": float(recall)}
        return out


def run_inspector(args: argparse.Namespace) -> None:
    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise SystemExit(f"Cases CSV not found: {cases_path}")

    cases = read_csv_rows(cases_path)
    if not cases:
        raise SystemExit(f"Cases CSV is empty: {cases_path}")

    case_row = None
    for row in cases:
        if row.get("case_id") == args.case_id:
            case_row = row
            break
    if case_row is None:
        raise SystemExit(f"case_id not found in {cases_path}: {args.case_id}")

    case_id = str(case_row.get("case_id", args.case_id))
    out_dir = Path(args.out_dir) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_run = Path(case_row.get("endpoint_run_path", ""))
    mixed_run = Path(case_row.get("mixed_run_path", ""))

    qrels = load_qrels(Path(args.qrels))
    run_end = load_trec_run(endpoint_run)
    run_mix = load_trec_run(mixed_run)

    endpoint_perquery_path_text = str(case_row.get("endpoint_perquery_path", "")).strip()
    mixed_perquery_path_text = str(case_row.get("mixed_perquery_path", "")).strip()
    endpoint_perquery_metrics = (
        load_perquery_eval_metrics(Path(endpoint_perquery_path_text)) if endpoint_perquery_path_text else {}
    )
    mixed_perquery_metrics = load_perquery_eval_metrics(Path(mixed_perquery_path_text)) if mixed_perquery_path_text else {}

    qrels_qids = set(qrels.keys())
    if endpoint_perquery_metrics and mixed_perquery_metrics:
        # Prefer evaluate.py per-query universe to match aggregate scoring behavior exactly.
        all_qids = sorted((set(endpoint_perquery_metrics.keys()) & set(mixed_perquery_metrics.keys())) & qrels_qids)
        qid_source = "evaluate_perquery_intersection"
    else:
        all_qids = sorted((set(run_end.keys()) | set(run_mix.keys())) & qrels_qids)
        qid_source = "run_qrels_intersection"

    if args.qid_list:
        allowed_qids = load_qid_list(Path(args.qid_list))
        all_qids = [qid for qid in all_qids if qid in allowed_qids]
    if not all_qids:
        raise SystemExit("No overlapping qids between qrels and runs.")
    log(
        f"QID universe: {len(all_qids)} queries "
        f"(source={qid_source}, endpoint_perquery={len(endpoint_perquery_metrics)}, mixed_perquery={len(mixed_perquery_metrics)})"
    )

    q_lang_a = (case_row.get("q_lang_a") or "").strip().lower()
    q_lang_b = (case_row.get("q_lang_b") or "").strip().lower()

    queries_a = load_queries_tsv(Path(args.queries_dir) / f"queries.{q_lang_a}.tsv") if q_lang_a else {}
    queries_b = load_queries_tsv(Path(args.queries_dir) / f"queries.{q_lang_b}.tsv") if q_lang_b else {}

    emb_a = load_embeddings(Path(args.query_cache_root), q_lang_a) if q_lang_a else {}
    emb_b = load_embeddings(Path(args.query_cache_root), q_lang_b) if q_lang_b else {}

    lambda_star = parse_float(case_row.get("lambda_star"))
    lam = to_lambda(lambda_star)
    label_config = FailureLabelConfig(
        mismatch_rate_gt=args.label_mismatch_rate_gt,
        endpoint_cos_lt=args.label_endpoint_cos_lt,
        len_ratio_min=args.label_len_ratio_min,
        len_ratio_max=args.label_len_ratio_max,
        delta_recall_lt=args.label_delta_recall_lt,
        rankdrop_delta_ndcg_lt=args.label_rankdrop_ndcg_lt,
        rankdrop_delta_recall_ge=args.label_rankdrop_recall_ge,
    )
    if label_config.len_ratio_min > label_config.len_ratio_max:
        raise SystemExit(
            "Invalid label thresholds: --label-len-ratio-min must be <= --label-len-ratio-max"
        )

    per_query: List[Dict[str, object]] = []
    metric_source_counter = Counter()

    for qid in all_qids:
        qrel = qrels.get(qid, {})
        end_entries = run_end.get(qid, [])
        mix_entries = run_mix.get(qid, [])

        end_top10 = top_entries(end_entries, args.k)
        mix_top10 = top_entries(mix_entries, args.k)
        end_top50 = top_entries(end_entries, args.rank_depth)
        mix_top50 = top_entries(mix_entries, args.rank_depth)

        end_doc10 = [d for _, d, _ in end_top10]
        mix_doc10 = [d for _, d, _ in mix_top10]
        end_doc50 = [d for _, d, _ in end_top50]
        mix_doc50 = [d for _, d, _ in mix_top50]

        metric_source = "recomputed_from_run_qrels"
        ndcg_end = ndcg_at_k(end_doc10, qrel, args.k)
        ndcg_mix = ndcg_at_k(mix_doc10, qrel, args.k)
        rec_end = recall_at_k(end_doc10, qrel, args.k)
        rec_mix = recall_at_k(mix_doc10, qrel, args.k)

        end_eval = endpoint_perquery_metrics.get(qid)
        mix_eval = mixed_perquery_metrics.get(qid)
        if end_eval and mix_eval:
            ndcg_end_eval = parse_float(end_eval.get("ndcg10"))
            ndcg_mix_eval = parse_float(mix_eval.get("ndcg10"))
            rec_end_eval = parse_float(end_eval.get("recall10"))
            rec_mix_eval = parse_float(mix_eval.get("recall10"))
            if all(is_finite(v) for v in (ndcg_end_eval, ndcg_mix_eval, rec_end_eval, rec_mix_eval)):
                ndcg_end = ndcg_end_eval
                ndcg_mix = ndcg_mix_eval
                rec_end = rec_end_eval
                rec_mix = rec_mix_eval
                metric_source = "evaluate_perquery"

        first_end = first_rel_rank(end_top50, qrel, args.rank_depth)
        first_mix = first_rel_rank(mix_top50, qrel, args.rank_depth)

        text_a = queries_a.get(qid, "")
        text_b = queries_b.get(qid, "")
        token_count_a = count_word_tokens(text_a, lang=q_lang_a)
        token_count_b = count_word_tokens(text_b, lang=q_lang_b)
        len_ratio = (token_count_a / token_count_b) if token_count_b > 0 else float("nan")

        vec_a = emb_a.get(qid)
        vec_b = emb_b.get(qid)
        r_proj, delta_perp, cos_to_a, cos_to_b, endpoint_cos = geometry_features(vec_a, vec_b, lam)

        per_query.append(
            {
                "case_id": case_id,
                "qid": qid,
                "set": "",
                "text_a": text_a,
                "text_b": text_b,
                "token_count_a": token_count_a,
                "token_count_b": token_count_b,
                "ndcg10_end": ndcg_end,
                "ndcg10_mix": ndcg_mix,
                "delta_ndcg10": ndcg_mix - ndcg_end,
                "recall10_end": rec_end,
                "recall10_mix": rec_mix,
                "delta_recall10": rec_mix - rec_end,
                "first_rel_rank50_end": first_end,
                "first_rel_rank50_mix": first_mix,
                "overlap10": len(set(end_doc10) & set(mix_doc10)),
                "overlap50": len(set(end_doc50) & set(mix_doc50)),
                "best_rel_rank_shift": shift_with_inf(first_mix, first_end),
                "len_ratio": len_ratio,
                "endpoint_cos": endpoint_cos,
                "r": r_proj,
                "delta_perp": delta_perp,
                "cos_to_a": cos_to_a,
                "cos_to_b": cos_to_b,
                "doc_lang_mismatch_rate10_end": float("nan"),
                "doc_lang_mismatch_rate10_mix": float("nan"),
                "ascii_ratio10_end": float("nan"),
                "ascii_ratio10_mix": float("nan"),
                "label": "",
                "metric_source": metric_source,
                "end_top10": end_top10,
                "mix_top10": mix_top10,
            }
        )
        metric_source_counter[metric_source] += 1

    per_query.sort(key=lambda r: (parse_float(r.get("delta_ndcg10")), str(r.get("qid"))))

    focus_label = "best" if args.query_focus == "best" else "worst"
    focus_title = "Best" if focus_label == "best" else "Worst"
    focus_effect_noun = "gain" if focus_label == "best" else "drop"

    if focus_label == "best":
        ordered_focus = sorted(per_query, key=lambda r: (-parse_float(r.get("delta_ndcg10")), str(r.get("qid"))))
    else:
        ordered_focus = list(per_query)

    focus_n = min(args.worst_n, len(ordered_focus))
    focus = ordered_focus[:focus_n]
    focus_qids = {str(r["qid"]) for r in focus}

    remaining = [r for r in per_query if str(r["qid"]) not in focus_qids]
    remaining.sort(key=lambda r: (abs(parse_float(r.get("delta_ndcg10"))), str(r.get("qid"))))
    control_n = min(args.control_n, len(remaining))
    control = remaining[:control_n]

    selected = focus + control

    for row in selected:
        row["set"] = focus_label if str(row["qid"]) in focus_qids else "control"

    doc_meta_by_hit: Dict[Tuple[str, str, int, str], Dict[str, str]] = {}
    doc_meta_by_doc: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if args.doc_meta:
        doc_meta_by_hit, doc_meta_by_doc = read_doc_meta_subset(Path(args.doc_meta), case_id=case_id)
        log(
            f"Loaded doc metadata rows: by_hit={len(doc_meta_by_hit)}, "
            f"by_qid_cond_docid={len(doc_meta_by_doc)}"
        )

    expected_doc_langs = set()
    doc_lang_text = (case_row.get("doc_lang") or "").strip().lower()
    if doc_lang_text:
        for tok in doc_lang_text.replace("+", " ").replace(",", " ").replace("/", " ").split():
            expected_doc_langs.add(LANG_NAME_TO_CODE.get(tok, tok))
    if not expected_doc_langs:
        expected_doc_langs.update(parse_doc_mix_codes(case_row.get("doc_mix", "")))

    for row in selected:
        qid = str(row["qid"])
        row["doc_lang_mismatch_rate10_end"] = _mismatch_rate_for_entries(
            row.get("end_top10", []),
            qid=qid,
            condition="endpoint",
            by_hit=doc_meta_by_hit,
            by_doc=doc_meta_by_doc,
            expected=expected_doc_langs,
        )
        row["doc_lang_mismatch_rate10_mix"] = _mismatch_rate_for_entries(
            row.get("mix_top10", []),
            qid=qid,
            condition="mixed",
            by_hit=doc_meta_by_hit,
            by_doc=doc_meta_by_doc,
            expected=expected_doc_langs,
        )
        row["ascii_ratio10_end"] = _mean_ascii_ratio_for_entries(
            row.get("end_top10", []),
            qid=qid,
            condition="endpoint",
            by_hit=doc_meta_by_hit,
            by_doc=doc_meta_by_doc,
        )
        row["ascii_ratio10_mix"] = _mean_ascii_ratio_for_entries(
            row.get("mix_top10", []),
            qid=qid,
            condition="mixed",
            by_hit=doc_meta_by_hit,
            by_doc=doc_meta_by_doc,
        )
        row["label"] = assign_failure_label(row, label_config)

    label_counter = Counter()
    label_deltas: Dict[str, List[float]] = defaultdict(list)
    for row in selected:
        if row.get("set") != focus_label:
            continue
        label = str(row.get("label", ""))
        delta = parse_float(row.get("delta_ndcg10"))
        label_counter[label] += 1
        if is_finite(delta):
            label_deltas[label].append(delta)

    selected_csv = out_dir / "selected_queries.csv"
    top_docs_csv = out_dir / "top_docs_diff.csv"
    summary_csv = out_dir / "case_summary.csv"
    report_md = out_dir / "case_report.md"

    selected_columns = [
        "case_id",
        "qid",
        "set",
        "metric_source",
        "ndcg10_end",
        "ndcg10_mix",
        "delta_ndcg10",
        "recall10_end",
        "recall10_mix",
        "delta_recall10",
        "first_rel_rank50_end",
        "first_rel_rank50_mix",
        "overlap10",
        "overlap50",
        "best_rel_rank_shift",
        "token_count_a",
        "token_count_b",
        "len_ratio",
        "endpoint_cos",
        "r",
        "delta_perp",
        "cos_to_a",
        "cos_to_b",
        "doc_lang_mismatch_rate10_end",
        "doc_lang_mismatch_rate10_mix",
        "ascii_ratio10_end",
        "ascii_ratio10_mix",
        "label",
        "text_a",
        "text_b",
    ]

    selected_rows_for_csv: List[Dict[str, object]] = []
    for row in selected:
        out = {k: row.get(k, "") for k in selected_columns}
        out["first_rel_rank50_end"] = rank_text(parse_float(row.get("first_rel_rank50_end")))
        out["first_rel_rank50_mix"] = rank_text(parse_float(row.get("first_rel_rank50_mix")))
        selected_rows_for_csv.append(out)

    write_csv_rows(selected_csv, selected_rows_for_csv, selected_columns)

    docs_rows: List[Dict[str, object]] = []
    for row in selected:
        qid = str(row["qid"])
        for condition, entries in (("endpoint", row.get("end_top10", [])), ("mixed", row.get("mix_top10", []))):
            for rank, docid, score in entries:
                qrel = qrels.get(qid, {})
                rel = qrel.get(docid, 0)
                meta = _lookup_doc_meta(
                    qid=qid,
                    condition=condition,
                    rank=rank,
                    docid=docid,
                    by_hit=doc_meta_by_hit,
                    by_doc=doc_meta_by_doc,
                )
                docs_rows.append(
                    {
                        "case_id": case_id,
                        "qid": qid,
                        "condition": condition,
                        "rank": rank,
                        "docid": docid,
                        "retrieval_score_raw": score,
                        "score": score,
                        "rel": rel,
                        "doc_lang": meta.get("lang", ""),
                        "snippet": meta.get("snippet", ""),
                    }
                )

    docs_columns = [
        "case_id",
        "qid",
        "condition",
        "rank",
        "docid",
        "retrieval_score_raw",
        "score",
        "rel",
        "doc_lang",
        "snippet",
    ]
    write_csv_rows(top_docs_csv, docs_rows, docs_columns)

    all_deltas = sorted(parse_float(r.get("delta_ndcg10")) for r in per_query)
    all_deltas = [d for d in all_deltas if is_finite(d)]

    focus_mean = statistics.mean(parse_float(r.get("delta_ndcg10")) for r in focus) if focus else float("nan")
    control_mean = statistics.mean(parse_float(r.get("delta_ndcg10")) for r in control) if control else float("nan")

    summary_row = {
        "case_id": case_id,
        "query_focus": focus_label,
        "pair": case_row.get("pair", ""),
        "doc_mix": case_row.get("doc_mix", ""),
        "model": case_row.get("model", ""),
        "method": case_row.get("method", ""),
        "doc_index_id": case_row.get("doc_index_id", ""),
        "endpoint_lambda": case_row.get("endpoint_lambda", ""),
        "lambda_star": case_row.get("lambda_star", ""),
        "endpoint_score": case_row.get("endpoint_score", ""),
        "mixed_best_score": case_row.get("mixed_best_score", ""),
        "delta": case_row.get("delta", ""),
        "ci_low": case_row.get("ci_low", ""),
        "ci_high": case_row.get("ci_high", ""),
        "label_mismatch_rate_gt": label_config.mismatch_rate_gt,
        "label_endpoint_cos_lt": label_config.endpoint_cos_lt,
        "label_len_ratio_min": label_config.len_ratio_min,
        "label_len_ratio_max": label_config.len_ratio_max,
        "label_delta_recall_lt": label_config.delta_recall_lt,
        "label_rankdrop_ndcg_lt": label_config.rankdrop_delta_ndcg_lt,
        "label_rankdrop_recall_ge": label_config.rankdrop_delta_recall_ge,
        "num_queries": len(per_query),
        "num_queries_metric_source_evaluate_perquery": metric_source_counter.get("evaluate_perquery", 0),
        "num_queries_metric_source_recomputed": metric_source_counter.get("recomputed_from_run_qrels", 0),
        "num_focus": len(focus),
        "num_control": len(control),
        "focus_mean_delta_ndcg10": focus_mean,
        "control_mean_delta_ndcg10": control_mean,
        # Backward-compatible aliases for older downstream readers.
        "num_worst": len(focus) if focus_label == "worst" else 0,
        "worst_mean_delta_ndcg10": focus_mean if focus_label == "worst" else float("nan"),
        "count_IndexLeakage": label_counter.get("IndexLeakage", 0),
        "count_TranslationDivergence": label_counter.get("TranslationDivergence", 0),
        "count_RecallDrop": label_counter.get("RecallDrop", 0),
        "count_RankDrop": label_counter.get("RankDrop", 0),
        "count_Unclassified": label_counter.get(FAILURE_LABEL_UNCLASSIFIED, 0),
    }

    summary_columns = [
        "case_id",
        "query_focus",
        "pair",
        "doc_mix",
        "model",
        "method",
        "doc_index_id",
        "endpoint_lambda",
        "lambda_star",
        "endpoint_score",
        "mixed_best_score",
        "delta",
        "ci_low",
        "ci_high",
        "label_mismatch_rate_gt",
        "label_endpoint_cos_lt",
        "label_len_ratio_min",
        "label_len_ratio_max",
        "label_delta_recall_lt",
        "label_rankdrop_ndcg_lt",
        "label_rankdrop_recall_ge",
        "num_queries",
        "num_queries_metric_source_evaluate_perquery",
        "num_queries_metric_source_recomputed",
        "num_focus",
        "num_worst",
        "num_control",
        "focus_mean_delta_ndcg10",
        "worst_mean_delta_ndcg10",
        "control_mean_delta_ndcg10",
        "count_IndexLeakage",
        "count_TranslationDivergence",
        "count_RecallDrop",
        "count_RankDrop",
        "count_Unclassified",
    ]
    write_csv_rows(summary_csv, [summary_row], summary_columns)

    with report_md.open("w", encoding="utf-8") as fh:
        fh.write(f"# Case Report: {case_id}\n\n")
        fh.write("## 1) Case Header\n\n")
        fh.write(f"- pair: `{case_row.get('pair','')}`\n")
        fh.write(f"- doc_mix: `{case_row.get('doc_mix','')}`\n")
        fh.write(f"- model: `{case_row.get('model','')}`\n")
        fh.write(f"- method: `{case_row.get('method','')}`\n")
        fh.write(f"- doc_index_id: `{case_row.get('doc_index_id','')}`\n")
        fh.write(f"- endpoint lambda: `{case_row.get('endpoint_lambda','')}`\n")
        fh.write(f"- lambda*: `{case_row.get('lambda_star','')}`\n")
        fh.write(
            f"- overall delta (mixed - endpoint): `{case_row.get('delta','')}` "
            f"(CI90: [{case_row.get('ci_low','')}, {case_row.get('ci_high','')}])\n\n"
        )

        fh.write(f"## 2) How Many Queries Drive the {focus_effect_noun.capitalize()}\n\n")
        fh.write(
            "- metric source counts: "
            f"evaluate_perquery={metric_source_counter.get('evaluate_perquery',0)}, "
            f"recomputed_from_run_qrels={metric_source_counter.get('recomputed_from_run_qrels',0)}\n"
        )
        if all_deltas:
            p25, p50, p75 = REPORT_DELTA_QUANTILES
            fh.write(
                "- ΔnDCG@10 quantiles (all queries): "
                f"min={fmt(all_deltas[0],4)}, "
                f"p25={fmt(percentile(all_deltas,p25),4)}, "
                f"p50={fmt(percentile(all_deltas,p50),4)}, "
                f"p75={fmt(percentile(all_deltas,p75),4)}, "
                f"max={fmt(all_deltas[-1],4)}\n"
            )
        else:
            fh.write("- No valid per-query deltas.\n")
        fh.write(f"- {focus_label}-{len(focus)} mean ΔnDCG@10: `{fmt(focus_mean,4)}`\n")
        fh.write(f"- control-{len(control)} mean ΔnDCG@10: `{fmt(control_mean,4)}`\n\n")

        fh.write(f"## 3) Failure Label Breakdown ({focus_title} Set)\n\n")
        if focus:
            fh.write(
                "- label thresholds: "
                f"mismatch_rate_mix>{fmt(label_config.mismatch_rate_gt,4)}, "
                f"endpoint_cos<{fmt(label_config.endpoint_cos_lt,4)}, "
                f"len_ratio<{fmt(label_config.len_ratio_min,4)} or >{fmt(label_config.len_ratio_max,4)}, "
                f"delta_recall<{fmt(label_config.delta_recall_lt,4)}, "
                f"rankdrop=(delta_ndcg<{fmt(label_config.rankdrop_delta_ndcg_lt,4)} "
                f"and delta_recall>={fmt(label_config.rankdrop_delta_recall_ge,4)})\n"
            )
            for label in FAILURE_LABEL_ORDER:
                vals = label_deltas.get(label, [])
                mean_delta = statistics.mean(vals) if vals else float("nan")
                fh.write(f"- {label}: count={label_counter.get(label,0)}, mean ΔnDCG@10={fmt(mean_delta,4)}\n")
        else:
            fh.write(f"- No {focus_label} queries selected.\n")
        fh.write("\n")

        fh.write(f"## 4) Top {max(args.report_top_worst, 0)} {focus_title} Queries\n\n")
        fh.write(
            "| qid | metric_source | ndcg_end | ndcg_mix | d_ndcg | rec_end | rec_mix | d_rec | "
            "first_end | first_mix | rank_shift | ov10 | ov50 | tok_a | tok_b | len_ratio | endpoint_cos | r | delta_perp | "
            "cos_to_a | cos_to_b | mismatch_end | mismatch_mix | ascii_end | ascii_mix | label |\n"
        )
        fh.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for row in focus[: max(args.report_top_worst, 0)]:
            fh.write(
                "| "
                f"{row.get('qid','')} | {row.get('metric_source','')} | "
                f"{fmt(row.get('ndcg10_end'),4)} | {fmt(row.get('ndcg10_mix'),4)} | {fmt(row.get('delta_ndcg10'),4)} | "
                f"{fmt(row.get('recall10_end'),4)} | {fmt(row.get('recall10_mix'),4)} | {fmt(row.get('delta_recall10'),4)} | "
                f"{rank_text(parse_float(row.get('first_rel_rank50_end')))} | {rank_text(parse_float(row.get('first_rel_rank50_mix')))} | "
                f"{fmt(row.get('best_rel_rank_shift'),4)} | {int(parse_float(row.get('overlap10'),0))} | "
                f"{int(parse_float(row.get('overlap50'),0))} | {int(parse_float(row.get('token_count_a'),0))} | "
                f"{int(parse_float(row.get('token_count_b'),0))} | {fmt(row.get('len_ratio'),4)} | "
                f"{fmt(row.get('endpoint_cos'),4)} | {fmt(row.get('r'),4)} | {fmt(row.get('delta_perp'),4)} | "
                f"{fmt(row.get('cos_to_a'),4)} | {fmt(row.get('cos_to_b'),4)} | "
                f"{fmt(row.get('doc_lang_mismatch_rate10_end'),4)} | {fmt(row.get('doc_lang_mismatch_rate10_mix'),4)} | "
                f"{fmt(row.get('ascii_ratio10_end'),4)} | {fmt(row.get('ascii_ratio10_mix'),4)} | {row.get('label','')} |\n"
            )
        fh.write("\n")

        fh.write(f"## 5) Per-Query Diff Blocks (Top {max(args.report_top_diff_blocks, 0)} {focus_title})\n\n")
        fh.write(
            "All metric deltas are `mixed - endpoint` in 0-100 point units.\n\n"
        )
        fh.write(
            "Note: `retrieval_score_raw` below is the original run ranking score from `.trec`, "
            "not an evaluation metric and not on the 0-100 nDCG/Recall scale.\n\n"
        )

        for row in focus[: max(args.report_top_diff_blocks, 0)]:
            qid = str(row.get("qid", ""))
            fh.write(f"### qid `{qid}`\n\n")
            fh.write(
                f"- query A (`{q_lang_a}`): "
                f"{to_markdown_cell(str(row.get('text_a','')), REPORT_QUERY_TEXT_CELL_LIMIT)}\n"
            )
            fh.write(
                f"- query B (`{q_lang_b}`): "
                f"{to_markdown_cell(str(row.get('text_b','')), REPORT_QUERY_TEXT_CELL_LIMIT)}\n"
            )
            fh.write(
                "- diagnosis: "
                f"{row.get('label','')}; "
                f"nDCG@10 end={fmt(row.get('ndcg10_end'),4)}, mix={fmt(row.get('ndcg10_mix'),4)}, Δ={fmt(row.get('delta_ndcg10'),4)}; "
                f"Recall@10 end={fmt(row.get('recall10_end'),4)}, mix={fmt(row.get('recall10_mix'),4)}, Δ={fmt(row.get('delta_recall10'),4)}; "
                f"tokens(a/b)={int(parse_float(row.get('token_count_a'),0))}/{int(parse_float(row.get('token_count_b'),0))}, "
                f"len_ratio={fmt(row.get('len_ratio'),4)}; "
                f"overlap@10={int(parse_float(row.get('overlap10'),0))}; source={row.get('metric_source','')}; "
                f"focus={focus_label} ({focus_effect_noun})\n\n"
            )

            fh.write("Endpoint top-10\n\n")
            fh.write("| rank | docid | rel | retrieval_score_raw | lang | snippet |\n")
            fh.write("|---:|---|---:|---:|---|---|\n")
            for rank, docid, score in row.get("end_top10", []):
                rel = qrels.get(qid, {}).get(docid, 0)
                meta = _lookup_doc_meta(
                    qid=qid,
                    condition="endpoint",
                    rank=rank,
                    docid=docid,
                    by_hit=doc_meta_by_hit,
                    by_doc=doc_meta_by_doc,
                )
                fh.write(
                    f"| {rank} | {docid} | {rel} | {fmt(score,4)} | "
                    f"{to_markdown_cell(meta.get('lang',''),REPORT_DOC_LANG_CELL_LIMIT)} | {to_markdown_cell(meta.get('snippet',''))} |\n"
                )
            fh.write("\n")

            fh.write("Mixed top-10\n\n")
            fh.write("| rank | docid | rel | retrieval_score_raw | lang | snippet |\n")
            fh.write("|---:|---|---:|---:|---|---|\n")
            for rank, docid, score in row.get("mix_top10", []):
                rel = qrels.get(qid, {}).get(docid, 0)
                meta = _lookup_doc_meta(
                    qid=qid,
                    condition="mixed",
                    rank=rank,
                    docid=docid,
                    by_hit=doc_meta_by_hit,
                    by_doc=doc_meta_by_doc,
                )
                fh.write(
                    f"| {rank} | {docid} | {rel} | {fmt(score,4)} | "
                    f"{to_markdown_cell(meta.get('lang',''),REPORT_DOC_LANG_CELL_LIMIT)} | {to_markdown_cell(meta.get('snippet',''))} |\n"
                )
            fh.write("\n")

    log(f"Wrote selected query diagnostics: {selected_csv} (rows={len(selected_rows_for_csv)})")
    log(f"Wrote top-doc diffs: {top_docs_csv} (rows={len(docs_rows)})")
    log(f"Wrote case summary: {summary_csv}")
    log(f"Wrote report: {report_md}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Inspect one mined case at per-query and top-doc level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cases", required=True, help="cases.csv produced by case_miner.py")
    ap.add_argument("--case-id", required=True, help="case_id to inspect (e.g., CASE_0007)")
    ap.add_argument("--qrels", default=str(DEFAULT_QRELS), help="qrels TSV path")
    ap.add_argument(
        "--queries-dir",
        default=str(DEFAULT_QUERIES_DIR),
        help="Directory with queries.<lang>.tsv files",
    )
    ap.add_argument(
        "--query-cache-root",
        default=str(DEFAULT_QUERY_CACHE_ROOT),
        help="Directory with per-language queries.npz embedding caches",
    )
    ap.add_argument(
        "--qid-list",
        default="",
        help="Optional qid list (one qid per line) to restrict inspection universe",
    )
    ap.add_argument(
        "--doc-meta",
        default="",
        help=(
            "Optional hit-level doc_meta_subset.csv with required columns "
            "case_id,qid,condition,rank,docid and optional lang,title,snippet"
        ),
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_INSPECT_OUT), help="Base output directory")
    ap.add_argument(
        "--query-focus",
        choices=["worst", "best"],
        default="worst",
        help="Whether to focus analysis on largest drops or largest gains",
    )
    ap.add_argument(
        "--worst-n",
        type=int,
        default=DEFAULT_INSPECT_WORST_N,
        help="Number of focused queries to inspect (best/worst based on --query-focus)",
    )
    ap.add_argument(
        "--control-n",
        type=int,
        default=DEFAULT_INSPECT_CONTROL_N,
        help="Number of control queries near delta=0",
    )
    ap.add_argument("-k", type=int, default=DEFAULT_INSPECT_K, help="Cutoff for nDCG@k / Recall@k and top-doc output")
    ap.add_argument(
        "--rank-depth",
        type=int,
        default=DEFAULT_INSPECT_RANK_DEPTH,
        help="Depth used for overlap@depth and first relevant rank",
    )
    ap.add_argument(
        "--report-top-worst",
        type=int,
        default=DEFAULT_REPORT_TOP_WORST,
        help="Rows to include in the report's top focused-query table",
    )
    ap.add_argument(
        "--report-top-diff-blocks",
        type=int,
        default=DEFAULT_REPORT_DIFF_BLOCKS,
        help="Per-query diff blocks to render for the focused query set",
    )
    ap.add_argument(
        "--label-mismatch-rate-gt",
        type=float,
        default=DEFAULT_FAILURE_LABEL_CONFIG.mismatch_rate_gt,
        help="IndexLeakage threshold: label if doc_lang_mismatch_rate10_mix > this value",
    )
    ap.add_argument(
        "--label-endpoint-cos-lt",
        type=float,
        default=DEFAULT_FAILURE_LABEL_CONFIG.endpoint_cos_lt,
        help="TranslationDivergence threshold: label if endpoint_cos < this value",
    )
    ap.add_argument(
        "--label-len-ratio-min",
        type=float,
        default=DEFAULT_FAILURE_LABEL_CONFIG.len_ratio_min,
        help="TranslationDivergence threshold: label if len_ratio < this value",
    )
    ap.add_argument(
        "--label-len-ratio-max",
        type=float,
        default=DEFAULT_FAILURE_LABEL_CONFIG.len_ratio_max,
        help="TranslationDivergence threshold: label if len_ratio > this value",
    )
    ap.add_argument(
        "--label-delta-recall-lt",
        type=float,
        default=DEFAULT_FAILURE_LABEL_CONFIG.delta_recall_lt,
        help="RecallDrop threshold: label if delta_recall10 < this value",
    )
    ap.add_argument(
        "--label-rankdrop-ndcg-lt",
        type=float,
        default=DEFAULT_FAILURE_LABEL_CONFIG.rankdrop_delta_ndcg_lt,
        help="RankDrop threshold: requires delta_ndcg10 < this value",
    )
    ap.add_argument(
        "--label-rankdrop-recall-ge",
        type=float,
        default=DEFAULT_FAILURE_LABEL_CONFIG.rankdrop_delta_recall_ge,
        help="RankDrop threshold: requires delta_recall10 >= this value",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_inspector(args)


if __name__ == "__main__":
    main()
