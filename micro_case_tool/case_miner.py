#!/usr/bin/env python3
"""Stage A: mine mixed-query settings from existing result CSVs."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from micro_case_common import (
    DEFAULT_MINER_CI_THRESHOLD,
    DEFAULT_MINER_DELTA_THRESHOLD,
    DEFAULT_MINER_TOP_N,
    DEFAULT_MINE_OUT,
    DEFAULT_PROCESSED_RESULTS,
    DEFAULT_RAW_RESULTS,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_ROOT,
    doc_lang_token,
    doc_type_from_codes,
    infer_doc_codes,
    is_endpoint_lambda,
    is_finite,
    is_non_english_doc_setting,
    log,
    normalize_doc_mix,
    normalize_pair,
    parse_float,
    read_csv_rows,
    resolve_run_path,
    select_processed_row,
    source_doc_index_id,
    source_eval_path,
    source_perquery_path,
    source_run_id,
    split_pair_codes,
    to_lambda,
    write_csv_rows,
    warn,
)


def build_method_comparison(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("pair", "")),
            str(row.get("doc_mix", "")),
            str(row.get("model", "")),
            str(row.get("doc_index_id", "")),
        )
        method = str(row.get("method", ""))
        grouped[key][method] = row

    out: List[Dict[str, object]] = []
    for (pair, doc_mix, model, doc_index_id), by_method in grouped.items():
        if "embed" not in by_method or "word" not in by_method:
            continue
        embed = by_method["embed"]
        word = by_method["word"]
        d_embed = parse_float(embed.get("delta"))
        d_word = parse_float(word.get("delta"))
        out.append(
            {
                "pair": pair,
                "doc_mix": doc_mix,
                "model": model,
                "doc_index_id": doc_index_id,
                "delta_embed": d_embed,
                "delta_word": d_word,
                "delta_gap_embed_minus_word": d_embed - d_word if is_finite(d_embed) and is_finite(d_word) else math.nan,
                "endpoint_embed": parse_float(embed.get("endpoint_score")),
                "endpoint_word": parse_float(word.get("endpoint_score")),
                "mixed_embed": parse_float(embed.get("mixed_best_score")),
                "mixed_word": parse_float(word.get("mixed_best_score")),
                "lambda_star_embed": parse_float(embed.get("lambda_star")),
                "lambda_star_word": parse_float(word.get("lambda_star")),
            }
        )
    out.sort(key=lambda r: (parse_float(r.get("delta_gap_embed_minus_word")), r["pair"], r["doc_mix"]))
    return out


def run_miner(args: argparse.Namespace) -> None:
    raw_path = Path(args.raw_results)
    processed_path = Path(args.processed_results)
    result_root = Path(args.results_root)
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)

    if not raw_path.exists():
        raise SystemExit(f"Raw results CSV not found: {raw_path}")
    if not processed_path.exists():
        raise SystemExit(f"Processed results CSV not found: {processed_path}")

    raw_rows = read_csv_rows(raw_path)
    processed_rows = read_csv_rows(processed_path)

    processed_index: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in processed_rows:
        key = (normalize_pair(row.get("pair", "")), normalize_doc_mix(row.get("doc_mix", "")))
        processed_index[key].append(row)

    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)

    for row in raw_rows:
        pair = normalize_pair(row.get("pair", ""))
        doc_mix = normalize_doc_mix(row.get("doc_mix", ""))
        method = (row.get("method") or "").strip().lower() or "unknown"
        model = (row.get("model") or "").strip() or "unknown"
        source_file = (row.get("source_file") or "").strip()

        if args.method != "all" and method != args.method:
            continue
        if not source_file:
            continue

        mix_ratio = parse_float(row.get("mix_ratio"))
        ndcg10 = parse_float(row.get("ndcg10"))
        if not is_finite(mix_ratio) or not is_finite(ndcg10):
            continue

        lam = to_lambda(mix_ratio)
        if not is_finite(lam):
            continue

        doc_index_id = source_doc_index_id(source_file)
        doc_codes = infer_doc_codes(doc_mix, doc_index_id)
        doc_type = doc_type_from_codes(doc_codes)

        if args.doc_type != "all" and doc_type != args.doc_type:
            continue
        if args.non_english_docs_only and not is_non_english_doc_setting(doc_codes, doc_mix):
            continue

        grouped[(pair, doc_mix, method, model, doc_index_id)].append(
            {
                "pair": pair,
                "doc_mix": doc_mix,
                "method": method,
                "model": model,
                "doc_index_id": doc_index_id,
                "doc_codes": doc_codes,
                "doc_type": doc_type,
                "mix_ratio": mix_ratio,
                "lambda": lam,
                "ndcg10": ndcg10,
                "source_file": source_file,
            }
        )

    all_rows: List[Dict[str, object]] = []

    for (pair, doc_mix, method, model, doc_index_id), records in grouped.items():
        endpoints = [r for r in records if is_endpoint_lambda(float(r["lambda"]))]
        mixed = [r for r in records if 0.0 < float(r["lambda"]) < 1.0]
        if not endpoints or not mixed:
            continue

        endpoints_sorted = sorted(
            endpoints,
            key=lambda r: (float(r["ndcg10"]), -abs(float(r["lambda"]))),
            reverse=True,
        )
        mixed_sorted = sorted(
            mixed,
            key=lambda r: (float(r["ndcg10"]), -abs(float(r["lambda"]))),
            reverse=True,
        )

        endpoint = endpoints_sorted[0]
        mixed_best = mixed_sorted[0]

        endpoint_score = float(endpoint["ndcg10"])
        mixed_score = float(mixed_best["ndcg10"])
        delta = mixed_score - endpoint_score

        p_row = select_processed_row(processed_index.get((pair, doc_mix), []), delta)
        ci_low = parse_float(p_row.get(args.ci_low_column)) if p_row else math.nan
        ci_high = parse_float(p_row.get(args.ci_high_column)) if p_row else math.nan

        if args.outlier_indicator == "ci":
            is_outlier = int(is_finite(ci_low) and ci_low < args.ci_threshold)
        else:
            is_outlier = int(is_finite(delta) and delta < args.delta_threshold)

        q_lang_a, q_lang_b = split_pair_codes(pair)
        doc_codes = endpoint.get("doc_codes") or infer_doc_codes(doc_mix, doc_index_id)

        endpoint_source = str(endpoint["source_file"])
        mixed_source = str(mixed_best["source_file"])

        endpoint_run_path = resolve_run_path(run_root, endpoint_source, method, float(endpoint["mix_ratio"]))
        mixed_run_path = resolve_run_path(run_root, mixed_source, method, float(mixed_best["mix_ratio"]))

        endpoint_eval_path = source_eval_path(result_root, endpoint_source)
        mixed_eval_path = source_eval_path(result_root, mixed_source)
        endpoint_perquery = source_perquery_path(result_root, endpoint_source)
        mixed_perquery = source_perquery_path(result_root, mixed_source)

        all_rows.append(
            {
                "case_id": "",
                "is_outlier": is_outlier,
                "pair": pair,
                "doc_mix": doc_mix,
                "doc_type": endpoint.get("doc_type", ""),
                "model": model,
                "method": method,
                "doc_index_id": doc_index_id,
                "doc_lang": doc_lang_token(doc_codes if isinstance(doc_codes, list) else []),
                "q_lang_a": q_lang_a,
                "q_lang_b": q_lang_b,
                "endpoint_lambda": float(endpoint["lambda"]),
                "lambda_star": float(mixed_best["lambda"]),
                "endpoint_score": endpoint_score,
                "mixed_best_score": mixed_score,
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "processed_delta_ndcg": parse_float(p_row.get("delta_ndcg")) if p_row else math.nan,
                "endpoint_run_id": source_run_id(endpoint_source),
                "mixed_run_id": source_run_id(mixed_source),
                "endpoint_run_path": str(endpoint_run_path),
                "mixed_run_path": str(mixed_run_path),
                "endpoint_eval_path": str(endpoint_eval_path),
                "mixed_eval_path": str(mixed_eval_path),
                "endpoint_perquery_path": str(endpoint_perquery),
                "mixed_perquery_path": str(mixed_perquery),
                "endpoint_source_file": endpoint_source,
                "mixed_source_file": mixed_source,
                "num_lambdas": len(records),
            }
        )

    if not all_rows:
        raise SystemExit("No valid settings found after filtering. Check --method/--doc-type filters.")

    def sort_key(row: Mapping[str, object]) -> Tuple[float, float, str, str, str, str, str]:
        delta = parse_float(row.get("delta"))
        ci_low = parse_float(row.get("ci_low"))
        delta_key = delta if is_finite(delta) else float("inf")
        ci_key = ci_low if is_finite(ci_low) else float("inf")
        primary = ci_key if args.outlier_indicator == "ci" else delta_key
        secondary = delta_key if args.outlier_indicator == "ci" else ci_key
        return (
            primary,
            secondary,
            str(row.get("pair", "")),
            str(row.get("doc_mix", "")),
            str(row.get("method", "")),
            str(row.get("model", "")),
            str(row.get("doc_index_id", "")),
        )

    all_rows.sort(key=sort_key)
    for idx, row in enumerate(all_rows, start=1):
        row["case_id"] = f"CASE_{idx:04d}"

    outliers = [r for r in all_rows if int(r.get("is_outlier", 0)) == 1]
    cases = outliers[: max(args.top_n, 0)]
    if args.fill_with_lowest and len(cases) < max(args.top_n, 0):
        needed = max(args.top_n, 0) - len(cases)
        used = {str(row.get("case_id", "")) for row in cases}
        fillers = [row for row in all_rows if str(row.get("case_id", "")) not in used]
        cases.extend(fillers[:needed])

    def best_sort_key(row: Mapping[str, object]) -> Tuple[float, float, str, str, str, str, str]:
        delta = parse_float(row.get("delta"))
        ci_low = parse_float(row.get("ci_low"))
        primary_val = ci_low if args.best_indicator == "ci" else delta
        secondary_val = delta if args.best_indicator == "ci" else ci_low
        primary = -primary_val if is_finite(primary_val) else float("inf")
        secondary = -secondary_val if is_finite(secondary_val) else float("inf")
        return (
            primary,
            secondary,
            str(row.get("pair", "")),
            str(row.get("doc_mix", "")),
            str(row.get("method", "")),
            str(row.get("model", "")),
            str(row.get("doc_index_id", "")),
        )

    best_cases: List[Dict[str, object]] = []
    if args.best_top_n > 0:
        for row in all_rows:
            delta = parse_float(row.get("delta"))
            ci_low = parse_float(row.get("ci_low"))
            if not is_finite(delta) or delta < args.best_delta_min:
                continue
            if not is_finite(ci_low) or ci_low < args.best_ci_low_min:
                continue
            best_cases.append(dict(row))
        best_cases.sort(key=best_sort_key)
        best_cases = best_cases[: max(args.best_top_n, 0)]

    all_columns = [
        "case_id",
        "is_outlier",
        "pair",
        "doc_mix",
        "doc_type",
        "model",
        "method",
        "doc_index_id",
        "doc_lang",
        "q_lang_a",
        "q_lang_b",
        "endpoint_lambda",
        "lambda_star",
        "endpoint_score",
        "mixed_best_score",
        "delta",
        "ci_low",
        "ci_high",
        "processed_delta_ndcg",
        "endpoint_run_id",
        "mixed_run_id",
        "endpoint_run_path",
        "mixed_run_path",
        "endpoint_eval_path",
        "mixed_eval_path",
        "endpoint_perquery_path",
        "mixed_perquery_path",
        "endpoint_source_file",
        "mixed_source_file",
        "num_lambdas",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "cases_all_summary.csv"
    cases_path = out_dir / "cases.csv"
    best_path = out_dir / args.best_cases_file

    write_csv_rows(all_path, all_rows, all_columns)
    write_csv_rows(cases_path, cases, all_columns)

    log(f"Wrote all settings: {all_path} (rows={len(all_rows)})")
    log(f"Wrote selected outlier cases: {cases_path} (rows={len(cases)})")
    if args.best_top_n > 0:
        write_csv_rows(best_path, best_cases, all_columns)
        log(
            f"Wrote selected top-performing cases: {best_path} (rows={len(best_cases)}; "
            f"delta>={args.best_delta_min}, ci_low>={args.best_ci_low_min}, sort={args.best_indicator})"
        )

    if args.method_comparison:
        method_rows = build_method_comparison(all_rows)
        if method_rows:
            method_path = out_dir / "method_comparison.csv"
            method_columns = [
                "pair",
                "doc_mix",
                "model",
                "doc_index_id",
                "delta_embed",
                "delta_word",
                "delta_gap_embed_minus_word",
                "endpoint_embed",
                "endpoint_word",
                "mixed_embed",
                "mixed_word",
                "lambda_star_embed",
                "lambda_star_word",
            ]
            write_csv_rows(method_path, method_rows, method_columns)
            log(f"Wrote method comparison: {method_path} (rows={len(method_rows)})")
        else:
            warn("Method comparison requested, but no settings contained both word and embed rows.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Mine mixed-query settings from existing result CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw-results", default=str(DEFAULT_RAW_RESULTS), help="CSV from collect_results.py")
    ap.add_argument(
        "--processed-results",
        default=str(DEFAULT_PROCESSED_RESULTS),
        help="Processed CSV from collect_results.py (contains delta + CI)",
    )
    ap.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory that `source_file` paths are relative to",
    )
    ap.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Root directory that stores .trec runs (from run_all_vector_pairs.sh)",
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_MINE_OUT), help="Output directory for cases.csv")
    ap.add_argument("--top-n", type=int, default=DEFAULT_MINER_TOP_N, help="Number of outlier cases to keep in cases.csv")
    ap.add_argument(
        "--delta-threshold",
        type=float,
        default=DEFAULT_MINER_DELTA_THRESHOLD,
        help="Outlier threshold on delta when --outlier-indicator=delta (0-100 metric scale)",
    )
    ap.add_argument(
        "--ci-low-column",
        default="delta_ndcg_ci90_low",
        help="Column name in processed CSV used for CI low bound",
    )
    ap.add_argument(
        "--ci-high-column",
        default="delta_ndcg_ci90_high",
        help="Column name in processed CSV used for CI high bound",
    )
    ap.add_argument(
        "--outlier-indicator",
        choices=["ci", "delta"],
        default="ci",
        help="Primary signal for outlier flag: CI low bound or raw delta",
    )
    ap.add_argument(
        "--ci-threshold",
        type=float,
        default=DEFAULT_MINER_CI_THRESHOLD,
        help="Outlier threshold on CI low bound when --outlier-indicator=ci",
    )
    ap.add_argument(
        "--doc-type",
        choices=["mono", "bi", "all"],
        default="all",
        help="Filter by document-index type",
    )
    ap.add_argument(
        "--method",
        choices=["embed", "word", "all"],
        default="embed",
        help="Filter by retrieval method in raw results",
    )
    ap.add_argument(
        "--non-english-docs-only",
        action="store_true",
        default=True,
        help="Keep only settings where doc language set excludes EN",
    )
    ap.add_argument(
        "--include-english-docs",
        dest="non_english_docs_only",
        action="store_false",
        help="Do not filter out EN doc settings",
    )
    ap.add_argument(
        "--method-comparison",
        action="store_true",
        help="Also emit method_comparison.csv when both embed and word rows are available",
    )
    ap.add_argument(
        "--fill-with-lowest",
        action="store_true",
        help="If outliers are fewer than --top-n, fill cases.csv with lowest-delta non-outliers",
    )
    ap.add_argument(
        "--best-top-n",
        type=int,
        default=0,
        help="If >0, also emit top-performing cases to --best-cases-file",
    )
    ap.add_argument(
        "--best-cases-file",
        default="cases_best.csv",
        help="Filename under --out-dir for top-performing case output",
    )
    ap.add_argument(
        "--best-indicator",
        choices=["ci", "delta"],
        default="ci",
        help="Primary sort key for top-performing output",
    )
    ap.add_argument(
        "--best-delta-min",
        type=float,
        default=0.0,
        help="Minimum delta required for top-performing output",
    )
    ap.add_argument(
        "--best-ci-low-min",
        type=float,
        default=0.0,
        help="Minimum CI low bound required for top-performing output",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_miner(args)


if __name__ == "__main__":
    main()
