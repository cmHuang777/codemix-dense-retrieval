#!/usr/bin/env python3
"""Analyze query-level mixing gains against doc purity buckets."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from micro_case_common import (
    DEFAULT_MINE_OUT,
    DEFAULT_DOC_PURITY_OUT,
    DEFAULT_INSPECT_K,
    DEFAULT_INSPECT_OUT,
    DEFAULT_INSPECT_RANK_DEPTH,
    DEFAULT_QRELS,
    DEFAULT_QUERIES_DIR,
    DEFAULT_QUERY_CACHE_ROOT,
    count_word_tokens,
    fmt,
    geometry_features,
    is_finite,
    load_embeddings,
    load_qid_list,
    load_qrels,
    load_queries_tsv,
    load_trec_run,
    parse_float,
    percentile,
    read_csv_rows,
    read_doc_purity_features,
    recall_at_k,
    ndcg_at_k,
    first_rel_rank,
    to_lambda,
    write_csv_rows,
)

DEFAULT_CASES = DEFAULT_MINE_OUT / "cases_best.csv"
DEFAULT_RANK_MISS = 101.0
SUMMARY_FIELDNAMES = (
    'case_id',
    'pair',
    'doc_lang',
    'rel_bucket',
    'query_quality_label',
    'n_queries',
    'mean_delta_ndcg10',
    'median_delta_ndcg10',
    'pos_delta_rate',
    'mean_delta_recall10',
    'mean_first_rel_rank_gain',
    'median_first_rel_rank_gain',
    'ci90_low',
    'ci90_high',
)
QUERY_EFFECT_FIELDNAMES = (
    'case_id',
    'pair',
    'doc_lang',
    'qid',
    'q_lang_a',
    'q_lang_b',
    'query_a',
    'query_b',
    'metric_source',
    'tok_a',
    'tok_b',
    'len_ratio',
    'endpoint_cos',
    'query_quality_label',
    'ndcg_end',
    'ndcg_mix',
    'delta_ndcg10',
    'recall_end',
    'recall_mix',
    'delta_recall10',
    'first_rel_rank_end',
    'first_rel_rank_mix',
    'first_rel_rank_shift',
    'num_rel_docs',
    'num_rel_pure',
    'num_rel_mixed_light',
    'num_rel_mixed_clear',
    'num_rel_indeterminate',
    'rel_bucket',
    'first_rel_docid_end',
    'first_rel_docid_end_purity',
    'first_rel_docid_mix',
    'first_rel_docid_mix_purity',
    'mixed_top1_docid',
    'mixed_top1_purity',
)


def qid_sort_key(qid: str) -> Tuple[int, object]:
    qid_text = (qid or '').strip()
    return (0, int(qid_text)) if qid_text.isdigit() else (1, qid_text)



def load_perquery_eval_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return {}
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        qid_col = field_map.get('qid') or field_map.get('query-id') or field_map.get('query_id')
        ndcg_col = field_map.get('ndcg@10')
        recall_col = field_map.get('r@10') or field_map.get('recall@10')
        if not qid_col or not ndcg_col or not recall_col:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for row in reader:
            qid = str(row.get(qid_col, '')).strip()
            if not qid:
                continue
            ndcg = parse_float(row.get(ndcg_col))
            recall = parse_float(row.get(recall_col))
            if not is_finite(ndcg) or not is_finite(recall):
                continue
            out[qid] = {'ndcg10': float(ndcg), 'recall10': float(recall)}
        return out



def first_rel_docid(entries: Sequence[Tuple[int, str, float]], qrels: Mapping[str, int], depth: int) -> str:
    for _, docid, _ in entries[:depth]:
        if qrels.get(docid, 0) > 0:
            return docid
    return ''



def purity_label(lookup: Mapping[Tuple[str, str], Mapping[str, str]], case_id: str, docid: str) -> str:
    if not case_id or not docid:
        return ''
    row = lookup.get((case_id, docid), {})
    return str(row.get('doc_purity_label', '')).strip()



def classify_rel_bucket(counts: Mapping[str, int]) -> str:
    if counts.get('mixed_L_clear', 0) > 0:
        return 'has_rel_mixed_clear'
    if counts.get('mixed_L_light', 0) > 0:
        return 'has_rel_mixed_light_only'
    if counts.get('pure_L', 0) > 0 and counts.get('pure_L', 0) == counts.get('num_rel_docs', 0):
        return 'all_rel_pure'
    return 'indeterminate_only'



def classify_query_quality(len_ratio: float, endpoint_cos: float) -> str:
    if is_finite(len_ratio) and (len_ratio < 0.67 or len_ratio > 1.50):
        return 'suspect_translation'
    if is_finite(endpoint_cos) and endpoint_cos < 0.55:
        return 'suspect_translation'
    if is_finite(len_ratio) and is_finite(endpoint_cos):
        return 'clean_translation'
    return 'unknown'



def capped_rank(rank: float, miss_rank: float = DEFAULT_RANK_MISS) -> float:
    return miss_rank if math.isinf(rank) else float(rank)



def rank_shift(mix_rank: float, end_rank: float, miss_rank: float = DEFAULT_RANK_MISS) -> float:
    return capped_rank(mix_rank, miss_rank) - capped_rank(end_rank, miss_rank)



def rank_gain(end_rank: float, mix_rank: float) -> float:
    end_val = capped_rank(end_rank)
    mix_val = capped_rank(mix_rank)
    return end_val - mix_val



def bootstrap_mean_ci(values: Sequence[float], seed: int = 0, samples: int = 1000) -> Tuple[float, float]:
    clean = [float(v) for v in values if is_finite(float(v))]
    if not clean:
        return (math.nan, math.nan)
    if len(clean) == 1:
        return (clean[0], clean[0])
    rng = random.Random(seed)
    means: List[float] = []
    n = len(clean)
    for _ in range(samples):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (percentile(means, 5.0), percentile(means, 95.0))



def md_cell(value: object, limit: int = 120) -> str:
    text = str(value if value is not None else '')
    text = text.replace('\n', ' ').replace('|', '\\|').strip()
    if limit > 0 and len(text) > limit:
        return text[: limit - 1] + '…'
    return text



def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = []
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('|' + '|'.join('---' for _ in headers) + '|')
    for row in rows:
        out.append('| ' + ' | '.join(md_cell(v) for v in row) + ' |')
    return '\n'.join(out)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Analyze query-level purity effects for micro cases.')
    parser.add_argument('--cases', default=str(DEFAULT_CASES), help='Path to cases.csv or cases_best.csv')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--case-id', help='Optional single case_id to process')
    group.add_argument('--all-cases', action='store_true', help='Process all rows in --cases (default behavior)')
    parser.add_argument('--qrels', default=str(DEFAULT_QRELS), help='Qrels TSV')
    parser.add_argument('--queries-dir', default=str(DEFAULT_QUERIES_DIR), help='Directory containing queries.<lang>.tsv')
    parser.add_argument('--query-cache-root', default=str(DEFAULT_QUERY_CACHE_ROOT), help='Embedding cache root')
    parser.add_argument('--doc-purity', default=str(DEFAULT_DOC_PURITY_OUT), help='Doc purity CSV from build_doc_purity_features.py')
    parser.add_argument('--qid-list', default='', help='Optional qid list to restrict analysis')
    parser.add_argument('--k', type=int, default=DEFAULT_INSPECT_K, help='Metric cutoff k')
    parser.add_argument('--rank-depth', type=int, default=DEFAULT_INSPECT_RANK_DEPTH, help='Depth for first relevant rank')
    parser.add_argument('--out-dir', default=str(DEFAULT_INSPECT_OUT), help='Output directory root')
    parser.add_argument('--summary-out', default=str(Path(DEFAULT_INSPECT_OUT) / 'purity_summary_by_case.csv'), help='Summary CSV path')
    return parser



def main() -> None:
    args = build_parser().parse_args()
    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise SystemExit(f'Cases CSV not found: {cases_path}')

    case_rows = read_csv_rows(cases_path)
    if not case_rows:
        raise SystemExit(f'Cases CSV is empty: {cases_path}')

    if args.case_id:
        selected_cases = [row for row in case_rows if (row.get('case_id') or '').strip() == args.case_id]
        if not selected_cases:
            raise SystemExit(f'case_id not found in {cases_path}: {args.case_id}')
    else:
        selected_cases = case_rows

    allowed_qids = load_qid_list(Path(args.qid_list)) if str(args.qid_list).strip() else None
    qrels = load_qrels(Path(args.qrels))
    purity_lookup = read_doc_purity_features(Path(args.doc_purity))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []

    for case_row in selected_cases:
        case_id = str(case_row.get('case_id', '')).strip()
        if not case_id:
            continue
        pair = str(case_row.get('pair', '')).strip()
        doc_lang = str(case_row.get('doc_lang', '')).strip().lower()
        q_lang_a = str(case_row.get('q_lang_a', '')).strip().lower()
        q_lang_b = str(case_row.get('q_lang_b', '')).strip().lower()

        run_end = load_trec_run(Path(case_row.get('endpoint_run_path', '')))
        run_mix = load_trec_run(Path(case_row.get('mixed_run_path', '')))
        endpoint_perquery_path = str(case_row.get('endpoint_perquery_path', '')).strip()
        mixed_perquery_path = str(case_row.get('mixed_perquery_path', '')).strip()
        endpoint_metrics = load_perquery_eval_metrics(Path(endpoint_perquery_path)) if endpoint_perquery_path else {}
        mixed_metrics = load_perquery_eval_metrics(Path(mixed_perquery_path)) if mixed_perquery_path else {}

        if endpoint_metrics and mixed_metrics:
            all_qids = sorted((set(endpoint_metrics) & set(mixed_metrics) & set(qrels.keys())), key=qid_sort_key)
        else:
            all_qids = sorted(((set(run_end) | set(run_mix)) & set(qrels.keys())), key=qid_sort_key)
        if allowed_qids is not None:
            all_qids = [qid for qid in all_qids if qid in allowed_qids]
        if not all_qids:
            continue

        queries_a = load_queries_tsv(Path(args.queries_dir) / f'queries.{q_lang_a}.tsv') if q_lang_a else {}
        queries_b = load_queries_tsv(Path(args.queries_dir) / f'queries.{q_lang_b}.tsv') if q_lang_b else {}
        emb_a = load_embeddings(Path(args.query_cache_root), q_lang_a) if q_lang_a else {}
        emb_b = load_embeddings(Path(args.query_cache_root), q_lang_b) if q_lang_b else {}
        lam = to_lambda(parse_float(case_row.get('lambda_star')))

        query_rows: List[Dict[str, object]] = []
        for qid in all_qids:
            rels = {docid: rel for docid, rel in qrels.get(qid, {}).items() if rel > 0}
            if not rels:
                continue

            end_entries = run_end.get(qid, [])
            mix_entries = run_mix.get(qid, [])
            end_docids = [docid for _, docid, _ in end_entries]
            mix_docids = [docid for _, docid, _ in mix_entries]

            if qid in endpoint_metrics and qid in mixed_metrics:
                ndcg_end = float(endpoint_metrics[qid]['ndcg10'])
                ndcg_mix = float(mixed_metrics[qid]['ndcg10'])
                recall_end = float(endpoint_metrics[qid]['recall10'])
                recall_mix = float(mixed_metrics[qid]['recall10'])
                metric_source = 'evaluate_perquery'
            else:
                ndcg_end = ndcg_at_k(end_docids, rels, args.k)
                ndcg_mix = ndcg_at_k(mix_docids, rels, args.k)
                recall_end = recall_at_k(end_docids, rels, args.k)
                recall_mix = recall_at_k(mix_docids, rels, args.k)
                metric_source = 'recomputed_from_run_qrels'

            first_end_rank = first_rel_rank(end_entries, rels, args.rank_depth)
            first_mix_rank = first_rel_rank(mix_entries, rels, args.rank_depth)
            first_end_docid = first_rel_docid(end_entries, rels, args.rank_depth)
            first_mix_docid = first_rel_docid(mix_entries, rels, args.rank_depth)
            mixed_top1_docid = mix_entries[0][1] if mix_entries else ''

            query_a = queries_a.get(qid, '')
            query_b = queries_b.get(qid, '')
            tok_a = count_word_tokens(query_a, q_lang_a)
            tok_b = count_word_tokens(query_b, q_lang_b)
            if tok_a > 0 and tok_b > 0:
                len_ratio = tok_a / tok_b
            elif tok_a > 0 and tok_b == 0:
                len_ratio = float('inf')
            elif tok_a == 0 and tok_b > 0:
                len_ratio = 0.0
            else:
                len_ratio = math.nan
            endpoint_cos = geometry_features(emb_a.get(qid), emb_b.get(qid), lam)[4]
            query_quality = classify_query_quality(len_ratio, endpoint_cos)

            rel_counts = defaultdict(int)
            for docid in rels:
                label = purity_label(purity_lookup, case_id, docid) or 'indeterminate_latin_script'
                rel_counts[label] += 1
            rel_counts['num_rel_docs'] = len(rels)
            rel_bucket = classify_rel_bucket(rel_counts)

            query_rows.append(
                {
                    'case_id': case_id,
                    'pair': pair,
                    'doc_lang': doc_lang,
                    'qid': qid,
                    'q_lang_a': q_lang_a,
                    'q_lang_b': q_lang_b,
                    'query_a': query_a,
                    'query_b': query_b,
                    'metric_source': metric_source,
                    'tok_a': tok_a,
                    'tok_b': tok_b,
                    'len_ratio': len_ratio,
                    'endpoint_cos': endpoint_cos,
                    'query_quality_label': query_quality,
                    'ndcg_end': ndcg_end,
                    'ndcg_mix': ndcg_mix,
                    'delta_ndcg10': ndcg_mix - ndcg_end,
                    'recall_end': recall_end,
                    'recall_mix': recall_mix,
                    'delta_recall10': recall_mix - recall_end,
                    'first_rel_rank_end': first_end_rank,
                    'first_rel_rank_mix': first_mix_rank,
                    'first_rel_rank_shift': rank_shift(first_mix_rank, first_end_rank),
                    'num_rel_docs': len(rels),
                    'num_rel_pure': rel_counts.get('pure_L', 0),
                    'num_rel_mixed_light': rel_counts.get('mixed_L_light', 0),
                    'num_rel_mixed_clear': rel_counts.get('mixed_L_clear', 0),
                    'num_rel_indeterminate': rel_counts.get('indeterminate_latin_script', 0),
                    'rel_bucket': rel_bucket,
                    'first_rel_docid_end': first_end_docid,
                    'first_rel_docid_end_purity': purity_label(purity_lookup, case_id, first_end_docid),
                    'first_rel_docid_mix': first_mix_docid,
                    'first_rel_docid_mix_purity': purity_label(purity_lookup, case_id, first_mix_docid),
                    'mixed_top1_docid': mixed_top1_docid,
                    'mixed_top1_purity': purity_label(purity_lookup, case_id, mixed_top1_docid),
                }
            )

        if not query_rows:
            continue

        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        query_out = case_dir / 'query_purity_effect.csv'
        write_csv_rows(query_out, query_rows, fieldnames=QUERY_EFFECT_FIELDNAMES)

        grouped = defaultdict(list)
        for row in query_rows:
            key = (row['rel_bucket'], row['query_quality_label'])
            grouped[key].append(row)

        case_summary_rows: List[Dict[str, object]] = []
        for (rel_bucket, query_quality), rows in sorted(grouped.items()):
            deltas = [float(row['delta_ndcg10']) for row in rows if is_finite(float(row['delta_ndcg10']))]
            delta_recall = [float(row['delta_recall10']) for row in rows if is_finite(float(row['delta_recall10']))]
            gains = [rank_gain(float(row['first_rel_rank_end']), float(row['first_rel_rank_mix'])) for row in rows]
            ci_low, ci_high = bootstrap_mean_ci(deltas, seed=len(rows) + len(case_id))
            summary_row = {
                'case_id': case_id,
                'pair': pair,
                'doc_lang': doc_lang,
                'rel_bucket': rel_bucket,
                'query_quality_label': query_quality,
                'n_queries': len(rows),
                'mean_delta_ndcg10': (sum(deltas) / len(deltas)) if deltas else math.nan,
                'median_delta_ndcg10': statistics.median(deltas) if deltas else math.nan,
                'pos_delta_rate': (sum(1 for v in deltas if v > 0) / len(deltas)) if deltas else math.nan,
                'mean_delta_recall10': (sum(delta_recall) / len(delta_recall)) if delta_recall else math.nan,
                'mean_first_rel_rank_gain': (sum(gains) / len(gains)) if gains else math.nan,
                'median_first_rel_rank_gain': statistics.median(gains) if gains else math.nan,
                'ci90_low': ci_low,
                'ci90_high': ci_high,
            }
            case_summary_rows.append(summary_row)
            summary_rows.append(summary_row)

        top_mixed = sorted(
            [row for row in query_rows if row['rel_bucket'] != 'all_rel_pure'],
            key=lambda row: float(row['delta_ndcg10']),
            reverse=True,
        )[:10]
        top_pure = sorted(
            [row for row in query_rows if row['rel_bucket'] == 'all_rel_pure'],
            key=lambda row: float(row['delta_ndcg10']),
            reverse=True,
        )[:10]
        top_drops = sorted(query_rows, key=lambda row: float(row['delta_ndcg10']))[:10]

        report_lines = [
            f'# Purity Analysis: {case_id}',
            '',
            f'- pair: `{pair}`',
            f'- doc_lang: `{doc_lang}`',
            f'- qids_analyzed: `{len(query_rows)}`',
            '',
            '## Summary',
            '',
            markdown_table(
                ['rel_bucket', 'query_quality', 'n', 'mean_d_ndcg', 'median_d_ndcg', 'pos_rate', 'mean_rank_gain', 'median_rank_gain', 'ci90'],
                [
                    [
                        row['rel_bucket'],
                        row['query_quality_label'],
                        row['n_queries'],
                        fmt(row['mean_delta_ndcg10'], 4),
                        fmt(row['median_delta_ndcg10'], 4),
                        fmt(row['pos_delta_rate'], 4),
                        fmt(row['mean_first_rel_rank_gain'], 4),
                        fmt(row['median_first_rel_rank_gain'], 4),
                        f"[{fmt(row['ci90_low'], 4)}, {fmt(row['ci90_high'], 4)}]",
                    ]
                    for row in case_summary_rows
                ],
            ),
            '',
            '## Top Gains With Mixed Relevant Docs',
            '',
            markdown_table(
                ['qid', 'd_ndcg', 'bucket', 'quality', 'mix_first_rel_purity', 'query_a'],
                [
                    [
                        row['qid'],
                        fmt(row['delta_ndcg10'], 4),
                        row['rel_bucket'],
                        row['query_quality_label'],
                        row['first_rel_docid_mix_purity'],
                        row['query_a'],
                    ]
                    for row in top_mixed
                ]
                or [['', '', '', '', '', '']],
            ),
            '',
            '## Top Gains With Pure Relevant Docs',
            '',
            markdown_table(
                ['qid', 'd_ndcg', 'quality', 'mix_first_rel_purity', 'query_a'],
                [
                    [
                        row['qid'],
                        fmt(row['delta_ndcg10'], 4),
                        row['query_quality_label'],
                        row['first_rel_docid_mix_purity'],
                        row['query_a'],
                    ]
                    for row in top_pure
                ]
                or [['', '', '', '', '']],
            ),
            '',
            '## Top Drops',
            '',
            markdown_table(
                ['qid', 'd_ndcg', 'bucket', 'quality', 'mix_first_rel_purity', 'query_a'],
                [
                    [
                        row['qid'],
                        fmt(row['delta_ndcg10'], 4),
                        row['rel_bucket'],
                        row['query_quality_label'],
                        row['first_rel_docid_mix_purity'],
                        row['query_a'],
                    ]
                    for row in top_drops
                ]
                or [['', '', '', '', '', '']],
            ),
            '',
        ]
        (case_dir / 'purity_analysis.md').write_text('\n'.join(report_lines), encoding='utf-8')

    write_csv_rows(Path(args.summary_out), summary_rows, fieldnames=SUMMARY_FIELDNAMES)


if __name__ == '__main__':
    main()
