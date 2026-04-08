#!/usr/bin/env python3
"""Shared helpers for micro-level case mining and inspection."""

from __future__ import annotations

import csv
import html
import math
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected but optional
    np = None
try:
    import stanza
except Exception:  # pragma: no cover - stanza is optional at runtime
    stanza = None
try:
    from langid.langid import LanguageIdentifier, model as _LANGID_MODEL
except Exception:  # pragma: no cover - langid is optional at runtime
    LanguageIdentifier = None
    _LANGID_MODEL = None

TOOL_DIR = Path(__file__).resolve().parent
TEST_ROOT = TOOL_DIR.parent

DEFAULT_RAW_RESULTS = TEST_ROOT / "compiled_results" / "full_mmarco_results_20260210.csv"
DEFAULT_PROCESSED_RESULTS = TEST_ROOT / "compiled_results" / "full_mmarco_processed_results_20260210.csv"
DEFAULT_RESULTS_ROOT = TEST_ROOT / "results" / "mmarco_full"

# External data paths (must be provided by user via CLI args or environment)
DEFAULT_RUN_ROOT = None  # User must provide via --run-root
DEFAULT_QRELS = None  # User must provide via --qrels
DEFAULT_QUERIES_DIR = None  # User must provide via --queries-dir
DEFAULT_QUERY_CACHE_ROOT = None  # User must provide via --query-cache-root

DEFAULT_MINE_OUT = TOOL_DIR / "micro_cases"
DEFAULT_INSPECT_OUT = TOOL_DIR / "micro_reports"
DEFAULT_DOC_PURITY_OUT = TOOL_DIR / "micro_cases" / "doc_purity_features.csv"

# Tunable defaults for miner/inspector.
DEFAULT_MINER_TOP_N = 30
DEFAULT_MINER_DELTA_THRESHOLD = -0.2
DEFAULT_MINER_CI_THRESHOLD = 0.0
DEFAULT_INSPECT_WORST_N = 100
DEFAULT_INSPECT_CONTROL_N = 20
DEFAULT_INSPECT_K = 10
DEFAULT_INSPECT_RANK_DEPTH = 50
DEFAULT_REPORT_TOP_WORST = 20
DEFAULT_REPORT_DIFF_BLOCKS = 20
REPORT_QUERY_TEXT_CELL_LIMIT = 300
REPORT_DOC_LANG_CELL_LIMIT = 24
REPORT_DELTA_QUANTILES = (25.0, 50.0, 75.0)

# Shared numeric tolerances and scales.
LAMBDA_AS_PERCENT_EPS = 1e-9
ENDPOINT_LAMBDA_TOL = 1e-6
ROUND_TO_INT_EPS = 1e-9
METRIC_PERCENT_SCALE = 100.0
INF_RANK_SENTINEL = 9999.0
DOC_META_SNIFF_BYTES = 4096
MARKDOWN_CELL_DEFAULT_LIMIT = 180

# Failure-label tuning.
FAILURE_LABEL_ORDER = (
    "IndexLeakage",
    "TranslationDivergence",
    "RecallDrop",
    "RankDrop",
    "Unclassified",
)


@dataclass(frozen=True)
class FailureLabelConfig:
    mismatch_rate_gt: float = 0.0
    endpoint_cos_lt: float = 0.5
    len_ratio_min: float = 0.5
    len_ratio_max: float = 1.50
    delta_recall_lt: float = 0.0
    rankdrop_delta_ndcg_lt: float = 0.0
    rankdrop_delta_recall_ge: float = 0.0


DEFAULT_FAILURE_LABEL_CONFIG = FailureLabelConfig()
FAILURE_LABEL_UNCLASSIFIED = "Unclassified"

LANG_NAME_TO_CODE = {
    "amharic": "am",
    "am": "am",
    "arabic": "ar",
    "ar": "ar",
    "burmese": "my",
    "myanmar": "my",
    "my": "my",
    "chinese": "zh",
    "cn": "zh",
    "zh": "zh",
    "de": "de",
    "german": "de",
    "dutch": "nl",
    "nl": "nl",
    "en": "en",
    "english": "en",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "hi": "hi",
    "hindi": "hi",
    "id": "id",
    "indonesian": "id",
    "it": "it",
    "italian": "it",
    "ja": "ja",
    "japanese": "ja",
    "km": "km",
    "khmer": "km",
    "ku": "ku",
    "kurdish": "ku",
    "ne": "ne",
    "nepali": "ne",
    "pt": "pt",
    "portuguese": "pt",
    "ru": "ru",
    "russian": "ru",
    "si": "si",
    "sinhala": "si",
    "shn": "shn",
    "shan": "shn",
    "sl": "sl",
    "slovene": "sl",
    "sw": "sw",
    "swahili": "sw",
    "vi": "vi",
    "vietnamese": "vi",
}
KNOWN_CODES = set(LANG_NAME_TO_CODE.values())

PHASE_TIMESTAMP_RE = re.compile(
    r"_(?:dev|test|validation|val|train)[-_]\d{8}[-_]\d{6}(?:[-_]\d+)?",
    re.IGNORECASE,
)
ALPHA_RE = re.compile(r"cm-alpha-([0-9]*\.?[0-9]+)", re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_HANDLE = re.compile(r"[@#]\w+")
RE_WORD_TOKEN = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)
_STANZA_TOKENIZERS: Dict[str, object] = {}
_LANGID_IDENTIFIERS: Dict[str, object] = {}
_LANGID_SPAN_CACHE: Dict[Tuple[str, str], bool] = {}
RE_ASCII_WORD_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HTML_ENTITY = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);")
LATIN_SCRIPT_LANGS = {"de", "en", "es", "fr", "id", "it", "nl", "pt", "sl", "sw", "vi"}
DOC_PURITY_FIELDNAMES = (
    "case_id",
    "docid",
    "doc_lang",
    "text_source",
    "text_chars",
    "alpha_chars_total",
    "ascii_alpha_chars",
    "ascii_alpha_ratio",
    "word_tokens_total",
    "ascii_word_tokens",
    "ascii_word_ratio",
    "ascii_run_max",
    "ascii_run_count_ge2",
    "ascii_run_count_ge3",
    "title_ascii_run_max",
    "url_count",
    "email_count",
    "handle_count",
    "has_english_span",
    "english_span_level",
    "doc_purity_label",
)


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def normalize_pair(pair: str) -> str:
    text = (pair or "").strip().replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", "", text)
    return text.upper()


def split_pair_codes(pair: str) -> Tuple[str, str]:
    cleaned = normalize_pair(pair)
    parts = [p for p in re.split(r"[-/]", cleaned) if p]
    if len(parts) >= 2:
        return parts[0].lower(), parts[1].lower()
    if len(parts) == 1:
        return parts[0].lower(), parts[0].lower()
    return "", ""


def normalize_doc_mix(doc_mix: str) -> str:
    text = (doc_mix or "").strip()
    return re.sub(r"\s+", " ", text)


def parse_doc_mix_codes(doc_mix: str) -> List[str]:
    text = normalize_doc_mix(doc_mix)
    text = re.sub(r"\bdocs?\b", "", text, flags=re.IGNORECASE)
    tokens = [t.strip().lower() for t in re.split(r"[+,&/\-]", text) if t.strip()]
    out: List[str] = []
    for tok in tokens:
        code = LANG_NAME_TO_CODE.get(tok)
        if code and code not in out:
            out.append(code)
        elif tok in KNOWN_CODES and tok not in out:
            out.append(tok)
    return out


def infer_doc_codes(doc_mix: str, doc_index_id: str) -> List[str]:
    codes = parse_doc_mix_codes(doc_mix)
    if codes:
        return codes
    parts = doc_index_id.split("-")
    if len(parts) >= 3:
        code = LANG_NAME_TO_CODE.get(parts[2].lower())
        if code:
            return [code]
    return []


def doc_type_from_codes(codes: Sequence[str]) -> str:
    if len(codes) > 1:
        return "bi"
    return "mono"


def doc_lang_token(codes: Sequence[str]) -> str:
    if not codes:
        return ""
    return "+".join(codes)


def is_non_english_doc_setting(codes: Sequence[str], doc_mix: str) -> bool:
    if codes:
        return all(code != "en" for code in codes)
    return "EN" not in normalize_doc_mix(doc_mix).upper()


def parse_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float):
            return value
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def is_finite(value: float) -> bool:
    return value is not None and math.isfinite(value)


def is_language_word_token(token: str, drop_digit_tokens: bool = True) -> bool:
    if not token:
        return False
    if RE_URL.search(token) or RE_EMAIL.search(token) or RE_HANDLE.search(token):
        return False
    if drop_digit_tokens and any(ch.isdigit() for ch in token):
        return False
    return any(ch.isalpha() for ch in token)


def _is_han_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0x2CEB0 <= code <= 0x2EBEF
    )


def _split_script_runs(token: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    current_is_han: Optional[bool] = None
    for ch in token:
        if not ch.isalpha():
            if current:
                parts.append("".join(current))
                current = []
                current_is_han = None
            continue
        is_han = _is_han_char(ch)
        if current and current_is_han != is_han:
            parts.append("".join(current))
            current = [ch]
            current_is_han = is_han
        else:
            current.append(ch)
            current_is_han = is_han
    if current:
        parts.append("".join(current))
    return parts


def _get_stanza_tokenizer(lang: str):
    if stanza is None or not lang:
        return None
    if lang in _STANZA_TOKENIZERS:
        return _STANZA_TOKENIZERS[lang]
    try:
        tok = stanza.Pipeline(lang, processors="tokenize", tokenize_pretokenized=False, verbose=False)
    except Exception:
        tok = None
    _STANZA_TOKENIZERS[lang] = tok
    return tok


def extract_language_word_tokens(text: str, lang: str = "", drop_digit_tokens: bool = True) -> List[str]:
    if not text:
        return []

    raw_tokens: List[str] = []
    used_stanza = False

    tok = _get_stanza_tokenizer(lang.strip().lower())
    if tok is not None:
        try:
            doc = tok(text)
            raw_tokens = [t.text for sent in doc.sentences for t in sent.tokens if t.text]
            used_stanza = True
        except Exception:
            raw_tokens = []

    if not raw_tokens:
        raw_tokens = RE_WORD_TOKEN.findall(text)

    pieces_out: List[str] = []
    for token in raw_tokens:
        for piece in _split_script_runs(token):
            if not is_language_word_token(piece, drop_digit_tokens=drop_digit_tokens):
                continue
            if not used_stanza and all(_is_han_char(ch) for ch in piece):
                # Fallback when no CJK tokenizer is available: approximate by character tokens.
                pieces_out.extend(list(piece))
                continue
            pieces_out.append(piece)
    return pieces_out


def count_word_tokens(text: str, lang: str = "", drop_digit_tokens: bool = True) -> int:
    return len(extract_language_word_tokens(text, lang=lang, drop_digit_tokens=drop_digit_tokens))


def normalize_text_for_purity(text: object) -> str:
    if text is None:
        return ""
    value = html.unescape(str(text))
    value = RE_HTML_TAG.sub(" ", value)
    value = RE_HTML_ENTITY.sub(" ", value)
    value = RE_URL.sub(" ", value)
    value = RE_EMAIL.sub(" ", value)
    value = RE_HANDLE.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def _is_ascii_word_token(token: str) -> bool:
    return bool(RE_ASCII_WORD_TOKEN.fullmatch(token or ""))


def _ascii_run_stats(tokens: Sequence[str]) -> Tuple[int, int, int]:
    max_run = 0
    run_count_ge2 = 0
    run_count_ge3 = 0
    current = 0
    for token in tokens:
        if _is_ascii_word_token(token):
            current += 1
            if current > max_run:
                max_run = current
            continue
        if current >= 2:
            run_count_ge2 += 1
        if current >= 3:
            run_count_ge3 += 1
        current = 0
    if current >= 2:
        run_count_ge2 += 1
    if current >= 3:
        run_count_ge3 += 1
    return max_run, run_count_ge2, run_count_ge3


def _ascii_token_runs(tokens: Sequence[str]) -> List[Tuple[str, ...]]:
    runs: List[Tuple[str, ...]] = []
    current: List[str] = []
    for token in tokens:
        if _is_ascii_word_token(token):
            current.append(token)
            continue
        if current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return runs


def _get_langid_identifier(lang_code: str):
    code = normalize_doc_lang(lang_code)
    if not code or code == "en" or LanguageIdentifier is None or _LANGID_MODEL is None:
        return None
    identifier = _LANGID_IDENTIFIERS.get(code)
    if identifier is None:
        identifier = LanguageIdentifier.from_modelstring(_LANGID_MODEL, norm_probs=True)
        identifier.set_languages(["en", code])
        _LANGID_IDENTIFIERS[code] = identifier
    return identifier


def _span_is_english_for_lang(span_text: str, lang_code: str) -> bool:
    text = re.sub(r"\s+", " ", span_text or "").strip()
    code = normalize_doc_lang(lang_code)
    if not text or not code or code == "en":
        return False
    cache_key = (code, text.casefold())
    cached = _LANGID_SPAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    identifier = _get_langid_identifier(code)
    if identifier is None:
        _LANGID_SPAN_CACHE[cache_key] = False
        return False
    try:
        ranked = identifier.rank(text)
    except Exception:
        _LANGID_SPAN_CACHE[cache_key] = False
        return False
    if not ranked:
        _LANGID_SPAN_CACHE[cache_key] = False
        return False
    best_lang, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    is_english = bool(best_lang == "en" and best_score > second_score)
    _LANGID_SPAN_CACHE[cache_key] = is_english
    return is_english


def _english_run_stats_for_latin_lang(tokens: Sequence[str], lang_code: str) -> Tuple[int, int, int, int]:
    english_token_count = 0
    english_run_max = 0
    english_run_count_ge2 = 0
    english_run_count_ge3 = 0
    current = 0
    for token in tokens:
        if _is_ascii_word_token(token) and _span_is_english_for_lang(token, lang_code):
            english_token_count += 1
            current += 1
            if current > english_run_max:
                english_run_max = current
            continue
        if current >= 2:
            english_run_count_ge2 += 1
        if current >= 3:
            english_run_count_ge3 += 1
        current = 0
    if current >= 2:
        english_run_count_ge2 += 1
    if current >= 3:
        english_run_count_ge3 += 1
    return english_token_count, english_run_max, english_run_count_ge2, english_run_count_ge3


def classify_doc_purity(
    *,
    title: str,
    body: str,
    lang: str = "",
    text_source: str = "",
) -> Dict[str, object]:
    clean_title = normalize_text_for_purity(title)
    clean_body = normalize_text_for_purity(body)
    combined = " ".join(part for part in (clean_title, clean_body) if part).strip()
    letters = [ch for ch in combined if ch.isalpha()]
    alpha_chars_total = len(letters)
    ascii_alpha_chars = sum(1 for ch in letters if ch in string.ascii_letters)
    ascii_alpha_ratio = (ascii_alpha_chars / alpha_chars_total) if alpha_chars_total else math.nan

    tokens = extract_language_word_tokens(combined, lang=lang)
    title_tokens = extract_language_word_tokens(clean_title, lang=lang)
    word_tokens_total = len(tokens)
    ascii_word_tokens = sum(1 for token in tokens if _is_ascii_word_token(token))
    ascii_word_ratio = (ascii_word_tokens / word_tokens_total) if word_tokens_total else math.nan
    ascii_run_max, ascii_run_count_ge2, ascii_run_count_ge3 = _ascii_run_stats(tokens)
    title_ascii_run_max, _, _ = _ascii_run_stats(title_tokens)

    lang_code = normalize_doc_lang(lang)
    english_span_level = "none"
    effective_word_tokens = ascii_word_tokens
    effective_word_ratio = ascii_word_ratio
    effective_run_max = ascii_run_max
    effective_run_count_ge2 = ascii_run_count_ge2
    effective_run_count_ge3 = ascii_run_count_ge3
    effective_title_run_max = title_ascii_run_max

    latin_langid_active = lang_code in LATIN_SCRIPT_LANGS and _get_langid_identifier(lang_code) is not None

    if latin_langid_active:
        effective_word_tokens, effective_run_max, effective_run_count_ge2, effective_run_count_ge3 = (
            _english_run_stats_for_latin_lang(tokens, lang_code)
        )
        _, effective_title_run_max, _, _ = _english_run_stats_for_latin_lang(title_tokens, lang_code)
        effective_word_ratio = (effective_word_tokens / word_tokens_total) if word_tokens_total else math.nan
        if effective_word_tokens > 0:
            english_span_level = "light"
        if effective_run_max >= 2 or effective_title_run_max >= 1:
            english_span_level = "clear"
        if english_span_level == "none":
            doc_purity_label = "pure_L"
        elif english_span_level == "light":
            doc_purity_label = "mixed_L_light"
        else:
            doc_purity_label = "mixed_L_clear"
    else:
        if ascii_word_tokens > 0:
            english_span_level = "light"
        if (
            ascii_run_max >= 2
            or title_ascii_run_max >= 2
            or (is_finite(ascii_word_ratio) and ascii_word_ratio >= 0.03)
        ):
            english_span_level = "clear"
        if lang_code in LATIN_SCRIPT_LANGS:
            if english_span_level == "clear":
                doc_purity_label = "mixed_L_clear"
            else:
                doc_purity_label = "indeterminate_latin_script"
        elif english_span_level == "none":
            doc_purity_label = "pure_L"
        elif english_span_level == "light":
            doc_purity_label = "mixed_L_light"
        else:
            doc_purity_label = "mixed_L_clear"

    if lang_code not in LATIN_SCRIPT_LANGS:
        if english_span_level == "none":
            doc_purity_label = "pure_L"
        elif english_span_level == "light":
            doc_purity_label = "mixed_L_light"
        else:
            doc_purity_label = "mixed_L_clear"

    raw_text = f"{title or ''} {body or ''}".strip()
    if latin_langid_active:
        has_english_span = int(effective_run_max >= 2 or effective_title_run_max >= 1)
    else:
        has_english_span = int(effective_run_max >= 2 or effective_title_run_max >= 2)
    return {
        "text_source": text_source or ("title+body" if clean_body else "title_only"),
        "text_chars": len(combined),
        "alpha_chars_total": alpha_chars_total,
        "ascii_alpha_chars": ascii_alpha_chars,
        "ascii_alpha_ratio": ascii_alpha_ratio,
        "word_tokens_total": word_tokens_total,
        "ascii_word_tokens": ascii_word_tokens,
        "ascii_word_ratio": ascii_word_ratio,
        "ascii_run_max": ascii_run_max,
        "ascii_run_count_ge2": ascii_run_count_ge2,
        "ascii_run_count_ge3": ascii_run_count_ge3,
        "title_ascii_run_max": title_ascii_run_max,
        "url_count": len(RE_URL.findall(raw_text)),
        "email_count": len(RE_EMAIL.findall(raw_text)),
        "handle_count": len(RE_HANDLE.findall(raw_text)),
        "has_english_span": has_english_span,
        "english_span_level": english_span_level,
        "doc_purity_label": doc_purity_label,
    }


def to_lambda(mix_ratio: float) -> float:
    if not is_finite(mix_ratio):
        return math.nan
    if mix_ratio > 1.0 + LAMBDA_AS_PERCENT_EPS:
        return mix_ratio / METRIC_PERCENT_SCALE
    return mix_ratio


def is_endpoint_lambda(value: float, tol: float = ENDPOINT_LAMBDA_TOL) -> bool:
    if not is_finite(value):
        return False
    return abs(value - 0.0) <= tol or abs(value - 1.0) <= tol


def normalize_alpha_token(value: object) -> Optional[str]:
    try:
        num = float(value)
    except Exception:
        return None
    if abs(num - round(num)) <= ROUND_TO_INT_EPS:
        return str(int(round(num)))
    text = f"{num:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def source_doc_index_id(source_file: str) -> str:
    p = PurePosixPath(source_file)
    if p.parts:
        return p.parts[0]
    return ""


def source_eval_path(result_root: Path, source_file: str) -> Path:
    p = Path(source_file)
    if p.is_absolute():
        return p
    return result_root / source_file


def source_perquery_path(result_root: Path, source_file: str) -> Path:
    eval_path = source_eval_path(result_root, source_file)
    name = eval_path.name
    if name.endswith("-agg.csv"):
        return eval_path.with_name(name[:-8] + "-perquery.csv")
    if name.endswith("_agg.csv"):
        return eval_path.with_name(name[:-8] + "_perquery.csv")
    return eval_path.with_suffix("")


def source_alpha_token(source_file: str) -> Optional[str]:
    name = PurePosixPath(source_file).name
    m = ALPHA_RE.search(name)
    if not m:
        return None
    return normalize_alpha_token(m.group(1))


def source_core_name(source_file: str) -> str:
    stem = Path(PurePosixPath(source_file).name).stem
    stem = PHASE_TIMESTAMP_RE.sub("", stem)
    stem = re.sub(r"[-_](agg|perquery)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[-_]{2,}", "-", stem)
    return stem.strip("-_")


def source_run_id(source_file: str) -> str:
    exp = source_doc_index_id(source_file)
    core = source_core_name(source_file)
    if exp:
        return f"{exp}/{core}"
    return core


def resolve_run_path(
    run_root: Path,
    source_file: str,
    method: str,
    mix_ratio: float,
) -> Path:
    exp = source_doc_index_id(source_file)
    core = source_core_name(source_file)
    alpha = source_alpha_token(source_file)
    if alpha is None:
        alpha = normalize_alpha_token(to_lambda(mix_ratio))

    exp_root = run_root / exp if exp else run_root
    candidates: List[Path] = []

    if method.lower() == "embed":
        if alpha:
            candidates.append(exp_root / "vector_mix" / f"cm-alpha-{alpha}.trec")
        if core:
            candidates.append(exp_root / "vector_mix" / f"{core}.trec")
            candidates.append(exp_root / f"{core}.trec")
    else:
        if core:
            candidates.append(exp_root / f"{core}.trec")
            candidates.append(exp_root / "word_mix" / f"{core}.trec")
            candidates.append(exp_root / "vector_mix" / f"{core}.trec")
        if alpha:
            candidates.append(exp_root / "vector_mix" / f"cm-alpha-{alpha}.trec")

    for cand in candidates:
        if cand.exists():
            return cand

    if candidates:
        return candidates[0]
    return exp_root / f"{core or 'run'}.trec"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def load_qid_list(path: Path) -> Set[str]:
    if not path.exists():
        raise SystemExit(f"QID list file not found: {path}")
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            out.add(line.split()[0])
    return out


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if pct <= 0:
        return sorted_values[0]
    if pct >= METRIC_PERCENT_SCALE:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * (pct / METRIC_PERCENT_SCALE)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        num = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(num):
        return ""
    return f"{num:.{digits}f}"


def select_processed_row(candidates: Sequence[Mapping[str, str]], delta: float) -> Optional[Mapping[str, str]]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    best = None
    best_dist = float("inf")
    for row in candidates:
        d = parse_float(row.get("delta_ndcg"))
        dist = abs(d - delta) if is_finite(d) and is_finite(delta) else float("inf")
        if dist < best_dist:
            best_dist = dist
            best = row
    return best if best is not None else candidates[0]


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    if not path.exists():
        raise SystemExit(f"qrels not found: {path}")
    table: Dict[str, Dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            head = parts[0].lower()
            if head in {"qid", "query-id", "query_id"}:
                continue
            if len(parts) >= 4:
                qid, docid, rel = parts[0], parts[2], parts[3]
            elif len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], parts[2]
            else:
                continue
            try:
                r = int(float(rel))
            except Exception:
                continue
            table[str(qid)][str(docid)] = r
    return table


def load_trec_run(path: Path) -> Dict[str, List[Tuple[int, str, float]]]:
    if not path.exists():
        raise SystemExit(f"Run file not found: {path}")
    data: Dict[str, List[Tuple[int, str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            docid = parts[2]
            try:
                rank = int(parts[3])
            except Exception:
                rank = len(data[qid]) + 1
            try:
                score = float(parts[4])
            except Exception:
                score = 0.0
            data[qid].append((rank, docid, score))
    for qid in data:
        data[qid].sort(key=lambda x: (x[0], -x[2], x[1]))
    return data


def top_entries(entries: Sequence[Tuple[int, str, float]], k: int) -> List[Tuple[int, str, float]]:
    return list(entries[:k]) if entries else []


def top_docids(entries: Sequence[Tuple[int, str, float]], k: int) -> List[str]:
    return [docid for _, docid, _ in top_entries(entries, k)]


def ndcg_at_k(docids: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    gains: List[float] = []
    for rank, docid in enumerate(docids[:k], start=1):
        rel = qrels.get(docid, 0)
        gains.append(((2 ** rel) - 1) / math.log2(rank + 1))
    dcg = sum(gains)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    if not ideal:
        return 0.0
    idcg = sum(((2 ** rel) - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    if idcg <= 0:
        return 0.0
    return METRIC_PERCENT_SCALE * (dcg / idcg)


def recall_at_k(docids: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    rel_docs = {docid for docid, rel in qrels.items() if rel > 0}
    if not rel_docs:
        return 0.0
    hits = sum(1 for docid in docids[:k] if docid in rel_docs)
    return METRIC_PERCENT_SCALE * (hits / len(rel_docs))


def first_rel_rank(entries: Sequence[Tuple[int, str, float]], qrels: Mapping[str, int], depth: int) -> float:
    for rank, docid, _ in entries[:depth]:
        if qrels.get(docid, 0) > 0:
            return float(rank)
    return float("inf")


def shift_with_inf(mix_rank: float, end_rank: float) -> float:
    if math.isinf(mix_rank) and math.isinf(end_rank):
        return 0.0
    mix_val = INF_RANK_SENTINEL if math.isinf(mix_rank) else mix_rank
    end_val = INF_RANK_SENTINEL if math.isinf(end_rank) else end_rank
    return mix_val - end_val


def load_queries_tsv(path: Path) -> Dict[str, str]:
    if not path.exists():
        warn(f"Query file not found: {path}")
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, text = parts
            out[qid] = text
    return out


def load_embeddings(cache_root: Path, lang: str) -> Dict[str, object]:
    if np is None:
        warn("numpy is unavailable; skipping embedding-based diagnostics")
        return {}
    npz_path = cache_root / lang / "queries.npz"
    if not npz_path.exists():
        warn(f"Embedding cache not found: {npz_path}")
        return {}
    data = np.load(npz_path, allow_pickle=False)
    if "qids" not in data or "vecs" not in data:
        warn(f"Invalid embedding cache format (missing qids/vecs): {npz_path}")
        return {}
    qids = data["qids"]
    vecs = data["vecs"]
    out: Dict[str, object] = {}
    for i, qid in enumerate(qids):
        out[str(qid)] = vecs[i]
    return out


def cosine(a: object, b: object) -> float:
    if np is None or a is None or b is None:
        return math.nan
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return math.nan
    return float(np.dot(a_arr, b_arr) / denom)


def geometry_features(a: object, b: object, lam: float) -> Tuple[float, float, float, float, float]:
    if np is None or a is None or b is None or not is_finite(lam):
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    qm = (1.0 - lam) * a_arr + lam * b_arr
    direction = b_arr - a_arr
    denom = float(np.dot(direction, direction))
    if denom == 0.0:
        r = math.nan
        delta_perp = math.nan
    else:
        r = float(np.dot(qm - a_arr, direction) / denom)
        proj = a_arr + r * direction
        delta_perp = float(np.linalg.norm(qm - proj))
    return (
        r,
        delta_perp,
        cosine(qm, a_arr),
        cosine(qm, b_arr),
        cosine(a_arr, b_arr),
    )


def normalize_doc_lang(value: str) -> str:
    if value is None:
        return ""
    token = value.strip().lower()
    return LANG_NAME_TO_CODE.get(token, token)


def read_doc_meta_subset(
    path: Path,
    case_id: str = "",
) -> Tuple[
    Dict[Tuple[str, str, int, str], Dict[str, str]],
    Dict[Tuple[str, str, str], Dict[str, str]],
]:
    if not path.exists():
        warn(f"Doc metadata file not found: {path}")
        return {}, {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(DOC_META_SNIFF_BYTES)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)

        if reader.fieldnames is None:
            warn(f"Doc metadata file has no header: {path}")
            return {}, {}

        lower_to_name = {name.strip().lower(): name for name in reader.fieldnames}
        case_id_col = lower_to_name.get("case_id")
        qid_col = lower_to_name.get("qid")
        condition_col = lower_to_name.get("condition")
        rank_col = lower_to_name.get("rank")
        docid_col = lower_to_name.get("docid")
        missing_required = []
        if not case_id_col:
            missing_required.append("case_id")
        if not qid_col:
            missing_required.append("qid")
        if not condition_col:
            missing_required.append("condition")
        if not rank_col:
            missing_required.append("rank")
        if not docid_col:
            missing_required.append("docid")
        if missing_required:
            warn(
                f"Doc metadata missing required columns ({', '.join(missing_required)}): {path}"
            )
            return {}, {}

        lang_col = lower_to_name.get("lang")
        title_col = lower_to_name.get("title")
        snippet_col = lower_to_name.get("snippet")

        by_hit: Dict[Tuple[str, str, int, str], Dict[str, str]] = {}
        by_doc: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        case_scope = (case_id or "").strip()
        for row in reader:
            row_case_id = (row.get(case_id_col) or "").strip()
            if case_scope and row_case_id != case_scope:
                continue
            qid = (row.get(qid_col) or "").strip()
            condition = (row.get(condition_col) or "").strip().lower()
            if not qid or not condition:
                continue
            docid = (row.get(docid_col) or "").strip()
            if not docid:
                continue
            try:
                rank = int((row.get(rank_col) or "").strip())
            except Exception:
                continue
            payload = {
                "lang": (row.get(lang_col) or "").strip() if lang_col else "",
                "title": (row.get(title_col) or "").strip() if title_col else "",
                "snippet": (row.get(snippet_col) or "").strip() if snippet_col else "",
            }
            by_hit[(qid, condition, rank, docid)] = payload
            by_doc.setdefault((qid, condition, docid), payload)
        return by_hit, by_doc


def read_doc_purity_features(
    path: Path,
    case_id: str = "",
) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.exists():
        warn(f"Doc purity file not found: {path}")
        return {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(DOC_META_SNIFF_BYTES)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        if reader.fieldnames is None:
            warn(f"Doc purity file has no header: {path}")
            return {}

        lower_to_name = {name.strip().lower(): name for name in reader.fieldnames}
        case_id_col = lower_to_name.get("case_id")
        docid_col = lower_to_name.get("docid")
        if not case_id_col or not docid_col:
            warn(f"Doc purity file missing required columns (case_id, docid): {path}")
            return {}

        out: Dict[Tuple[str, str], Dict[str, str]] = {}
        case_scope = (case_id or "").strip()
        for row in reader:
            row_case_id = (row.get(case_id_col) or "").strip()
            if case_scope and row_case_id != case_scope:
                continue
            docid = (row.get(docid_col) or "").strip()
            if not row_case_id or not docid:
                continue
            out[(row_case_id, docid)] = {str(k): str(v) for k, v in row.items()}
        return out


def snippet_ascii_ratio(text: str) -> float:
    if not text:
        return math.nan
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return math.nan
    ascii_letters = [ch for ch in letters if ch in string.ascii_letters]
    return len(ascii_letters) / len(letters)


def mismatch_rate(top_docids_list: Sequence[str], meta: Mapping[str, Mapping[str, str]], expected: Set[str]) -> float:
    if not expected:
        return math.nan
    langs: List[str] = []
    for docid in top_docids_list:
        m = meta.get(docid)
        if not m:
            continue
        lang = normalize_doc_lang(m.get("lang", ""))
        if not lang:
            continue
        langs.append(lang)
    if not langs:
        return math.nan
    mismatches = sum(1 for lang in langs if lang not in expected)
    return mismatches / len(langs)


def mean_ascii_ratio(top_docids_list: Sequence[str], meta: Mapping[str, Mapping[str, str]]) -> float:
    vals: List[float] = []
    for docid in top_docids_list:
        m = meta.get(docid)
        if not m:
            continue
        ratio = snippet_ascii_ratio(m.get("snippet", ""))
        if is_finite(ratio):
            vals.append(ratio)
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def assign_failure_label(
    row: Mapping[str, object],
    config: FailureLabelConfig = DEFAULT_FAILURE_LABEL_CONFIG,
) -> str:
    mix_mismatch = parse_float(row.get("doc_lang_mismatch_rate10_mix"))
    endpoint_cos = parse_float(row.get("endpoint_cos"))
    len_ratio = parse_float(row.get("len_ratio"))
    delta_ndcg = parse_float(row.get("delta_ndcg10"))
    delta_recall = parse_float(row.get("delta_recall10"))

    if is_finite(mix_mismatch) and mix_mismatch > config.mismatch_rate_gt:
        return "IndexLeakage"
    if (is_finite(endpoint_cos) and endpoint_cos < config.endpoint_cos_lt) or (
        is_finite(len_ratio) and (len_ratio < config.len_ratio_min or len_ratio > config.len_ratio_max)
    ):
        return "TranslationDivergence"
    if is_finite(delta_recall) and delta_recall < config.delta_recall_lt:
        return "RecallDrop"
    if (
        is_finite(delta_ndcg)
        and is_finite(delta_recall)
        and delta_ndcg < config.rankdrop_delta_ndcg_lt
        and delta_recall >= config.rankdrop_delta_recall_ge
    ):
        return "RankDrop"
    return FAILURE_LABEL_UNCLASSIFIED


def to_markdown_cell(text: str, limit: int = MARKDOWN_CELL_DEFAULT_LIMIT) -> str:
    t = (text or "").replace("\n", " ").replace("|", "\\|")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        return t[: limit - 3] + "..."
    return t


def rank_text(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return str(int(value)) if abs(value - round(value)) < ROUND_TO_INT_EPS else fmt(value, 3)
