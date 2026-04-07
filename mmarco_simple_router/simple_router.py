from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

ALPHAS = [0, 10, 30, 50, 70, 90, 100]
MODEL_TYPE = "pair_doc_mix_lookup_router"
META_COLUMNS = ("pair", "doc_mix")


def alpha_column(alpha: int) -> str:
    return f"ndcg_{int(alpha)}"


def qid_fold(qid: int) -> int:
    return int(qid) % 5


def alpha_columns() -> list[str]:
    return [alpha_column(alpha) for alpha in ALPHAS]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    payload = joblib.load(dataset_path)
    dataset = payload["dataset"] if isinstance(payload, dict) and "dataset" in payload else payload
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(f"Unsupported dataset payload from {dataset_path}")
    return dataset.copy()


def load_qids_common(qids_common_path: Path) -> set[int]:
    qids = pd.read_csv(qids_common_path, sep="\t", header=None, names=["qid"])
    return set(qids["qid"].astype(int).tolist())


def assign_splits_from_qids(dataset: pd.DataFrame, qids_common: set[int]) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["split"] = "train"
    dataset.loc[dataset["qid"].isin(qids_common), "split"] = "test"
    train_pool = dataset["split"].eq("train")
    dataset.loc[train_pool & dataset["qid"].map(qid_fold).eq(0), "split"] = "val"
    return dataset


def fit_router(dataset: pd.DataFrame, fit_splits: Iterable[str]) -> dict:
    fit_splits = tuple(fit_splits)
    fit_frame = dataset[dataset["split"].isin(fit_splits)].copy()
    if fit_frame.empty:
        raise ValueError(f"No rows found for fit_splits={fit_splits}")

    cols = alpha_columns()
    context_means = (
        fit_frame.groupby(list(META_COLUMNS), sort=True)[cols]
        .mean()
        .reset_index()
    )
    context_counts = (
        fit_frame.groupby(list(META_COLUMNS), sort=True)
        .size()
        .reset_index(name="fit_rows")
    )
    context_table = context_means.merge(context_counts, on=list(META_COLUMNS), how="left")
    global_means = fit_frame[cols].mean()
    fallback_alpha = int(global_means.idxmax().replace("ndcg_", ""))

    smoothed_scores = context_table[cols].copy()
    context_table["predicted_alpha"] = (
        smoothed_scores.idxmax(axis=1).str.replace("ndcg_", "", regex=False).astype(np.int16)
    )

    return {
        "model_type": MODEL_TYPE,
        "fit_splits": list(fit_splits),
        "fallback_alpha": fallback_alpha,
        "context_table": context_table,
        "global_mean_by_alpha": {
            int(col.replace("ndcg_", "")): float(global_means[col]) for col in cols
        },
        "pair_prior_weight": 0.0,
        "doc_prior_weight": 0.0,
        "global_prior_weight": 0.0,
        "fit_row_count": int(len(fit_frame)),
        "context_count": int(len(context_table)),
    }


def fit_smoothed_router(
    dataset: pd.DataFrame,
    fit_splits: Iterable[str],
    pair_prior_weight: float = 0.0,
    doc_prior_weight: float = 0.0,
    global_prior_weight: float = 0.0,
) -> dict:
    fit_splits = tuple(fit_splits)
    fit_frame = dataset[dataset["split"].isin(fit_splits)].copy()
    if fit_frame.empty:
        raise ValueError(f"No rows found for fit_splits={fit_splits}")

    cols = alpha_columns()
    context_means = (
        fit_frame.groupby(list(META_COLUMNS), sort=True)[cols]
        .mean()
        .reset_index()
    )
    context_counts = (
        fit_frame.groupby(list(META_COLUMNS), sort=True)
        .size()
        .reset_index(name="fit_rows")
    )
    context_table = context_means.merge(context_counts, on=list(META_COLUMNS), how="left")
    pair_means = fit_frame.groupby("pair", sort=True)[cols].mean().reset_index()
    doc_means = fit_frame.groupby("doc_mix", sort=True)[cols].mean().reset_index()
    global_means = fit_frame[cols].mean()
    fallback_alpha = int(global_means.idxmax().replace("ndcg_", ""))

    merged = context_table.merge(pair_means, on="pair", how="left", suffixes=("_ctx", "_pair"))
    merged = merged.merge(doc_means, on="doc_mix", how="left", suffixes=("", "_doc"))
    denom = (
        merged["fit_rows"].astype(np.float32)
        + np.float32(pair_prior_weight)
        + np.float32(doc_prior_weight)
        + np.float32(global_prior_weight)
    )

    smoothed_scores = pd.DataFrame(index=merged.index)
    for alpha in ALPHAS:
        col = alpha_column(alpha)
        numerator = merged["fit_rows"].to_numpy(dtype=np.float32) * merged[f"{col}_ctx"].to_numpy(dtype=np.float32)
        if pair_prior_weight:
            numerator += np.float32(pair_prior_weight) * merged[f"{col}_pair"].to_numpy(dtype=np.float32)
        if doc_prior_weight:
            numerator += np.float32(doc_prior_weight) * merged[f"{col}_doc"].to_numpy(dtype=np.float32)
        if global_prior_weight:
            numerator += np.float32(global_prior_weight) * np.float32(global_means[col])
        smoothed_scores[col] = numerator / denom

    context_table["predicted_alpha"] = (
        smoothed_scores[cols].idxmax(axis=1).str.replace("ndcg_", "", regex=False).astype(np.int16)
    )

    return {
        "model_type": MODEL_TYPE,
        "fit_splits": list(fit_splits),
        "fallback_alpha": fallback_alpha,
        "context_table": context_table,
        "global_mean_by_alpha": {
            int(col.replace("ndcg_", "")): float(global_means[col]) for col in cols
        },
        "pair_prior_weight": float(pair_prior_weight),
        "doc_prior_weight": float(doc_prior_weight),
        "global_prior_weight": float(global_prior_weight),
        "fit_row_count": int(len(fit_frame)),
        "context_count": int(len(context_table)),
    }


def predict_alpha(frame: pd.DataFrame, model: dict) -> tuple[np.ndarray, np.ndarray]:
    lookup = model["context_table"][list(META_COLUMNS) + ["predicted_alpha"]]
    merged = frame.merge(lookup, on=list(META_COLUMNS), how="left", sort=False)
    seen_mask = merged["predicted_alpha"].notna().to_numpy(dtype=bool, copy=False)
    predicted_alpha = (
        merged["predicted_alpha"].fillna(model["fallback_alpha"]).astype(np.int16).to_numpy()
    )
    return predicted_alpha, seen_mask


def mean_actual_ndcg(frame: pd.DataFrame, predicted_alpha: np.ndarray) -> float:
    values = np.zeros(len(frame), dtype=np.float32)
    for alpha in ALPHAS:
        mask = predicted_alpha == alpha
        if np.any(mask):
            values[mask] = frame.loc[mask, alpha_column(alpha)].to_numpy(dtype=np.float32, copy=False)
    return float(values.mean())


def best_fixed_alpha_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    means = {alpha: float(frame[alpha_column(alpha)].mean()) for alpha in ALPHAS}
    best_alpha = int(max(ALPHAS, key=lambda alpha: (means[alpha], -alpha)))
    return {
        "best_fixed_alpha": best_alpha,
        "best_fixed_alpha_mean_ndcg10": means[best_alpha],
        "mean_ndcg10_by_alpha": means,
    }


def basic_split_metrics(frame: pd.DataFrame, predicted_alpha: np.ndarray, seen_mask: np.ndarray) -> dict:
    pred_ndcg = mean_actual_ndcg(frame, predicted_alpha)
    fixed = best_fixed_alpha_metrics(frame)
    return {
        "rows": int(len(frame)),
        "contexts": int(frame.groupby(list(META_COLUMNS)).ngroups),
        "mean_ndcg10": pred_ndcg,
        "always_0_mean_ndcg10": float(frame[alpha_column(0)].mean()),
        "always_100_mean_ndcg10": float(frame[alpha_column(100)].mean()),
        "best_endpoint_mean_ndcg10": float(frame["best_endpoint_ndcg"].mean()),
        "oracle_full_mean_ndcg10": float(frame["oracle_ndcg"].mean()),
        "exact_best_alpha_acc": float(np.mean(predicted_alpha == frame["best_alpha"].to_numpy())),
        "seen_context_row_pct": float(seen_mask.mean() * 100.0),
        **fixed,
    }


def context_comparison(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    cols = alpha_columns()
    context_means = (
        frame.groupby(list(META_COLUMNS), sort=True)[cols]
        .mean()
        .reset_index()
    )
    context_counts = (
        frame.groupby(list(META_COLUMNS), sort=True)
        .size()
        .reset_index(name="eval_rows")
    )
    comparison = context_means.merge(context_counts, on=list(META_COLUMNS), how="left")
    comparison = comparison.merge(
        model["context_table"][list(META_COLUMNS) + ["predicted_alpha"]],
        on=list(META_COLUMNS),
        how="left",
        sort=False,
    )
    comparison["seen_in_model"] = comparison["predicted_alpha"].notna().astype(np.int8)
    comparison["predicted_alpha"] = comparison["predicted_alpha"].fillna(model["fallback_alpha"]).astype(np.int16)

    optimal_set_strings = []
    optimal_argmax = []
    router_scores = []
    optimal_scores = []
    is_argmax = []
    is_optimal = []

    for row in comparison.itertuples(index=False):
        scores = {alpha: float(getattr(row, alpha_column(alpha))) for alpha in ALPHAS}
        max_score = max(scores.values())
        optimal_alphas = [
            alpha for alpha in ALPHAS if np.isclose(scores[alpha], max_score, atol=1e-9, rtol=0.0)
        ]
        predicted_alpha = int(row.predicted_alpha)
        optimal_set_strings.append("|".join(str(alpha) for alpha in optimal_alphas))
        optimal_argmax.append(optimal_alphas[0])
        router_scores.append(scores[predicted_alpha])
        optimal_scores.append(max_score)
        is_argmax.append(predicted_alpha == optimal_alphas[0])
        is_optimal.append(predicted_alpha in optimal_alphas)

    comparison["optimal_context_alpha_argmax"] = np.asarray(optimal_argmax, dtype=np.int16)
    comparison["optimal_context_alpha_set"] = optimal_set_strings
    comparison["router_context_mean_ndcg10"] = np.asarray(router_scores, dtype=np.float32)
    comparison["optimal_context_mean_ndcg10"] = np.asarray(optimal_scores, dtype=np.float32)
    comparison["router_is_context_argmax"] = np.asarray(is_argmax, dtype=np.int8)
    comparison["router_is_context_optimal_set"] = np.asarray(is_optimal, dtype=np.int8)
    comparison["context_oracle_gap"] = (
        comparison["optimal_context_mean_ndcg10"] - comparison["router_context_mean_ndcg10"]
    ).astype(np.float32)
    return comparison


def context_metrics(comparison: pd.DataFrame) -> dict[str, float | int]:
    weights = comparison["eval_rows"].to_numpy(dtype=np.float64, copy=False)
    return {
        "context_count": int(len(comparison)),
        "seen_context_count": int(comparison["seen_in_model"].sum()),
        "unseen_context_count": int((1 - comparison["seen_in_model"]).sum()),
        "context_argmax_match_pct": float(comparison["router_is_context_argmax"].mean() * 100.0),
        "context_optimal_set_match_pct": float(comparison["router_is_context_optimal_set"].mean() * 100.0),
        "row_weighted_context_optimal_set_match_pct": float(
            np.average(
                comparison["router_is_context_optimal_set"].to_numpy(dtype=np.float64, copy=False),
                weights=weights,
            )
            * 100.0
        ),
        "context_oracle_mean_ndcg10": float(
            np.average(
                comparison["optimal_context_mean_ndcg10"].to_numpy(dtype=np.float64, copy=False),
                weights=weights,
            )
        ),
        "router_from_context_table_mean_ndcg10": float(
            np.average(
                comparison["router_context_mean_ndcg10"].to_numpy(dtype=np.float64, copy=False),
                weights=weights,
            )
        ),
        "mean_context_oracle_gap": float(
            np.average(
                comparison["context_oracle_gap"].to_numpy(dtype=np.float64, copy=False),
                weights=weights,
            )
        ),
    }


def save_router_model(model: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = model["context_table"].copy()
    payload = {
        "model_type": model["model_type"],
        "fit_splits": model["fit_splits"],
        "fallback_alpha": int(model["fallback_alpha"]),
        "fit_row_count": int(model["fit_row_count"]),
        "context_count": int(model["context_count"]),
        "feature_names": ["pair", "doc_mix"],
        "global_mean_by_alpha": model["global_mean_by_alpha"],
        "pair_prior_weight": float(model.get("pair_prior_weight", 0.0)),
        "doc_prior_weight": float(model.get("doc_prior_weight", 0.0)),
        "global_prior_weight": float(model.get("global_prior_weight", 0.0)),
        "contexts": table.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def load_router_model(model_path: Path) -> dict:
    payload = json.loads(model_path.read_text())
    context_table = pd.DataFrame(payload["contexts"])
    return {
        "model_type": payload["model_type"],
        "fit_splits": payload["fit_splits"],
        "fallback_alpha": int(payload["fallback_alpha"]),
        "fit_row_count": int(payload["fit_row_count"]),
        "context_count": int(payload["context_count"]),
        "global_mean_by_alpha": {
            int(alpha): float(score) for alpha, score in payload["global_mean_by_alpha"].items()
        },
        "pair_prior_weight": float(payload.get("pair_prior_weight", 0.0)),
        "doc_prior_weight": float(payload.get("doc_prior_weight", 0.0)),
        "global_prior_weight": float(payload.get("global_prior_weight", 0.0)),
        "context_table": context_table,
    }
