from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from simple_router import (
    ALPHAS,
    alpha_column,
    assign_splits_from_qids,
    basic_split_metrics,
    context_comparison,
    context_metrics,
    fit_smoothed_router,
    load_dataset,
    load_qids_common,
    predict_alpha,
    save_router_model,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Train a validation-tuned smoothed router.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=script_dir / "artifacts" / "dataset.joblib",
        help="Path to dataset.joblib. Required for training.",
    )
    parser.add_argument(
        "--qids-common",
        type=Path,
        required=True,
        help="Path to qids-common.tsv file (test split definition).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "artifacts" / "smoothed_router",
        help="Output directory for trained model.",
    )
    parser.add_argument(
        "--fit-splits",
        nargs="+",
        default=["train"],
    )
    parser.add_argument(
        "--pair-prior-grid",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    )
    parser.add_argument(
        "--doc-prior-grid",
        nargs="+",
        type=float,
        default=[0.0],
    )
    parser.add_argument(
        "--global-prior-grid",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_mean_ndcg10", "val_context_argmax_match_pct"],
        default="val_context_argmax_match_pct",
    )
    return parser.parse_args()


def build_split_context_targets(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    targets = {}
    cols = [alpha_column(alpha) for alpha in ALPHAS]
    for split in ("train", "val", "test"):
        frame = dataset[dataset["split"] == split].copy()
        means = frame.groupby(["pair", "doc_mix"], sort=True)[cols].mean().reset_index()
        counts = frame.groupby(["pair", "doc_mix"], sort=True).size().reset_index(name="rows")
        table = means.merge(counts, on=["pair", "doc_mix"], how="left")
        table["opt_alpha"] = (
            table[cols].idxmax(axis=1).str.replace("ndcg_", "", regex=False).astype(int)
        )
        targets[split] = table
    return targets


def score_from_targets(targets: pd.DataFrame, model: dict) -> tuple[float, float]:
    merged = targets.merge(
        model["context_table"][["pair", "doc_mix", "predicted_alpha"]],
        on=["pair", "doc_mix"],
        how="left",
    )
    if merged["predicted_alpha"].isna().any():
        merged["predicted_alpha"] = merged["predicted_alpha"].fillna(model["fallback_alpha"])
    predicted_alpha = merged["predicted_alpha"].to_numpy(dtype=np.int16, copy=False)
    weights = merged["rows"].to_numpy(dtype=np.float64, copy=False)
    values = np.zeros(len(merged), dtype=np.float64)
    for alpha in ALPHAS:
        mask = predicted_alpha == alpha
        if np.any(mask):
            values[mask] = merged.loc[mask, alpha_column(alpha)].to_numpy(dtype=np.float64, copy=False)
    mean_ndcg10 = float(np.average(values, weights=weights))
    context_argmax_match_pct = float(
        np.mean(predicted_alpha == merged["opt_alpha"].to_numpy(dtype=np.int16, copy=False)) * 100.0
    )
    return mean_ndcg10, context_argmax_match_pct


def evaluate_model(dataset: pd.DataFrame, model: dict) -> dict:
    split_metrics = {}
    for split in ("train", "val", "test"):
        frame = dataset[dataset["split"] == split].copy()
        predicted_alpha, seen_mask = predict_alpha(frame, model)
        metrics = basic_split_metrics(frame, predicted_alpha, seen_mask)
        metrics.update(context_metrics(context_comparison(frame, model)))
        split_metrics[split] = metrics
    return split_metrics


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    dataset = assign_splits_from_qids(dataset, load_qids_common(args.qids_common))
    split_targets = build_split_context_targets(dataset)

    search_rows = []
    best_key = None
    best_model = None

    for pair_weight, doc_weight, global_weight in product(
        args.pair_prior_grid,
        args.doc_prior_grid,
        args.global_prior_grid,
    ):
        model = fit_smoothed_router(
            dataset,
            fit_splits=args.fit_splits,
            pair_prior_weight=pair_weight,
            doc_prior_weight=doc_weight,
            global_prior_weight=global_weight,
        )
        val_mean_ndcg10, val_context_argmax_match_pct = score_from_targets(split_targets["val"], model)
        test_mean_ndcg10, test_context_argmax_match_pct = score_from_targets(split_targets["test"], model)
        row = {
            "pair_prior_weight": pair_weight,
            "doc_prior_weight": doc_weight,
            "global_prior_weight": global_weight,
            "val_mean_ndcg10": val_mean_ndcg10,
            "val_context_argmax_match_pct": val_context_argmax_match_pct,
            "test_mean_ndcg10": test_mean_ndcg10,
            "test_context_argmax_match_pct": test_context_argmax_match_pct,
        }
        search_rows.append(row)
        if args.selection_metric == "val_context_argmax_match_pct":
            key = (
                val_context_argmax_match_pct,
                val_mean_ndcg10,
                -pair_weight,
                -doc_weight,
                -global_weight,
            )
        else:
            key = (
                val_mean_ndcg10,
                val_context_argmax_match_pct,
                -pair_weight,
                -doc_weight,
                -global_weight,
            )
        if best_key is None or key > best_key:
            best_key = key
            best_model = model

    if best_model is None:
        raise RuntimeError("Search failed to produce a model")

    best_metrics = evaluate_model(dataset, best_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_router_model(best_model, args.output_dir / "router_model.json")
    best_model["context_table"].to_csv(args.output_dir / "router_table.csv", index=False)
    pd.DataFrame(search_rows).sort_values(
        ["val_mean_ndcg10", "val_context_argmax_match_pct"],
        ascending=[False, False],
    ).to_csv(args.output_dir / "search_results.csv", index=False)

    summary = {
        "model_type": best_model["model_type"],
        "feature_names": ["pair", "doc_mix"],
        "fit_splits": best_model["fit_splits"],
        "fallback_alpha": best_model["fallback_alpha"],
        "pair_prior_weight": best_model["pair_prior_weight"],
        "doc_prior_weight": best_model["doc_prior_weight"],
        "global_prior_weight": best_model["global_prior_weight"],
        "selection_metric": args.selection_metric,
        "fit_row_count": best_model["fit_row_count"],
        "context_count": best_model["context_count"],
        "global_mean_by_alpha": best_model["global_mean_by_alpha"],
        "splits": best_metrics,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
