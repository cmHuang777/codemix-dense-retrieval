from __future__ import annotations

import argparse
import json
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
    load_dataset,
    load_qids_common,
    load_router_model,
    predict_alpha,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Evaluate the minimal pair+doc_mix router baseline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=script_dir / "artifacts" / "dataset.joblib",
        help="Path to dataset.joblib. Required for evaluation.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "artifacts" / "final_router" / "router_model.json",
        help="Path to saved router model.",
    )
    parser.add_argument(
        "--qids-common",
        type=Path,
        required=True,
        help="Path to qids-common.tsv file (test split definition).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "artifacts" / "eval_qids_common",
        help="Output directory for evaluation results.",
    )
    return parser.parse_args()


def predicted_ndcg(frame: pd.DataFrame, predicted_alpha: np.ndarray) -> np.ndarray:
    values = np.zeros(len(frame), dtype=np.float32)
    for alpha in ALPHAS:
        mask = predicted_alpha == alpha
        if np.any(mask):
            values[mask] = frame.loc[mask, alpha_column(alpha)].to_numpy(dtype=np.float32, copy=False)
    return values


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    dataset = assign_splits_from_qids(dataset, load_qids_common(args.qids_common))
    frame = dataset[dataset["split"] == args.split].copy()
    model = load_router_model(args.model)

    predicted_alpha, seen_mask = predict_alpha(frame, model)
    ndcg_pred = predicted_ndcg(frame, predicted_alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = frame[["qid", "pair", "doc_mix", "best_alpha", "best_endpoint_alpha"]].copy()
    predictions["predicted_alpha"] = predicted_alpha
    predictions["predicted_ndcg10"] = ndcg_pred
    predictions["oracle_ndcg10"] = frame["oracle_ndcg"].to_numpy(dtype=np.float32, copy=False)
    predictions["best_endpoint_ndcg10"] = frame["best_endpoint_ndcg"].to_numpy(dtype=np.float32, copy=False)
    predictions["seen_context"] = seen_mask.astype(np.int8)
    predictions.to_csv(args.output_dir / f"{args.split}_predictions.csv", index=False)

    comparison = context_comparison(frame, model)
    comparison.to_csv(args.output_dir / f"{args.split}_context_comparison.csv", index=False)

    metrics = {
        "split": args.split,
        "model_type": model["model_type"],
        "feature_names": ["pair", "doc_mix"],
        "fallback_alpha": model["fallback_alpha"],
        "fit_splits": model["fit_splits"],
        "pair_prior_weight": model.get("pair_prior_weight", 0.0),
        "doc_prior_weight": model.get("doc_prior_weight", 0.0),
        "global_prior_weight": model.get("global_prior_weight", 0.0),
        **basic_split_metrics(frame, predicted_alpha, seen_mask),
        **context_metrics(comparison),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
