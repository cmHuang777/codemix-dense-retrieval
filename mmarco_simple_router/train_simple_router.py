from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_router import (
    assign_splits_from_qids,
    basic_split_metrics,
    context_comparison,
    context_metrics,
    fit_router,
    load_dataset,
    load_qids_common,
    predict_alpha,
    save_router_model,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Train the minimal pair+doc_mix router baseline.")
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
        default=script_dir / "artifacts" / "final_router",
        help="Output directory for trained model.",
    )
    parser.add_argument(
        "--fit-splits",
        nargs="+",
        default=["train"],
        help="Dataset splits used to learn the router table. Default: train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    dataset = assign_splits_from_qids(dataset, load_qids_common(args.qids_common))

    model = fit_router(dataset, fit_splits=args.fit_splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_router_model(model, args.output_dir / "router_model.json")
    model["context_table"].to_csv(args.output_dir / "router_table.csv", index=False)

    split_metrics = {}
    for split in ("train", "val", "test"):
        frame = dataset[dataset["split"] == split].copy()
        predicted_alpha, seen_mask = predict_alpha(frame, model)
        metrics = basic_split_metrics(frame, predicted_alpha, seen_mask)
        metrics.update(context_metrics(context_comparison(frame, model)))
        split_metrics[split] = metrics

    summary = {
        "model_type": model["model_type"],
        "feature_names": ["pair", "doc_mix"],
        "fit_splits": model["fit_splits"],
        "fallback_alpha": model["fallback_alpha"],
        "fit_row_count": model["fit_row_count"],
        "context_count": model["context_count"],
        "global_mean_by_alpha": model["global_mean_by_alpha"],
        "splits": split_metrics,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
