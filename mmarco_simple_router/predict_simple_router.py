from __future__ import annotations

import argparse
from pathlib import Path

from simple_router import load_router_model


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Predict a mix alpha from the minimal pair+doc_mix router.")
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "artifacts" / "final_router" / "router_model.json",
    )
    parser.add_argument("--pair", required=True, help="Language pair key, for example AR-HI")
    parser.add_argument("--doc-mix", required=True, help="Document regime key, for example AR + HI docs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_router_model(args.model)
    lookup = model["context_table"].set_index(["pair", "doc_mix"])["predicted_alpha"]
    alpha = int(lookup.get((args.pair, args.doc_mix), model["fallback_alpha"]))
    print(alpha)


if __name__ == "__main__":
    main()
