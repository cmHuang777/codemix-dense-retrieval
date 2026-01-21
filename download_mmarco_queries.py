#!/usr/bin/env python
"""
Grab selected mMARCO query sets (choose languages and split) and write TSV files:   <id>\t<text>
Keeps everything in streaming mode, so it never stores the big passage
collection locally.

sample usage:
    python download_mmarco_queries.py --out_dir ./data/mmarco_dev --split dev 
        --languages english chinese french
"""
import argparse, pathlib, tempfile, shutil, sys
from datasets import load_dataset
import tqdm

def dump(lang: str, split: str, out_path: pathlib.Path) -> int:
    cfg = f"queries-{lang}"
    ds  = load_dataset("unicamp-dl/mmarco", cfg, split=split, streaming=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # write to a temp file first, then atomically move into place
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=out_path.parent) as tmp:
        tmp_name = tmp.name
        try:
            for rec in tqdm.tqdm(ds, desc=f"{lang}-{split}"):
                tmp.write(f"{rec['id']}\t{rec['text']}\n")
                n += 1
        except KeyboardInterrupt:
            print(f"\n[!] Interrupted while writing {lang}-{split}. Partial file kept at: {tmp_name}")
            return n

    shutil.move(tmp_name, out_path)
    print(f"[✓] Wrote {n} queries to {out_path}")
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data/mmarco_dev")
    ap.add_argument(
        "--split",
        choices=["train", "dev.full", "dev"],
        default="dev",
        help="Pick a split small enough for code-mix runs",
    )
    ap.add_argument(
        "--languages",
        "--langs",
        nargs="+",
        default=["english", "chinese"],
        help="Languages to download queries for (matching mMARCO config names)",
    )
    args = ap.parse_args()

    suffix_map = {
        "english": "en",
        "chinese": "zh",
        "french": "fr",
        "german": "de",
        "indonesian": "id",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "spanish": "es",
        "arabic": "ar",
        "dutch": "nl",
        "hindi": "hi",
        "japanese": "ja",
        "vietnamese": "vi",
    }
    out_dir = pathlib.Path(args.out_dir)
    total = 0
    for lang in args.languages:
        lang_cfg = lang.lower()
        suffix = suffix_map.get(lang_cfg, lang_cfg[:2])  # default to first two letters if unmapped
        total += dump(lang_cfg, args.split, out_dir / f"queries.{suffix}.tsv")
    print(f"[✓] Total written across languages: {total}")
