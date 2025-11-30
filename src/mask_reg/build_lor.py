# src/mask_reg/build_lor.py
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .hashing import stable_hash_token


def compute_lor_tables(
    dataset_name: str,
    languages: list[str],
    tokenizer_name: str,
    out_dir: Path,
    alpha: float,
    mask_fraction: float,
    max_examples: int | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_name)  # returns dict: {en: Dataset, ru: Dataset, ...}
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    for lang in languages:
        if lang not in ds:
            print(f"[WARN] Language split '{lang}' not found in dataset keys: {list(ds.keys())}")
            continue

        toxic_counts = Counter()
        neutral_counts = Counter()
        total_toxic = 0
        total_neutral = 0

        split = ds[lang]
        n = len(split) if max_examples is None else min(len(split), max_examples)

        for i in range(n):
            ex = split[i]
            toxic_text = ex.get("toxic_sentence", "") or ""
            neutral_text = ex.get("neutral_sentence", "") or ""

            # tokenize into mt5 subwords (works for zh/ar/hi/etc without space rules)
            t_ids = tok(toxic_text, add_special_tokens=False).get("input_ids", [])
            n_ids = tok(neutral_text, add_special_tokens=False).get("input_ids", [])

            t_toks = tok.convert_ids_to_tokens(t_ids)
            n_toks = tok.convert_ids_to_tokens(n_ids)

            for tt in t_toks:
                toxic_counts[stable_hash_token(tt)] += 1
            for nt in n_toks:
                neutral_counts[stable_hash_token(nt)] += 1

            total_toxic += len(t_toks)
            total_neutral += len(n_toks)

        vocab_hashes = set(toxic_counts.keys()) | set(neutral_counts.keys())
        V = len(vocab_hashes)

        # LOR per hashed token
        lor = {}
        for h in vocab_hashes:
            ct = toxic_counts[h]
            cn = neutral_counts[h]
            # smoothed probabilities
            p_t = (ct + alpha) / (total_toxic + alpha * V)
            p_n = (cn + alpha) / (total_neutral + alpha * V)
            lor[h] = math.log(p_t) - math.log(p_n)

        # choose threshold as top "mask_fraction" of LOR values
        lor_values = sorted(lor.values())
        if len(lor_values) == 0:
            thr = float("inf")
        else:
            # e.g., mask_fraction=0.02 -> threshold at 98th percentile
            q_index = int((1.0 - mask_fraction) * (len(lor_values) - 1))
            q_index = max(0, min(q_index, len(lor_values) - 1))
            thr = lor_values[q_index]

        # write jsonl table
        table_path = out_dir / f"{lang}.jsonl"
        with table_path.open("w", encoding="utf-8") as f:
            for h, v in lor.items():
                f.write(
                    json.dumps(
                        {
                            "h": h,
                            "lor": float(v),
                            "ct": int(toxic_counts[h]),
                            "cn": int(neutral_counts[h]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        meta_path = out_dir / f"{lang}.meta.json"
        meta = {
            "dataset": dataset_name,
            "language": lang,
            "tokenizer": tokenizer_name,
            "alpha": alpha,
            "mask_fraction": mask_fraction,
            "lor_threshold": float(thr),
            "max_examples": max_examples,
            "total_toxic_tokens": int(total_toxic),
            "total_neutral_tokens": int(total_neutral),
            "V": int(V),
            "table_file": str(table_path.as_posix()),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {lang}: wrote {table_path} and {meta_path} (threshold={thr:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="textdetox/multilingual_paradetox")
    ap.add_argument("--languages", default="en,ru,uk,es,am,zh,ar,hi,de")
    ap.add_argument("--tokenizer", default="google/mt5-small")
    ap.add_argument("--out_dir", default="artifacts/lor")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--mask_fraction", type=float, default=0.02, help="Top fraction of tokens (by LOR) to treat as toxic-ish")
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    compute_lor_tables(
        dataset_name=args.dataset,
        languages=languages,
        tokenizer_name=args.tokenizer,
        out_dir=Path(args.out_dir),
        alpha=args.alpha,
        mask_fraction=args.mask_fraction,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
