# src/mask_reg/infer_mt5.py
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .masker import load_lor_artifact, mask_text_with_lor

_EXTRA = re.compile(r"<extra_id_\d+>")
_WS = re.compile(r"\s+")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clean_out(s: str) -> str:
    s = _EXTRA.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    # remove dangling punctuation-only outputs
    if len(s) <= 1 and s and all(c in ".!?," for c in s):
        return ""
    return s


def detoxify_sentence(model, tokenizer, text_for_model: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(
        text_for_model,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Pass 1: safe beam search
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=6,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.10,
            length_penalty=1.0,
            min_new_tokens=3,  # prevents “empty” generations (EOS immediately)
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    text = _clean_out(text)
    if text:
        return text

    # Pass 2 (fallback): sampling when beam collapses to EOS
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.92,
            temperature=0.9,
            num_beams=1,
            repetition_penalty=1.05,
            min_new_tokens=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return _clean_out(text) or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--lor_dir", default="artifacts/lor")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_name", default="google/mt5-small")

    # New: loosen/tighten threshold without rebuilding LOR
    ap.add_argument("--threshold_mult", type=float, default=1.0)

    # New: optional task prefix (ONLY use if you trained with it)
    ap.add_argument("--task_prefix", default="", help="Must match training if you used one, e.g. 'detoxify: '")

    ap.add_argument("--input_jsonl", default=None)
    ap.add_argument("--output_jsonl", default="outputs/mask_reg_outputs.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--always_generate", action="store_true")
    args = ap.parse_args()

    lor_art = load_lor_artifact(args.lang, args.lor_dir)
    thr = float(lor_art.threshold) * float(args.threshold_mult)

    # IMPORTANT CHANGE (matches your new masker.py):
    # use slow tokenizer so sentinel ids (<extra_id_i>) are always valid
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.eval()
    model.to(pick_device())

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_one(text: str):
        masked, spans = mask_text_with_lor(
            text,
            tokenizer,
            lor_art.lor_by_hash,
            thr,
        )

        if (not spans) and (not args.always_generate):
            return masked, text, spans

        # IMPORTANT: prefix must match training (if you used one)
        text_for_model = (args.task_prefix + " " + masked).strip() if args.task_prefix else masked

        detox = detoxify_sentence(
            model,
            tokenizer,
            text_for_model,
            max_new_tokens=args.max_new_tokens,
        )

        # Final fallback if STILL empty
        if not detox:
            detox = _clean_out(text) or text

        return masked, detox, spans

    if args.input_jsonl is None:
        while True:
            try:
                text = input("\nEnter toxic sentence (CTRL+C or CTRL+D to quit): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                return

            if not text:
                continue

            masked, detox, spans = run_one(text)
            print(f"Masked: {masked}\nDetox:  {detox}\n(threshold={thr:.4f}, mult={args.threshold_mult})")
        return

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            text = ex.get("text", "") or ""
            masked, detox, spans = run_one(text)
            out = {**ex, "masked": masked, "detox": detox, "masked_spans": spans, "threshold": thr}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {out_path} (threshold={thr:.4f}, mult={args.threshold_mult})")


if __name__ == "__main__":
    main()
