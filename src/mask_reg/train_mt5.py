# src/mask_reg/train_mt5.py
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .masker import load_lor_artifact, mask_text_with_lor


def main():
    ap = argparse.ArgumentParser()

    # Core
    ap.add_argument("--lang", default="en")
    ap.add_argument("--dataset", default="textdetox/multilingual_paradetox")
    ap.add_argument("--lor_dir", default="artifacts/lor")
    ap.add_argument("--model_name", default="google/mt5-small")
    ap.add_argument("--out_dir", default="checkpoints/mask-reg-mt5-small-en-lora")

    # NEW: task prefix (must match inference for best results)
    ap.add_argument("--task_prefix", default="detoxify:", help="Prefix prepended to masked input during training.")

    # Sequence lengths (defaults lowered for MPS safety)
    ap.add_argument("--max_source_len", type=int, default=96)
    ap.add_argument("--max_target_len", type=int, default=96)

    # Training
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)

    # Dataset truncation (debugging)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_eval", type=int, default=200)

    # Tokenizer (NOTE: for MT5 we prefer SLOW tokenizer to keep <extra_id_*> reliable)
    ap.add_argument("--use_fast_tokenizer", action="store_true",
                    help="Optional. For MT5 masking reliability, slow tokenizer is recommended; this flag is ignored by default.")

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_targets",
        default="q,v",
        help="Comma-separated target modules for LoRA. For (m)T5 attention common: q,k,v,o (try q,v first).",
    )
    ap.add_argument(
        "--merge_lora",
        action="store_true",
        help="After training, merge LoRA into base weights and save a normal HF model (easy inference).",
    )

    args = ap.parse_args()

    # 1) Load LOR artifact
    lor_art = load_lor_artifact(args.lang, args.lor_dir)

    # 2) Tokenizer
    # IMPORTANT CHANGE: force slow tokenizer for MT5 to preserve <extra_id_*> and avoid <unk> sentinels
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # 3) Base model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # MPS memory saver
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # 4) Wrap with LoRA (PEFT)
    target_modules = [m.strip() for m in args.lora_targets.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # 5) Load dataset split
    ds_all = load_dataset(args.dataset)
    ds = ds_all[args.lang]
    ds_split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds = ds_split["train"]
    eval_ds = ds_split["test"]

    if args.max_train is not None:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    if args.max_eval is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval, len(eval_ds))))

    def _add_prefix(s: str) -> str:
        pref = (args.task_prefix or "").strip()
        if not pref:
            return s
        return f"{pref} {s}".strip()

    # 6) Preprocess (mask -> tokenize input, neutral -> tokenize target)
    def preprocess(ex):
        toxic = ex.get("toxic_sentence", "") or ""
        neutral = ex.get("neutral_sentence", "") or ""

        masked, _spans = mask_text_with_lor(
            toxic,
            tokenizer=tokenizer,
            lor_by_hash=lor_art.lor_by_hash,
            threshold=lor_art.threshold,
            max_spans=20,
        )

        # IMPORTANT CHANGE (Step 3): train with the SAME prefix you use at inference time
        masked_for_model = _add_prefix(masked)

        model_inp = tokenizer(
            masked_for_model,
            max_length=args.max_source_len,
            truncation=True,
        )

        labels = tokenizer(
            text_target=neutral,
            max_length=args.max_target_len,
            truncation=True,
        )

        model_inp["labels"] = labels["input_ids"]
        return model_inp

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 7) Training args (MPS friendly)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        predict_with_generate=True,

        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=25,

        save_total_limit=2,
        report_to="none",

        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 8) Train
    trainer.train()

    # 9) Save
    tokenizer.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir))

    # Save pointer config including prefix so you remember what you trained with
    (out_dir / "mask_reg_config.txt").write_text(
        "lang={}\nlor_dir={}\nlor_threshold={}\ntask_prefix={}\n".format(
            args.lang, args.lor_dir, lor_art.threshold, (args.task_prefix or "").strip()
        ),
        encoding="utf-8",
    )

    # 10) Optional: merge LoRA into base weights and save a "normal" model for easy inference
    if args.merge_lora:
        merged_dir = out_dir.parent / (out_dir.name + "-MERGED")
        merged_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model, PeftModel):
            merged = model.merge_and_unload()
        else:
            merged = model

        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

        (merged_dir / "mask_reg_config.txt").write_text(
            "lang={}\nlor_dir={}\nlor_threshold={}\ntask_prefix={}\n".format(
                args.lang, args.lor_dir, lor_art.threshold, (args.task_prefix or "").strip()
            ),
            encoding="utf-8",
        )
        print(f"[OK] Saved MERGED model to {merged_dir}")

    print(f"[OK] Saved LoRA adapter checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
