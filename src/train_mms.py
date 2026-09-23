#!/usr/bin/env python
"""
Approach 2: Meta MMS Khmer CTC Fine-tuning and Evaluation.

Architecture: Connectionist Temporal Classification (CTC) Acoustic Model
Model: facebook/mms-1b-all (or facebook/mms-300m) with Khmer (khm) vocabulary adapter.
Dataset: Exact same train/val/test split as Approach 1 (FLEURS km_kh / DDD Cambodia).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    set_seed,
)

try:
    from .matched_fleurs import load_manifest, select_manifest_split
except ImportError:  # Direct execution: python src/train_mms.py
    from matched_fleurs import load_manifest, select_manifest_split

try:
    import evaluate
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "evaluate", "jiwer"])
    import evaluate

MODEL_ID = "facebook/mms-1b-all"
TARGET_LANG = "khm"  # ISO 639-3 code for Khmer
SAMPLING_RATE = 16_000
FLEURS_DATASET = "google/fleurs"
FLEURS_CONFIG = "km_kh"


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    print(f"Random seed fixed to {seed} across Python, NumPy, PyTorch, and Transformers.")


def normalize_khmer_text(value: Any) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFC", str(value))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/Evaluate Meta MMS Khmer CTC Model (Approach 2)")
    parser.add_argument("--output-dir", default="./mms-khmer-ctc")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--target-lang", default=TARGET_LANG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--max-test-samples", type=int, default=200)
    parser.add_argument("--split-manifest", default=None, help="Shared FLEURS row manifest for a controlled comparison.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Write test and run metrics to this JSON file.")
    parser.add_argument("--trainer-state-output", type=Path, default=None, help="Copy saved training history to this JSON file.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Resume a saved Trainer checkpoint after interruption.")
    parser.add_argument("--skip-test", action="store_true", help="Use validation only during hyperparameter tuning.")
    parser.add_argument("--tuning-only", action="store_true", help="Save validation history without large pilot checkpoints.")
    parser.add_argument("--num-train-epochs", type=float, default=15.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay regularization.")
    parser.add_argument("--unfreeze-top-layers", type=int, default=4, help="Number of top transformer encoder layers to unfreeze (0 = freeze all)")
    parser.add_argument("--apply-spec-augment", action=argparse.BooleanOptionalAction, default=True, help="Apply SpecAugment data masking during training")
    parser.add_argument("--lr-scheduler-type", default="cosine", help="Learning rate scheduler type (e.g. cosine, linear)")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps for learning rate schedule")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run zero-shot / pretrained evaluation on test set")
    return parser.parse_args()


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_dataset(batch: dict[str, Any], processor: AutoProcessor, vocab_size: int = 152) -> dict[str, Any]:
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    cleaned_sentence = normalize_khmer_text(batch["transcription"])
    # In Khmer scriptio continua, spaces are not word delimiters.
    # Meta MMS khm adapter vocabulary only has 152 classes (0-151) and does not include the space token '|' (id 152).
    # Stripping spaces prevents out-of-vocab index 152 errors.
    cleaned_sentence_no_space = cleaned_sentence.replace(" ", "")
    token_ids = processor(text=cleaned_sentence_no_space).input_ids
    unk_id = getattr(processor.tokenizer, "unk_token_id", 3)
    batch["labels"] = [tok if tok < vocab_size else unk_id for tok in token_ids]
    return batch


def build_compute_metrics(processor: AutoProcessor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        pred_str = [normalize_khmer_text(t) for t in pred_str]
        label_str = [normalize_khmer_text(t) for t in label_str]

        # Khmer spaces are not reliable word boundaries. Keep conventional
        # WER for reference, but compute the comparison CER without whitespace.
        cer_pred = [re.sub(r"\s+", "", text) for text in pred_str]
        cer_ref = [re.sub(r"\s+", "", text) for text in label_str]
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=cer_pred, references=cer_ref)

        return {"wer": wer, "cer": cer}

    return compute_metrics


def main() -> None:
    args = parse_args()
    if args.tuning_only and not args.skip_test:
        raise ValueError("--tuning-only requires --skip-test so held-out test rows remain untouched")
    set_all_seeds(args.seed)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware compute environment: {device_name}")

    print(f"Loading processor for {args.model_id} (target_lang: {args.target_lang})...")
    processor = AutoProcessor.from_pretrained(args.model_id, target_lang=args.target_lang)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_id,
        target_lang=args.target_lang,
        ignore_mismatched_sizes=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    processor.tokenizer.set_target_lang(args.target_lang)
    model.load_adapter(args.target_lang)

    # SpecAugment and CTC numerical stability settings
    model.config.apply_spec_augment = args.apply_spec_augment
    if args.apply_spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.05
        model.config.mask_feature_length = 10
        print("SpecAugment enabled (time mask prob: 0.05, feature mask prob: 0.05).")
    model.config.ctc_zero_infinity = True

    # Layer freezing / unfreezing:
    # 1. Always freeze the raw waveform feature encoder (CNN)
    model.freeze_feature_encoder()

    if args.unfreeze_top_layers > 0:
        # Freeze entire wav2vec2 base parameters first
        for param in model.wav2vec2.parameters():
            param.requires_grad = False

        # Unfreeze top N transformer encoder layers
        total_layers = len(model.wav2vec2.encoder.layers)
        unfreeze_n = min(args.unfreeze_top_layers, total_layers)
        for layer in model.wav2vec2.encoder.layers[-unfreeze_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze adapter module if present
        if hasattr(model.wav2vec2, "adapter") and model.wav2vec2.adapter is not None:
            for param in model.wav2vec2.adapter.parameters():
                param.requires_grad = True

        # Unfreeze LM head
        for param in model.lm_head.parameters():
            param.requires_grad = True

        print(f"Deep Fine-Tuning: Unfroze top {unfreeze_n}/{total_layers} Transformer layers + LM head/adapter.")
    else:
        model.freeze_base_model()
        print("Base model frozen. Only LM head / adapter is trainable.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    print("Loading Google FLEURS km_kh dataset...")
    fleurs = load_dataset(FLEURS_DATASET, FLEURS_CONFIG)
    raw_train = fleurs["train"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    raw_val = fleurs["validation"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    raw_test = fleurs["test"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.split_manifest:
        manifest = load_manifest(args.split_manifest)
        if manifest["seed"] != args.seed:
            raise ValueError("Training seed differs from the shared split manifest seed")
        raw_train = select_manifest_split(raw_train, manifest, "train")
        raw_val = select_manifest_split(raw_val, manifest, "validation")
        raw_test = select_manifest_split(raw_test, manifest, "test")
        print(f"Matched FLEURS rows (train/validation/test): {(len(raw_train), len(raw_val), len(raw_test))}")
    else:
        # Historical selection retained for reproducing the earlier, diagnostic run.
        raw_train = raw_train.shuffle(seed=args.seed)
        if args.max_train_samples:
            raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
        if args.max_eval_samples:
            raw_val = raw_val.select(range(min(args.max_eval_samples, len(raw_val))))
        if args.max_test_samples:
            raw_test = raw_test.select(range(min(args.max_test_samples, len(raw_test))))
        raw_train = raw_train.filter(lambda b: len(b["audio"]["array"]) <= 160_000)
        raw_val = raw_val.filter(lambda b: len(b["audio"]["array"]) <= 160_000)

    vocab_size = model.config.vocab_size
    train_data = raw_train.map(lambda b: prepare_dataset(b, processor, vocab_size=vocab_size), remove_columns=raw_train.column_names)
    eval_data = raw_val.map(lambda b: prepare_dataset(b, processor, vocab_size=vocab_size), remove_columns=raw_val.column_names)
    test_data = None if args.skip_test else raw_test.map(
        lambda b: prepare_dataset(b, processor, vocab_size=vocab_size),
        remove_columns=raw_test.column_names,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    compute_metrics = build_compute_metrics(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        gradient_checkpointing=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no" if args.tuning_only else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        load_best_model_at_end=not args.tuning_only,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    train_metrics = None
    if not args.eval_only:
        print("Starting training for Approach 2 (Meta MMS Khmer CTC)...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train_metrics = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint).metrics
        if not args.tuning_only:
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)
        trainer.save_state()
        if args.trainer_state_output:
            args.trainer_state_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(args.output_dir) / "trainer_state.json", args.trainer_state_output)

    test_metrics = {}
    if not args.skip_test:
        print("Evaluating Approach 2 on FLEURS km_kh held-out test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
        print(f"Approach 2 Test Results: {test_metrics}")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), "results")
    if not os.path.exists(results_dir):
        results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = args.metrics_output or Path(results_dir) / "mms_test_metrics.json"
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(test_metrics)
    report.update({
        "train_metrics": train_metrics,
        "train_examples": len(train_data),
        "validation_examples": len(eval_data),
        "test_examples": len(raw_test),
        "hardware": device_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "spec_augment": args.apply_spec_augment,
        "split_manifest": args.split_manifest,
    })
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
