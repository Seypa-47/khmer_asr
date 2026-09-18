#!/usr/bin/env python
"""
Approach 2: Meta MMS Khmer CTC Fine-tuning and Evaluation.

Architecture: Connectionist Temporal Classification (CTC) Acoustic Model
Model: facebook/mms-1b-all (or facebook/mms-300m) with Khmer (khm) vocabulary adapter.
Dataset: Exact same train/val/test split as Approach 1 (FLEURS km_kh / DDD Cambodia).
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
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
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=25)
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


def prepare_dataset(batch: dict[str, Any], processor: AutoProcessor) -> dict[str, Any]:
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    cleaned_sentence = normalize_khmer_text(batch["transcription"])
    batch["labels"] = processor(text=cleaned_sentence).input_ids
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

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics


def main() -> None:
    args = parse_args()
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    print("Loading Google FLEURS km_kh dataset...")
    fleurs = load_dataset(FLEURS_DATASET, FLEURS_CONFIG)
    raw_train = fleurs["train"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    raw_val = fleurs["validation"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    raw_test = fleurs["test"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    if args.max_train_samples:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
    if args.max_eval_samples:
        raw_val = raw_val.select(range(min(args.max_eval_samples, len(raw_val))))
    if args.max_test_samples:
        raw_test = raw_test.select(range(min(args.max_test_samples, len(raw_test))))

    train_data = raw_train.map(lambda b: prepare_dataset(b, processor), remove_columns=raw_train.column_names)
    eval_data = raw_val.map(lambda b: prepare_dataset(b, processor), remove_columns=raw_val.column_names)
    test_data = raw_test.map(lambda b: prepare_dataset(b, processor), remove_columns=raw_test.column_names)

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    compute_metrics = build_compute_metrics(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
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

    if not args.eval_only:
        print("Starting training for Approach 2 (Meta MMS Khmer CTC)...")
        trainer.train()
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    print("Evaluating Approach 2 on FLEURS km_kh held-out test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
    print(f"Approach 2 Test Results: {test_metrics}")


if __name__ == "__main__":
    main()
