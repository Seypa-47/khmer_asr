#!/usr/bin/env python
"""
Fine-tune Whisper Small for Khmer ASR.

Main train set:
  - DDD-Cambodia/khmer-speech-dataset, with a fallback to the current
    Digital-Divide-Data/khmer-speech-dataset Hugging Face namespace.

Optional train set:
  - OpenSLR SLR42, included only when it can be loaded and normalized.

Validation/test benchmark:
  - Google FLEURS km_kh.

The dataset loaders are defensive because ASR datasets often differ in their
audio and transcript column names, and some Hugging Face dataset repos change
their split names over time.
"""

from __future__ import annotations

import argparse
import io
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Platform-aware default cache directory (Google Colab vs Windows vs generic)
if os.path.exists("/content"):
    DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", "/content/hf_cache")
elif os.path.exists("D:\\"):
    DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", r"D:\hf_cache")
else:
    DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", os.path.abspath("./hf_cache"))

os.environ.setdefault("HF_HOME", DEFAULT_CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(DEFAULT_CACHE_DIR, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(DEFAULT_CACHE_DIR, "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(DEFAULT_CACHE_DIR, "torch"))
temp_dir = os.path.join(DEFAULT_CACHE_DIR, "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir
os.environ["TMPDIR"] = temp_dir
tempfile.tempdir = temp_dir

try:
    import evaluate
except ImportError:
    print("Package 'evaluate' not found. Auto-installing 'evaluate' and 'jiwer'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "evaluate", "jiwer"])
    import evaluate
import random
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    set_seed,
)

try:
    from .matched_fleurs import load_manifest, select_manifest_split
except ImportError:  # Direct execution: python src/finetune_whisper.py
    from matched_fleurs import load_manifest, select_manifest_split


MODEL_NAME = "openai/whisper-tiny"
LANGUAGE = "Khmer"
TASK = "transcribe"
SAMPLING_RATE = 16_000

DDD_DATASET_ALIASES = (
    "DDD-Cambodia/khmer-speech-dataset",
    "Digital-Divide-Data/khmer-speech-dataset",
)
FLEURS_DATASET = "google/fleurs"
FLEURS_CONFIG = "km_kh"
OPENSLR_DATASET = "openslr/openslr"
OPENSLR_CONFIG = "SLR42"

AUDIO_COLUMN_CANDIDATES = (
    "audio",
    "audio_file",
    "audio_filepath",
    "audio_path",
    "path",
    "file",
    "file_name",
    "filename",
    "wav",
    "wav_path",
    "speech",
)
TEXT_COLUMN_CANDIDATES = (
    "transcription",
    "raw_transcription",
    "sentence",
    "text",
    "transcript",
    "normalized_text",
    "utt",
    "utterance",
    "label",
)



def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    print(f"Random seed fixed to {seed} across Python, NumPy, PyTorch, and Transformers.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Khmer ASR.")
    parser.add_argument("--output-dir", default="./whisper-tiny-khmer")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory on a drive with lots of free space.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze Whisper encoder to train only decoder (linear probe / transfer strategy).")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to a checkpoint folder to continue training from.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for regularization.")
    parser.add_argument("--use-fleurs-train", action="store_true", help="Use Google FLEURS km_kh train split for fast, lightweight training.")
    parser.add_argument("--include-slr42", action="store_true", help="Try to add OpenSLR SLR42 to training (WARNING: very large 100+ hour dataset).")
    parser.add_argument("--skip-ddd", action="store_true", help="Train without DDD dataset.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--split-manifest", default=None, help="Shared FLEURS row manifest for a controlled comparison.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Write test and run metrics to this JSON file.")
    parser.add_argument("--trainer-state-output", type=Path, default=None, help="Copy saved training history to this JSON file.")
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    return parser.parse_args()


def load_first_available_dataset(
    names: tuple[str, ...],
    split: str | None = None,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict:
    last_error: Exception | None = None
    for name in names:
        try:
            print(f"Loading {name}" + (f" split={split}" if split else ""))
            return load_dataset(name, split=split, cache_dir=cache_dir) if split else load_dataset(name, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - keep fallback chain visible.
            last_error = exc
            print(f"Could not load {name}: {exc}")
    raise RuntimeError(f"Could not load any dataset from {names}") from last_error


def choose_split(data: Dataset | DatasetDict, preferred: tuple[str, ...]) -> Dataset:
    if isinstance(data, Dataset):
        return data
    for split_name in preferred:
        if split_name in data:
            return data[split_name]
    if len(data) == 1:
        return next(iter(data.values()))
    raise ValueError(f"None of the preferred splits {preferred} exist. Available splits: {list(data.keys())}")


def find_column(dataset: Dataset, candidates: tuple[str, ...], kind: str) -> str:
    lower_to_actual = {column.lower(): column for column in dataset.column_names}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]

    for column in dataset.column_names:
        lowered = column.lower()
        if kind == "audio" and any(token in lowered for token in ("audio", "wav", "speech", "path", "file")):
            return column
        if kind == "text" and any(token in lowered for token in ("trans", "text", "sentence", "utt", "label")):
            return column

    raise ValueError(
        f"Could not infer {kind} column from columns {dataset.column_names}. "
        f"Pass a dataset with one of: {candidates}"
    )


def normalize_khmer_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def standardize_asr_dataset(dataset: Dataset, dataset_name: str) -> Dataset:
    audio_col = find_column(dataset, AUDIO_COLUMN_CANDIDATES, "audio")
    text_col = find_column(dataset, TEXT_COLUMN_CANDIDATES, "text")
    keep_cols = {audio_col, text_col}
    remove_cols = [col for col in dataset.column_names if col not in keep_cols]

    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
    rename_map = {}
    if audio_col != "audio":
        rename_map[audio_col] = "audio"
    if text_col != "sentence":
        rename_map[text_col] = "sentence"
    for old, new in rename_map.items():
        if new in dataset.column_names and old != new:
            dataset = dataset.remove_columns([new])
        dataset = dataset.rename_column(old, new)

    # Keep the encoded audio bytes here. datasets 5.x otherwise requires the
    # optional TorchCodec/FFmpeg stack when each row is accessed. SoundFile
    # handles the FLEURS WAV files used for this experiment directly.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    print(f"Standardized {dataset_name}: audio column -> audio, text column -> sentence")
    return dataset


def decode_audio_record(record: dict[str, Any]) -> dict[str, Any]:
    source = io.BytesIO(record["bytes"]) if record.get("bytes") is not None else record["path"]
    waveform, sample_rate = sf.read(source, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLING_RATE:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLING_RATE)
        sample_rate = SAMPLING_RATE
    return {"array": waveform, "sampling_rate": sample_rate}


def limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def load_training_datasets(args: argparse.Namespace) -> Dataset:
    if args.split_manifest:
        if not args.use_fleurs_train or not args.skip_ddd or args.include_slr42:
            raise ValueError("Matched FLEURS runs require --use-fleurs-train --skip-ddd without --include-slr42")
        manifest = load_manifest(args.split_manifest)
        if manifest["seed"] != args.seed:
            raise ValueError("Training seed differs from the shared split manifest seed")
        raw = load_dataset(FLEURS_DATASET, FLEURS_CONFIG, split="train", cache_dir=args.cache_dir)
        return standardize_asr_dataset(select_manifest_split(raw, manifest, "train"), "matched FLEURS train")

    train_sets: list[Dataset] = []

    if args.use_fleurs_train or (args.skip_ddd and not args.include_slr42):
        print("Loading FLEURS km_kh train split (lightweight dataset)...")
        fleurs_raw = load_dataset(FLEURS_DATASET, FLEURS_CONFIG, split="train", cache_dir=args.cache_dir)
        train_sets.append(standardize_asr_dataset(fleurs_raw, "FLEURS train"))

    if not args.skip_ddd:
        ddd_raw = load_first_available_dataset(DDD_DATASET_ALIASES, cache_dir=args.cache_dir)
        ddd_train = choose_split(ddd_raw, ("train", "training", "data"))
        train_sets.append(standardize_asr_dataset(ddd_train, "DDD Khmer speech"))

    if args.include_slr42:
        try:
            print("WARNING: OpenSLR SLR42 is a very large dataset (~100 hours). Download and extraction will require significant time.")
            slr42 = load_dataset(OPENSLR_DATASET, OPENSLR_CONFIG, split="train", cache_dir=args.cache_dir)
            train_sets.append(standardize_asr_dataset(slr42, "OpenSLR SLR42"))
        except Exception as exc:  # noqa: BLE001 - optional dataset should not stop DDD training.
            print(f"Skipping optional OpenSLR SLR42 because it could not be loaded or normalized: {exc}")

    if not train_sets:
        raise RuntimeError("No training datasets were loaded. Use DDD, --use-fleurs-train, or enable --include-slr42.")

    train = train_sets[0] if len(train_sets) == 1 else concatenate_datasets(train_sets)
    return limit_dataset(train.shuffle(seed=args.seed), args.max_train_samples)


def load_fleurs_eval_sets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    fleurs = load_dataset(FLEURS_DATASET, FLEURS_CONFIG, cache_dir=args.cache_dir)
    if args.split_manifest:
        manifest = load_manifest(args.split_manifest)
        validation = standardize_asr_dataset(
            select_manifest_split(fleurs["validation"], manifest, "validation"), "matched FLEURS validation"
        )
        test = standardize_asr_dataset(
            select_manifest_split(fleurs["test"], manifest, "test"), "matched FLEURS test"
        )
        return validation, test
    validation = standardize_asr_dataset(choose_split(fleurs, ("validation", "dev", "valid")), "FLEURS validation")
    test = standardize_asr_dataset(choose_split(fleurs, ("test",)), "FLEURS test")
    return limit_dataset(validation, args.max_eval_samples), limit_dataset(test, args.max_test_samples)


def prepare_dataset_fn(processor: WhisperProcessor, max_target_positions: int = 448):
    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = decode_audio_record(batch["audio"])
        sentence = normalize_khmer_text(batch["sentence"])
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            sentence,
            truncation=True,
            max_length=max_target_positions,
        ).input_ids
        batch["label_length"] = len(batch["labels"])
        return batch

    return prepare


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    max_target_positions: int = 448

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Absolute safety guard: ensure label length never exceeds Whisper's decoder position limit (448)
        if labels.shape[1] > self.max_target_positions:
            labels = labels[:, : self.max_target_positions]

        batch["labels"] = labels
        return batch


def build_compute_metrics(processor: WhisperProcessor, cache_dir: str | None = None):
    wer_metric = evaluate.load("wer", cache_dir=cache_dir)
    cer_metric = evaluate.load("cer", cache_dir=cache_dir)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_khmer_text(text) for text in pred_str]
        label_str = [normalize_khmer_text(text) for text in label_str]

        # Khmer spaces mark phrase boundaries, not word boundaries. Score CER
        # without whitespace so it matches the MMS evaluation policy.
        cer_pred = [re.sub(r"\s+", "", text) for text in pred_str]
        cer_ref = [re.sub(r"\s+", "", text) for text in label_str]
        return {
            "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": 100 * cer_metric.compute(predictions=cer_pred, references=cer_ref),
        }

    return compute_metrics


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    if not args.output_dir or args.output_dir.strip() == "":
        args.output_dir = "./whisper-tiny-khmer"
    if not args.cache_dir or args.cache_dir.strip() == "":
        args.cache_dir = DEFAULT_CACHE_DIR

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.cache_dir, "hub")
        os.environ["TORCH_HOME"] = os.path.join(args.cache_dir, "torch")
        t_dir = os.path.join(args.cache_dir, "temp")
        os.makedirs(t_dir, exist_ok=True)
        os.environ["TEMP"] = t_dir
        os.environ["TMP"] = t_dir
        os.environ["TMPDIR"] = t_dir
        tempfile.tempdir = t_dir

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware compute environment: {device_name}")

    if args.max_train_samples is None:
        print(
            "Full training can require dozens of GB for downloads, cached features, and checkpoints. "
            "Use --max-train-samples for a smaller run or --cache-dir on a larger drive."
        )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=LANGUAGE, task=TASK, cache_dir=args.cache_dir)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=LANGUAGE, task=TASK, cache_dir=args.cache_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    model.generation_config.language = LANGUAGE.lower()
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.freeze_encoder:
        print("Freezing Whisper encoder weights (Linear Probe / Transfer Learning strategy)...")
        model.freeze_encoder()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    train_dataset = load_training_datasets(args)
    eval_dataset, test_dataset = load_fleurs_eval_sets(args)

    max_label_length = getattr(model.config, "max_target_positions", 448)
    prepare = prepare_dataset_fn(processor, max_target_positions=max_label_length)
    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        prepare,
        remove_columns=remove_columns,
        num_proc=args.preprocessing_num_workers,
        desc="Preparing train features",
    )
    eval_dataset = eval_dataset.map(
        prepare,
        remove_columns=eval_dataset.column_names,
        num_proc=args.preprocessing_num_workers,
        desc="Preparing validation features",
    )
    test_dataset = test_dataset.map(
        prepare,
        remove_columns=test_dataset.column_names,
        num_proc=args.preprocessing_num_workers,
        desc="Preparing test features",
    )

    # Filter out audio samples where label exceeds or equals max_target_positions to prevent Whisper crash
    train_dataset = train_dataset.filter(
        lambda length: length < max_label_length,
        input_columns=["label_length"],
        desc="Filtering long training labels",
    )
    eval_dataset = eval_dataset.filter(
        lambda length: length < max_label_length,
        input_columns=["label_length"],
        desc="Filtering long eval labels",
    )
    test_dataset = test_dataset.filter(
        lambda length: length < max_label_length,
        input_columns=["label_length"],
        desc="Filtering long test labels",
    )
    if args.split_manifest:
        manifest = load_manifest(args.split_manifest)
        actual = (len(train_dataset), len(eval_dataset), len(test_dataset))
        expected = tuple(len(manifest["indices"][split]) for split in ("train", "validation", "test"))
        if actual != expected:
            raise ValueError(f"Whisper dropped rows from the shared split: expected {expected}, got {actual}")
        print(f"Matched FLEURS rows (train/validation/test): {actual}")
    train_dataset = train_dataset.remove_columns(["label_length"])
    eval_dataset = eval_dataset.remove_columns(["label_length"])
    test_dataset = test_dataset.remove_columns(["label_length"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_target_positions=max_label_length,
    )

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "gradient_checkpointing": False,
        "fp16": args.fp16,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "report_to": ["tensorboard"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
    }
    strategy_arg = "eval_strategy" if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters else "evaluation_strategy"
    training_kwargs[strategy_arg] = "steps"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "args": training_args,
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": build_compute_metrics(processor, cache_dir=args.cache_dir),
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = feature_extractor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    train_metrics = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint).metrics
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    trainer.save_state()
    if args.trainer_state_output:
        args.trainer_state_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(args.output_dir) / "trainer_state.json", args.trainer_state_output)

    print("Evaluating best checkpoint on FLEURS km_kh test split.")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(test_metrics)
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(test_metrics)
        report.update({
            "train_metrics": train_metrics,
            "train_examples": len(train_dataset),
            "validation_examples": len(eval_dataset),
            "test_examples": len(test_dataset),
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "split_manifest": args.split_manifest,
        })
        args.metrics_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.push_to_hub:
        dataset_tags = [DDD_DATASET_ALIASES[0], FLEURS_DATASET]
        if args.include_slr42:
            dataset_tags.append(OPENSLR_DATASET)
        trainer.push_to_hub(
            language="km",
            model_name=args.hub_model_id or os.path.basename(args.output_dir),
            finetuned_from=args.model_name,
            tasks="automatic-speech-recognition",
            dataset_tags=dataset_tags,
            dataset="DDD Khmer speech + optional OpenSLR SLR42; evaluated on FLEURS km_kh",
        )


if __name__ == "__main__":
    main()
