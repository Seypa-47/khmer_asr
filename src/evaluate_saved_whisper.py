#!/usr/bin/env python
"""Evaluate a saved Whisper checkpoint on the first N FLEURS test examples.

CER removes whitespace before scoring because Khmer spaces mark phrase
boundaries rather than word boundaries. Raw references and hypotheses are
retained in the JSON output so the score can be audited.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import unicodedata
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset
from jiwer import cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def normalize(text: str, *, remove_spaces: bool = False) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+", "", text) if remove_spaces else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/whisper-tiny-khmer")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-length", type=int, default=225)
    parser.add_argument("--output", default="results/whisper_test_predictions.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model_dir = Path(args.model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True
    ).to(device).eval()
    model.generation_config.language = "khmer"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    dataset = load_dataset("google/fleurs", "km_kh", split=args.split)
    count = min(args.max_samples, len(dataset))
    refs: list[str] = []
    hyps: list[str] = []

    audio_column = dataset.data.column("audio")
    text_column = dataset.data.column("transcription")
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        waveforms = []
        for index in range(start, stop):
            audio_record = audio_column[index].as_py()
            audio, sampling_rate = sf.read(io.BytesIO(audio_record["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sampling_rate != 16_000:
                raise ValueError(f"Expected 16 kHz FLEURS audio, got {sampling_rate} Hz")
            waveforms.append(audio)

        inputs = processor(
            waveforms, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            token_ids = model.generate(
                inputs.input_features.to(device),
                attention_mask=inputs.attention_mask.to(device),
                max_length=args.max_length,
                language="khmer",
                task="transcribe",
            )
        refs.extend(normalize(text_column[index].as_py()) for index in range(start, stop))
        hyps.extend(normalize(text) for text in processor.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))

        if stop % 25 < args.batch_size or stop == count:
            print(f"Evaluated {stop}/{count}", flush=True)

    metric_refs = [normalize(text, remove_spaces=True) for text in refs]
    metric_hyps = [normalize(text, remove_spaces=True) for text in hyps]
    metrics = {
        "approach": "Whisper-Tiny full fine-tuning",
        "dataset": f"google/fleurs/km_kh {args.split} split",
        "test_examples": count,
        "test_indices": f"0:{count}",
        "cer_percent_whitespace_removed": 100 * cer(metric_refs, metric_hyps),
        "text_normalization": "NFC, remove U+200B/U+FEFF, collapse whitespace, then remove whitespace for CER",
        "generation_max_length": args.max_length,
        "model_checkpoint": str(model_dir),
        "references": refs,
        "hypotheses": hyps,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"references", "hypotheses"}}, indent=2))
    print(f"Saved audited predictions to {output_path}")


if __name__ == "__main__":
    main()
