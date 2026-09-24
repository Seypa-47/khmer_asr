#!/usr/bin/env python
"""Re-score a saved MMS CTC checkpoint on selected FLEURS test rows."""

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
from transformers import AutoProcessor, Wav2Vec2ForCTC

try:
    from .matched_fleurs import load_manifest
except ImportError:
    from matched_fleurs import load_manifest


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", "", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/mms-khmer-ctc")
    parser.add_argument(
        "--processor-id",
        default=None,
        help="Processor source when a Trainer checkpoint lacks tokenizer files (e.g. facebook/mms-1b-all).",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split-manifest", default=None, help="Evaluate exactly the shared FLEURS test row indices.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", default="results/mms_test_predictions.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    processor = AutoProcessor.from_pretrained(
        args.processor_id or args.model_dir,
        target_lang="khm",
        local_files_only=args.processor_id is None,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_dir, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()

    dataset = load_dataset("google/fleurs", "km_kh", split="test")
    indices = (
        load_manifest(args.split_manifest)["indices"]["test"]
        if args.split_manifest else list(range(min(args.max_samples, len(dataset))))
    )
    count = len(indices)
    audio_column = dataset.data.column("audio")
    text_column = dataset.data.column("transcription")
    rows: list[dict[str, object]] = []

    for position, index in enumerate(indices, 1):
        record = audio_column[index].as_py()
        audio, sample_rate = sf.read(io.BytesIO(record["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16 kHz FLEURS audio; got {sample_rate} Hz")

        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        hypothesis = processor.batch_decode(logits.argmax(-1))[0]
        reference = text_column[index].as_py()
        rows.append(
            {
                "test_index": index,
                "reference": reference,
                "hypothesis": hypothesis,
                "cer_percent": 100 * cer(normalize(reference), normalize(hypothesis)),
                "audio_seconds": len(audio) / sample_rate,
            }
        )
        if position % 25 == 0 or position == count:
            print(f"Evaluated {position}/{count}", flush=True)

    cer_value = 100 * cer(
        [normalize(str(row["reference"])) for row in rows],
        [normalize(str(row["hypothesis"])) for row in rows],
    )
    result = {
        "approach": "Meta MMS-1B Khmer CTC saved checkpoint",
        "dataset": "google/fleurs/km_kh test split",
        "test_examples": count,
        "test_indices": indices if args.split_manifest else f"0:{count}",
        "cer_percent_whitespace_removed": cer_value,
        "text_normalization": "NFC, remove U+200B/U+FEFF, then remove whitespace",
        "model_checkpoint": args.model_dir,
        "examples": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MMS CER on {count} selected test rows: {cer_value:.2f}%")
    print(f"Saved references and predictions to {output_path}")


if __name__ == "__main__":
    main()
