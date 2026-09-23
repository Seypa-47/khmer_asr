#!/usr/bin/env python
"""Re-score a saved MMS CTC checkpoint on the first N FLEURS test rows."""

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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", "", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/mms-khmer-ctc")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="results/mms_test_predictions.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    processor = AutoProcessor.from_pretrained(
        args.model_dir, target_lang="khm", local_files_only=True
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_dir, local_files_only=True, low_cpu_mem_usage=True
    ).eval()

    dataset = load_dataset("google/fleurs", "km_kh", split="test")
    count = min(args.max_samples, len(dataset))
    audio_column = dataset.data.column("audio")
    text_column = dataset.data.column("transcription")
    rows: list[dict[str, object]] = []

    for index in range(count):
        record = audio_column[index].as_py()
        audio, sample_rate = sf.read(io.BytesIO(record["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16 kHz FLEURS audio; got {sample_rate} Hz")

        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
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
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"Evaluated {index + 1}/{count}", flush=True)

    cer_value = 100 * cer(
        [normalize(str(row["reference"])) for row in rows],
        [normalize(str(row["hypothesis"])) for row in rows],
    )
    result = {
        "approach": "Meta MMS-1B Khmer CTC saved checkpoint",
        "dataset": "google/fleurs/km_kh test split",
        "test_examples": count,
        "test_indices": f"0:{count}",
        "cer_percent_whitespace_removed": cer_value,
        "text_normalization": "NFC, remove U+200B/U+FEFF, then remove whitespace",
        "model_checkpoint": args.model_dir,
        "examples": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MMS CER on first {count}: {cer_value:.2f}%")
    print(f"Saved references and predictions to {output_path}")


if __name__ == "__main__":
    main()
