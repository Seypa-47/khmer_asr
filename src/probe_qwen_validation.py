"""Benchmark a public Khmer ASR checkpoint on the fixed FLEURS validation rows.

This is model selection, not a student-trained approach or a held-out test result.
The script never reads the test split. It saves each prediction so a Colab restart
can resume without rerunning completed audio. References are loaded from FLEURS.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from jiwer import process_characters, process_words

from matched_fleurs import load_manifest
from score_matched_cer_wer import cer_text, word_tokens


def score_rows(rows: list[dict]) -> dict:
    references = [row["reference"] for row in rows]
    hypotheses = [row["hypothesis"] for row in rows]
    chars = process_characters(
        [cer_text(text) for text in references],
        [cer_text(text) for text in hypotheses],
    )
    words = process_words(
        [" ".join(word_tokens(text)) for text in references],
        [" ".join(word_tokens(text)) for text in hypotheses],
    )
    return {
        "evaluated_examples": len(rows),
        "cer_percent": 100 * chars.cer,
        "wer_percent_icu_segmented": 100 * words.wer,
        "cer_reference_characters": chars.hits + chars.substitutions + chars.deletions,
        "wer_reference_words": words.hits + words.substitutions + words.deletions,
    }


def save(path: Path, result: dict) -> None:
    result["metrics_so_far"] = score_rows(result["examples"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    score = result["metrics_so_far"]
    print(
        f"Saved {score['evaluated_examples']}/{result['validation_examples']} validation clips: "
        f"CER {score['cer_percent']:.2f}%, ICU WER {score['wer_percent_icu_segmented']:.2f}%",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/matched_fleurs_split.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="seanghay/Qwen3-ASR-0.6B-Khmer")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.save_every < 1 or args.batch_size < 1:
        parser.error("--save-every and --batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime for the Qwen validation probe")

    manifest = load_manifest(args.manifest)
    indices = manifest["indices"]["validation"]
    dataset = load_dataset("google/fleurs", "km_kh", split="validation")
    if max(indices) >= len(dataset):
        raise ValueError("Validation manifest index exceeds loaded FLEURS split")
    audio_column = dataset.data.column("audio")
    text_column = dataset.data.column("transcription")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result["model"] != args.model or result["validation_indices"] != indices:
            raise ValueError("Saved prediction file uses a different model or manifest")
    else:
        result = {
            "purpose": "public-model validation probe; not a student-trained approach",
            "dataset": "google/fleurs km_kh validation",
            "model": args.model,
            "validation_indices": indices,
            "validation_examples": len(indices),
            "cer_policy": "NFC; remove U+200B/U+FEFF and all whitespace; punctuation retained",
            "wer_policy": "NFC; remove U+200B/U+FEFF; ICU km_KH word breaks; discard punctuation and whitespace segments",
            "examples": [],
        }

    completed = len(result["examples"])
    if [row["validation_index"] for row in result["examples"]] != indices[:completed]:
        raise ValueError("Saved examples are not an exact prefix of the validation manifest")
    if completed == len(indices):
        save(args.output, result)
        return

    from qwen_asr import Qwen3ASRModel

    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)}", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda:0",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=256,
    )
    last_saved = completed
    for position in range(completed, len(indices), args.batch_size):
        batch_indices = indices[position : position + args.batch_size]
        batch_audio = []
        batch_references = []
        for index in batch_indices:
            batch_references.append(str(text_column[index].as_py()))
            record = audio_column[index].as_py()
            if record["bytes"] is None:
                raise ValueError(f"Missing embedded audio for validation index {index}")
            audio, sample_rate = sf.read(io.BytesIO(record["bytes"]), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            waveform = np.ascontiguousarray(audio, dtype=np.float32)
            batch_audio.append((waveform, sample_rate))
        predictions = model.transcribe(audio=batch_audio)
        if len(predictions) != len(batch_indices):
            raise RuntimeError("Qwen returned the wrong number of batch predictions")
        for index, reference, prediction in zip(batch_indices, batch_references, predictions):
            result["examples"].append(
                {
                    "validation_index": index,
                    "reference": reference,
                    "hypothesis": str(prediction.text),
                }
            )
        if len(result["examples"]) - last_saved >= args.save_every or len(result["examples"]) == len(indices):
            save(args.output, result)
            last_saved = len(result["examples"])


if __name__ == "__main__":
    main()
