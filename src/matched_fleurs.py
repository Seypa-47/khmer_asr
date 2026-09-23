"""Create and load one fixed FLEURS row manifest for both ASR approaches.

Run ``python src/matched_fleurs.py`` before the matched Colab experiments.
Rows are selected from the official FLEURS splits, then the same eligibility
rules are applied before either model sees a row.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

DATASET = "google/fleurs"
CONFIG = "km_kh"
SPLITS = ("train", "validation", "test")


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != DATASET or manifest.get("config") != CONFIG:
        raise ValueError("Expected a version-1 Google FLEURS km_kh split manifest")
    for split in SPLITS:
        rows = manifest.get("indices", {}).get(split)
        if not isinstance(rows, list) or not rows or any(type(i) is not int or i < 0 for i in rows):
            raise ValueError(f"Invalid or empty {split} indices in split manifest")
        if len(rows) != len(set(rows)):
            raise ValueError(f"Duplicate {split} indices in split manifest")
    return manifest


def select_manifest_split(dataset, manifest: dict, split: str):
    indices = manifest["indices"][split]
    if max(indices) >= len(dataset):
        raise ValueError(f"{split} manifest index exceeds the loaded FLEURS split")
    return dataset.select(indices)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/matched_fleurs_split.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-candidates", type=int, default=1000)
    parser.add_argument("--validation-candidates", type=int, default=200)
    parser.add_argument("--test-candidates", type=int, default=200)
    parser.add_argument("--test-start", type=int, default=200, help="Skip earlier test rows already inspected in historical runs.")
    parser.add_argument("--max-duration-seconds", type=float, default=15.0)
    parser.add_argument("--whisper-max-label-tokens", type=int, default=448)
    args = parser.parse_args()

    from datasets import Audio, load_dataset
    from transformers import WhisperTokenizer

    fleurs = load_dataset(DATASET, CONFIG)
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-tiny", language="Khmer", task="transcribe"
    )
    candidate_limits = {
        "train": args.train_candidates,
        "validation": args.validation_candidates,
        "test": args.test_candidates,
    }
    indices = {}
    for split in SPLITS:
        dataset = fleurs[split].cast_column("audio", Audio(sampling_rate=16_000))
        candidates = list(range(len(dataset)))
        if split == "train":
            random.Random(args.seed).shuffle(candidates)
        elif split == "test":
            candidates = candidates[args.test_start:]
        candidates = candidates[: candidate_limits[split]]
        kept = []
        for index in candidates:
            example = dataset[index]
            audio = example["audio"]
            duration = len(audio["array"]) / audio["sampling_rate"]
            label_length = len(tokenizer(normalize_text(example["transcription"])).input_ids)
            if duration <= args.max_duration_seconds and label_length < args.whisper_max_label_tokens:
                kept.append(index)
        if not kept:
            raise ValueError(f"No {split} rows remain after shared filtering")
        indices[split] = kept
        print(f"{split}: {len(kept)} of {len(candidates)} candidate rows retained")

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "config": CONFIG,
        "seed": args.seed,
        "selection": "Seeded Python shuffle for train; official validation/test order",
        "test_start": args.test_start,
        "eligibility": {
            "max_duration_seconds": args.max_duration_seconds,
            "whisper_max_label_tokens": args.whisper_max_label_tokens,
        },
        "indices": indices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved shared row manifest: {args.output}")


if __name__ == "__main__":
    main()
