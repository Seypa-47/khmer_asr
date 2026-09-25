"""Prepare a separate Qwen data-scaling experiment without touching test audio.

The original comparison manifest stays unchanged. This follow-up includes every
official FLEURS train clip of at most 15 seconds while keeping its validation
and held-out test row lists fixed. It is not a matched-training-size comparison.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

from matched_fleurs import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=Path("results/matched_fleurs_split.json"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic/qwen_expanded_train_split.json"))
    args = parser.parse_args()
    base = load_manifest(args.base_manifest)
    dataset = load_dataset("google/fleurs", "km_kh", split="train")
    audio_column = dataset.data.column("audio")
    indices = []
    for index in range(len(dataset)):
        record = audio_column[index].as_py()
        if record["bytes"] is None:
            raise ValueError(f"FLEURS train index {index} lacks audio bytes")
        with sf.SoundFile(io.BytesIO(record["bytes"])) as audio_file:
            duration = len(audio_file) / audio_file.samplerate
        if duration <= 15.0:
            indices.append(index)
    random.Random(base["seed"]).shuffle(indices)
    if not set(base["indices"]["train"]).issubset(indices):
        raise ValueError("Expanded train selection omitted a row from the fixed comparison manifest")
    result = {
        "schema_version": 1,
        "dataset": "google/fleurs",
        "config": "km_kh",
        "seed": base["seed"],
        "selection": "All official train rows up to 15 seconds; seeded shuffle; fixed original validation/test rows",
        "not_a_matched_training_size_comparison": True,
        "parent_manifest": str(args.base_manifest),
        "eligibility": {"max_duration_seconds": 15.0},
        "indices": {
            "train": indices,
            "validation": base["indices"]["validation"],
            "test": base["indices"]["test"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(indices)} public training clips; validation remains {len(result['indices']['validation'])} clips")


if __name__ == "__main__":
    main()
