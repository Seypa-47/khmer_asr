"""Describe validation errors for a student-trained Qwen adapter, without test data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jiwer import process_characters, process_words

from score_matched_cer_wer import cer_text, word_tokens
from matched_fleurs import load_manifest


def metrics(rows: list[dict]) -> dict:
    chars = process_characters(
        [cer_text(row["reference"]) for row in rows],
        [cer_text(row["hypothesis"]) for row in rows],
    )
    words = process_words(
        [" ".join(word_tokens(row["reference"])) for row in rows],
        [" ".join(word_tokens(row["hypothesis"])) for row in rows],
    )
    return {
        "clips": len(rows),
        "reference_characters": chars.hits + chars.substitutions + chars.deletions,
        "character_errors": chars.substitutions + chars.deletions + chars.insertions,
        "cer_percent": 100 * chars.cer,
        "reference_words": words.hits + words.substitutions + words.deletions,
        "word_errors": words.substitutions + words.deletions + words.insertions,
        "wer_percent": 100 * words.wer,
        "word_substitutions": words.substitutions,
        "word_deletions": words.deletions,
        "word_insertions": words.insertions,
    }


def analyze(data: dict, manifest: dict) -> dict:
    if data.get("split", data.get("dataset")) != "google/fleurs km_kh validation":
        raise ValueError("Expected Qwen validation predictions; refusing to analyze another split")
    rows = data["examples"]
    index_list = [row["validation_index"] for row in rows]
    if index_list != manifest["indices"]["validation"]:
        raise ValueError("Predictions must match every validation manifest index in order")
    mixed = [row for row in rows if re.search(r"[A-Za-z0-9]", row["reference"])]
    khmer = [row for row in rows if not re.search(r"[A-Za-z0-9]", row["reference"])]
    return {
        "purpose": "public FLEURS validation failure analysis; no personal recordings or held-out test",
        "all": metrics(rows),
        "references_with_ascii_letters_or_digits": metrics(mixed) if mixed else None,
        "other_references": metrics(khmer) if khmer else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/matched_fleurs_split.json"))
    args = parser.parse_args()
    data = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = analyze(data, load_manifest(args.manifest))
    result["source"] = str(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}; WER all {result['all']['wer_percent']:.2f}%")


if __name__ == "__main__":
    main()
