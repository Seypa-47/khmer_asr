#!/usr/bin/env python
"""Score saved raw audio predictions against explicit, user-supplied references."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from jiwer import cer, process_characters


def normalize(text: str, remove_punctuation: bool = False) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    if remove_punctuation:
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", "", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))["references"]
    by_name = {row["filename"]: row for row in predictions["files"]}
    missing = sorted(set(references) - set(by_name))
    if missing:
        raise ValueError(f"Missing predictions: {missing}")

    rows = []
    for filename, reference in references.items():
        prediction = by_name[filename]["prediction"]
        normalized_reference = normalize(reference)
        normalized_prediction = normalize(prediction)
        rows.append({
            "filename": filename,
            "duration_seconds": by_name[filename]["duration_seconds"],
            "reference": reference,
            "prediction": prediction,
            "cer_percent": 100 * cer(normalized_reference, normalized_prediction),
            "cer_percent_without_punctuation": 100 * cer(
                normalize(reference, True), normalize(prediction, True)
            ),
        })

    refs = [normalize(row["reference"]) for row in rows]
    hyps = [normalize(row["prediction"]) for row in rows]
    edits = process_characters(refs, hyps)
    summary = {
        "reference_status": "User-pasted text; exact audio wording not independently verified",
        "model_choice": predictions["model_choice"],
        "sample_count": len(rows),
        "total_audio_seconds": sum(row["duration_seconds"] for row in rows),
        "cer_percent": 100 * cer(refs, hyps),
        "cer_percent_without_punctuation": 100 * cer(
            [normalize(row["reference"], True) for row in rows],
            [normalize(row["prediction"], True) for row in rows],
        ),
        "reference_characters": sum(map(len, refs)),
        "substitutions": edits.substitutions,
        "deletions": edits.deletions,
        "insertions": edits.insertions,
        "normalization": "NFC; remove U+200B/U+FEFF and whitespace; punctuation remains in primary score",
        "notes": "Separate user-recorded domain from FLEURS. No test reference was used to change model output.",
    }
    output = {"summary": summary, "examples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
