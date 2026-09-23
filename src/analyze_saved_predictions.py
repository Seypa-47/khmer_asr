#!/usr/bin/env python
"""Summarize character-level errors in saved Whisper test predictions."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from jiwer import cer, process_characters

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/whisper_test_predictions.json"
OUTPUT = ROOT / "results/error_analysis.md"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", "", text)


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    refs = [norm(text) for text in data["references"]]
    hyps = [norm(text) for text in data["hypotheses"]]
    output = process_characters(refs, hyps)
    order = sorted(range(len(refs)), key=lambda i: cer(refs[i], hyps[i]), reverse=True)

    lines = [
        "# Saved Whisper prediction error analysis",
        "",
        f"Evaluated {len(refs)} examples from the first 200 rows of the FLEURS `km_kh` test split. CER ignores whitespace after NFC and zero-width-character cleanup.",
        "",
        f"- Character error rate: {100 * output.cer:.2f}%",
        f"- Substitutions: {output.substitutions:,}",
        f"- Deletions: {output.deletions:,}",
        f"- Insertions: {output.insertions:,}",
        f"- Reference characters: {output.hits + output.substitutions + output.deletions:,}",
        "",
        "## Highest-error examples",
        "",
        "| FLEURS test row | Example CER | Reference | Whisper output |",
        "|---:|---:|---|---|",
    ]
    for i in order[:5]:
        lines.append(f"| {i} | {100 * cer(refs[i], hyps[i]):.1f}% | {refs[i]} | {hyps[i]} |")
    lines += [
        "",
        "These examples document the observed text differences. CER alone cannot establish whether a specific mismatch came from coeng clusters, vowel register, noise, or decoding. Listen to the corresponding public FLEURS clips before assigning an acoustic cause. MMS per-example predictions were not retained, so this error analysis covers Whisper only.",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT}")
    print(f"CER {100 * output.cer:.2f}%: substitutions={output.substitutions}, deletions={output.deletions}, insertions={output.insertions}")


if __name__ == "__main__":
    main()
