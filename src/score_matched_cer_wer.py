"""Score saved FLEURS predictions with matched CER and Khmer-segmented WER.

WER uses ICU's Khmer word-break dictionary on both references and hypotheses.
It is a post-hoc metric: no model outputs or reference text are changed.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from importlib.metadata import version
from pathlib import Path

import icu
from jiwer import process_characters, process_words


def clean(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\u200b", "").replace("\ufeff", "")


def cer_text(text: str) -> str:
    return re.sub(r"\s+", "", clean(text))


def word_tokens(text: str) -> list[str]:
    value = icu.UnicodeString(clean(text))
    breaker = icu.BreakIterator.createWordInstance(icu.Locale("km_KH"))
    breaker.setText(value)
    start = breaker.first()
    tokens = []
    for end in breaker:
        # ICU status 0 marks punctuation and whitespace. Word, number, and
        # letter segments have positive statuses and count as words here.
        if breaker.getRuleStatus() > 0:
            tokens.append(str(value[start:end]))
        start = end
    return tokens


def load_pairs(path: Path) -> tuple[list[int], list[str], list[str], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "examples" in data:
        rows = data["examples"]
        indices = [row["test_index"] for row in rows]
        references = [row["reference"] for row in rows]
        hypotheses = [row["hypothesis"] for row in rows]
    else:
        indices = data["test_indices"]
        references = data["references"]
        hypotheses = data["hypotheses"]
    if not isinstance(indices, list) or not (len(indices) == len(references) == len(hypotheses)):
        raise ValueError(f"Invalid or incomplete prediction pairs: {path}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate test indices in {path}")
    return indices, references, hypotheses, data


def score(path: Path, expected_indices: list[int]) -> tuple[dict, list[str]]:
    indices, references, hypotheses, data = load_pairs(path)
    if indices != expected_indices:
        raise ValueError(f"Prediction indices do not match the saved test manifest: {path}")
    normalized_references = [cer_text(text) for text in references]
    normalized_hypotheses = [cer_text(text) for text in hypotheses]
    if any(not text for text in normalized_references):
        raise ValueError(f"Empty normalized reference in {path}")

    characters = process_characters(normalized_references, normalized_hypotheses)
    reference_words = [word_tokens(text) for text in references]
    hypothesis_words = [word_tokens(text) for text in hypotheses]
    if any(not words for words in reference_words):
        raise ValueError(f"Empty word segmentation of a reference in {path}")
    words = process_words(
        [" ".join(tokens) for tokens in reference_words],
        [" ".join(tokens) for tokens in hypothesis_words],
    )
    saved_cer = data.get("cer_percent_whitespace_removed", data.get("test_cer"))
    if saved_cer is not None and abs(100 * characters.cer - float(saved_cer)) > 0.01:
        raise ValueError(f"Recomputed CER disagrees with saved result in {path}")
    return {
        "prediction_file": path.as_posix(),
        "test_examples": len(indices),
        "cer_percent": 100 * characters.cer,
        "cer_reference_characters": characters.hits + characters.substitutions + characters.deletions,
        "cer_substitutions": characters.substitutions,
        "cer_deletions": characters.deletions,
        "cer_insertions": characters.insertions,
        "wer_percent_icu_segmented": 100 * words.wer,
        "wer_reference_words": words.hits + words.substitutions + words.deletions,
        "wer_substitutions": words.substitutions,
        "wer_deletions": words.deletions,
        "wer_insertions": words.insertions,
    }, normalized_references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/matched_fleurs_split.json"))
    parser.add_argument("--whisper", type=Path)
    parser.add_argument("--mms", type=Path)
    parser.add_argument("--frozen-mms", type=Path, help="Optional third approach: frozen MMS encoder with trained CTC head.")
    parser.add_argument("--output", type=Path, default=Path("results/matched_cer_wer_summary.json"))
    args = parser.parse_args()
    if not args.whisper and not args.mms and not args.frozen_mms:
        parser.error("Provide at least one saved prediction file")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_indices = manifest["indices"]["test"]
    models = {}
    shared_references = None
    for name, path in (("whisper_tiny", args.whisper), ("mms_1b_ctc", args.mms), ("mms_frozen_ctc", args.frozen_mms)):
        if path is None:
            continue
        models[name], references = score(path, expected_indices)
        if shared_references is not None and references != shared_references:
            raise ValueError("The models' normalized test references differ")
        shared_references = references
    result = {
        "dataset": "google/fleurs km_kh held-out test",
        "manifest": args.manifest.as_posix(),
        "test_examples": len(expected_indices),
        "cer_policy": "NFC; remove U+200B/U+FEFF and all whitespace; punctuation retained",
        "wer_policy": "NFC; remove U+200B/U+FEFF; ICU km_KH word breaks; discard punctuation and whitespace segments; corpus edit distance / reference words",
        "icu_version": icu.ICU_VERSION,
        "pyicu_wheels_version": version("pyicu-wheels"),
        "models": models,
        "comparison_complete": "whisper_tiny" in models and "mms_1b_ctc" in models,
        "third_approach_complete": "mms_frozen_ctc" in models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, metrics in models.items():
        print(f"{name}: CER {metrics['cer_percent']:.2f}%, ICU WER {metrics['wer_percent_icu_segmented']:.2f}%")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
