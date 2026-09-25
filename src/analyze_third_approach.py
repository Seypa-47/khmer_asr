"""Summarize the completed frozen-MMS run from saved, matched evidence."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jiwer import process_characters, process_words

from score_matched_cer_wer import cer_text, load_pairs, word_tokens


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def points(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    rows = [(int(row["step"]), float(row[key])) for row in history if key in row]
    return [step for step, _ in rows], [value for _, value in rows]


def plot_learning_curves(state: dict) -> None:
    history = state["log_history"]
    panels = [
        ("loss", "Training CTC loss", "CTC loss", "#187c75"),
        ("eval_cer", "Validation CER", "CER (%)", "#26336b"),
        ("eval_wer_icu", "Validation ICU WER", "WER (%)", "#ad5a34"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    best_step = int(Path(state["best_model_checkpoint"]).name.split("-")[-1])
    for axis, (key, title, ylabel, color) in zip(axes, panels):
        steps, values = points(history, key)
        if not steps:
            raise ValueError(f"Missing saved training series: {key}")
        axis.plot(steps, values, marker="o" if key != "loss" else None, markersize=3, color=color)
        if key.startswith("eval_"):
            axis.axvline(best_step, color="#6c6674", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.25)
    figure.suptitle(f"Frozen MMS encoder: validation WER selected checkpoint {best_step}", fontsize=13)
    figure.savefig(RESULTS / "mms_frozen_learning_curves.png", dpi=180)
    plt.close(figure)


def edit_counts(reference: str, hypothesis: str, *, words: bool) -> tuple[int, int]:
    if words:
        output = process_words(" ".join(word_tokens(reference)), " ".join(word_tokens(hypothesis)))
    else:
        output = process_characters(cer_text(reference), cer_text(hypothesis))
    return output.substitutions + output.deletions + output.insertions, output.hits + output.substitutions + output.deletions


def paired_interval(rows: list[tuple[int, int, int]], *, repeats: int = 4000) -> tuple[float, float, float]:
    """Return tuned minus frozen corpus error rate, in percentage points."""
    rng = random.Random(42)
    size = len(rows)
    differences = []
    for _ in range(repeats):
        sample = [rows[rng.randrange(size)] for _ in range(size)]
        denominator = sum(row[2] for row in sample)
        differences.append(100 * sum(row[0] - row[1] for row in sample) / denominator)
    differences.sort()
    denominator = sum(row[2] for row in rows)
    observed = 100 * sum(row[0] - row[1] for row in rows) / denominator
    return observed, differences[int(0.025 * repeats)], differences[int(0.975 * repeats)]


def analyze() -> None:
    state = json.loads((RESULTS / "mms_frozen_trainer_state.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS / "matched_cer_wer_three_approaches.json").read_text(encoding="utf-8"))
    tuned_indices, tuned_references, tuned_hypotheses, _ = load_pairs(RESULTS / "mms_matched_best300_predictions.json")
    frozen_indices, frozen_references, frozen_hypotheses, _ = load_pairs(RESULTS / "mms_frozen_predictions.json")
    if tuned_indices != frozen_indices or tuned_references != frozen_references:
        raise ValueError("The two MMS runs must use identical held-out reference clips")
    if len(tuned_indices) != summary["test_examples"]:
        raise ValueError("Prediction count differs from the saved comparison")

    plot_learning_curves(state)
    lines = [
        "# Third approach: frozen MMS encoder with a trained CTC head",
        "",
        "Both MMS runs used the same 531/124/114 FLEURS Khmer split. The third run trained only 194,712 head parameters; its 964,843,288-parameter model otherwise stayed frozen. Validation ICU WER selected checkpoint 100 of 335 optimizer steps across five epochs.",
        "",
        "| Approach | Held-out CER | Held-out ICU WER |",
        "|---|---:|---:|",
        f"| MMS, top four encoder layers and head trained | {summary['models']['mms_1b_ctc']['cer_percent']:.2f}% | {summary['models']['mms_1b_ctc']['wer_percent_icu_segmented']:.2f}% |",
        f"| MMS, head only trained | {summary['models']['mms_frozen_ctc']['cer_percent']:.2f}% | {summary['models']['mms_frozen_ctc']['wer_percent_icu_segmented']:.2f}% |",
        "",
    ]
    for label, words in (("CER", False), ("WER", True)):
        rows = []
        for ref, tuned, frozen in zip(tuned_references, tuned_hypotheses, frozen_hypotheses):
            tuned_errors, units = edit_counts(ref, tuned, words=words)
            frozen_errors, frozen_units = edit_counts(ref, frozen, words=words)
            if units != frozen_units:
                raise ValueError("Reference unit count changed between approaches")
            rows.append((tuned_errors, frozen_errors, units))
        improvement, lower, upper = paired_interval(rows)
        improved = sum(tuned > frozen for tuned, frozen, _ in rows)
        worsened = sum(tuned < frozen for tuned, frozen, _ in rows)
        tied = len(rows) - improved - worsened
        lines.append(
            f"{label} improvement (top-four tuned minus head-only): {improvement:.2f} percentage points. "
            f"Paired utterance bootstrap 95% interval: {lower:.2f} to {upper:.2f} points "
            f"(4,000 seed-42 resamples). Across clips: {improved} improved, {worsened} worsened, {tied} tied."
        )
    lines += [
        "",
        "The small measured difference does not establish a reliable improvement if its bootstrap interval crosses zero. Both CER and WER use the documented shared normalization and ICU Khmer word breaks. The WER remains far above the lecturer's under-20% error target.",
        "",
        "The saved prediction pairs contain public FLEURS references and model outputs. Personal recordings and the student's pasted test sentences were not used as training labels, test references, or hardcoded corrections.",
    ]
    (RESULTS / "mms_frozen_error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved frozen MMS learning curves and paired error analysis")


if __name__ == "__main__":
    analyze()
