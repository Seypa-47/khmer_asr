#!/usr/bin/env python
"""Build comparison artifacts from saved, auditable evaluation files.

This script intentionally fails if the Whisper prediction file is absent. It
does not insert placeholder scores or synthesize training history.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def find_whisper_state() -> Path | None:
    candidates = [
        ROOT / "models/whisper-tiny-khmer/trainer_state.json",
        RESULTS / "whisper_trainer_state_backup.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def make_learning_curves(state_path: Path | None, output: Path) -> None:
    history = read_json(state_path).get("log_history", []) if state_path else []
    train = [(row["step"], row["loss"]) for row in history if "loss" in row]
    val_loss = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
    val_cer = [(row["step"], row["eval_cer"]) for row in history if "eval_cer" in row]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if train:
        ax.plot(*zip(*train), label="Whisper train loss", color="#1f77b4", marker="o", ms=3)
    if val_loss:
        ax.plot(*zip(*val_loss), label="Whisper validation loss", color="#ff7f0e", marker="s", ms=5)
    ax.set(title="Whisper loss history and validation CER", xlabel="Optimization step", ylabel="Loss")
    ax.grid(alpha=0.25)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if val_cer:
        cer_axis = ax.twinx()
        line, = cer_axis.plot(*zip(*val_cer), label="Whisper validation CER", color="#2ca02c", marker="^", ms=6)
        cer_axis.set_ylabel("Validation CER (%)")
        legend_handles.append(line)
        legend_labels.append("Whisper validation CER")
    ax.legend(legend_handles, legend_labels, loc="best")
    fig.text(0.5, 0.015, "Whisper history only; historical CER retained spaces. MMS trainer history was not retained.", ha="center", fontsize=9, color="#8a3b12")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_comparison_chart(records: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [row["name"] for row in records]
    values = [row["test_cer_percent"] for row in records]
    bars = ax.bar(labels, values, color=["#315b8a", "#2f9b8f", "#688bb0"], width=0.55)
    for bar, row in zip(bars, records):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{row['test_cer_percent']:.1f}%\n(n={row['test_examples']})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Character error rate (%) - lower is better")
    ax.set_title("Saved checkpoints on the first 200 FLEURS test examples")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, "Diagnostic only: saved runs used different selected examples and filtering.", ha="center", fontsize=9, color="#8a3b12")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    whisper_path = RESULTS / "whisper_test_predictions.json"
    if not whisper_path.is_file():
        raise FileNotFoundError(
            f"{whisper_path} is missing. Run src/evaluate_saved_whisper.py first."
        )
    mms_candidates = [RESULTS / "mms_test_metrics.json", ROOT / "models/results/mms_test_metrics.json"]
    mms_path = next((path for path in mms_candidates if path.is_file()), None)
    if mms_path is None:
        raise FileNotFoundError("Saved MMS test metrics are missing.")

    whisper = read_json(whisper_path)
    mms = read_json(mms_path)
    rerun_path = RESULTS / "whisper_warmup35_test_predictions.json"
    rerun = read_json(rerun_path) if rerun_path.is_file() else None
    records = [
        {
            "name": "Whisper-Tiny full fine-tuning",
            "architecture": "Seq2Seq Transformer",
            "training_strategy": "Full fine-tuning",
            "trainable_parameters_m": 37.8,
            "test_cer_percent": whisper["cer_percent_whitespace_removed"],
            "test_examples": whisper["test_examples"],
            "saved_training_examples": 941,
            "saved_validation_examples": 189,
            "data_count_status": "inferred from 354 optimizer steps and validation throughput; exact original selected indices not retained",
            "training_epochs": 3,
            "training_time": "Not recorded",
        },
        {
            "name": "Meta MMS-1B CTC",
            "architecture": "Wav2Vec 2.0 with CTC",
            "training_strategy": "Top four encoder layers plus CTC head/adapter unfrozen",
            "trainable_parameters_m": 79.08,
            "test_cer_percent": mms["test_cer"],
            "test_examples": 200,
            "saved_training_examples": 1000,
            "saved_validation_examples": 200,
            "data_count_status": "requested counts in saved training code; actual retained counts after duration filtering not recorded",
            "training_epochs": 15,
            "training_time": "Not recorded",
        },
    ]
    if rerun is not None:
        records.append({
            "name": "Whisper-Tiny 35-step warmup",
            "architecture": "Seq2Seq Transformer",
            "training_strategy": "Full fine-tuning, 35-step warmup",
            "trainable_parameters_m": 37.8,
            "test_cer_percent": rerun["cer_percent_whitespace_removed"],
            "test_examples": rerun["test_examples"],
            "saved_training_examples": 941,
            "saved_validation_examples": 189,
            "data_count_status": "941 and 189 retained after label filtering; seed-42 selected subsets",
            "training_epochs": 3,
            "training_time": "About 11 minutes on RTX 4050 Laptop GPU",
        })
    RESULTS.mkdir(exist_ok=True)
    make_learning_curves(find_whisper_state(), RESULTS / "learning_curves.png")
    make_comparison_chart(records, RESULTS / "metrics_comparison.png")
    summary = {
        "dataset": "Google FLEURS km_kh",
        "test_subset": "first 200 rows of official test split, shared by all listed checkpoints",
        "cer_policy": "NFC text, remove U+200B/U+FEFF, then remove whitespace before CER",
        "comparison_status": "diagnostic_only",
        "comparability_note": "Saved runs did not retain identical train/validation example manifests and applied different filtering. The Whisper effective counts are inferred; MMS retained counts were not logged. Do not claim a controlled architecture comparison.",
        "approaches": records,
    }
    (RESULTS / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Saved checkpoint comparison (diagnostic)",
        "",
        "All listed checkpoints are scored on the first 200 examples of the official Google FLEURS `km_kh` test split. CER uses NFC normalization, removes U+200B/U+FEFF, then removes whitespace before scoring.",
        "",
        "| Approach | Training strategy | Trainable parameters | Train / validation evidence | Epochs | Test CER | Test rows | Training time |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['name']} | {row['training_strategy']} | {row['trainable_parameters_m']:.2f}M | "
            f"{('about 941 / 189 retained (inferred)' if row['name'] == 'Whisper-Tiny full fine-tuning' else '941 / 189 retained' if row['name'].startswith('Whisper') else '1,000 / 200 requested; retained counts unlogged')} | {row['training_epochs']} | "
            f"{row['test_cer_percent']:.2f}% | {row['test_examples']} | {row['training_time']} |"
        )
    lines += [
        "",
        "> These results are diagnostic: the runs did not retain identical train/validation example manifests and applied different filtering. The shared test evaluation does not remove that training-data difference.",
        "",
        "> MMS training history was not saved, so its training/validation learning curve cannot be reconstructed from the available files. The frozen-encoder run also has no saved checkpoint or metrics and is excluded.",
    ]
    (RESULTS / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Updated results/metrics_summary.json, summary_table.md, and comparison figures.")


if __name__ == "__main__":
    main()
