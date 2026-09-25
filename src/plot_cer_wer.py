"""Plot the matched held-out CER/WER results from audited prediction pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/matched_cer_wer_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/matched_cer_wer.png"))
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    if not data.get("comparison_complete"):
        raise ValueError("Both models must be scored before plotting")
    models = data["models"]
    labels = ["Whisper-Tiny\nSeq2Seq", "MMS-1B\nCTC"]
    colors = ["#232e66", "#19856e"]
    metrics = [
        ("CER", "cer_percent", "Characters, spaces removed"),
        ("WER", "wer_percent_icu_segmented", "ICU Khmer word breaks"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (title, key, subtitle) in zip(axes, metrics):
        values = [models["whisper_tiny"][key], models["mms_1b_ctc"][key]]
        bars = ax.bar(labels, values, color=colors, width=0.54)
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
        ax.set_ylim(0, max(115, max(values) * 1.13))
        ax.set_ylabel("Error rate (%)")
        ax.set_title(f"{title}\n{subtitle}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Matched FLEURS Khmer test: 114 clips", fontsize=14, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
