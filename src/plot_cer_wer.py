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
    approaches = [
        ("whisper_tiny", "Whisper-Tiny\nSeq2Seq", "#232e66"),
        ("mms_1b_ctc", "MMS-1B\nTop 4 tuned", "#19856e"),
    ]
    if "mms_frozen_ctc" in models:
        approaches.append(("mms_frozen_ctc", "MMS-1B\nHead only", "#dd8b36"))
    labels = [label for _, label, _ in approaches]
    colors = [color for _, _, color in approaches]
    metrics = [
        ("CER", "cer_percent", "Characters, spaces removed"),
        ("WER", "wer_percent_icu_segmented", "ICU Khmer word breaks"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (title, key, subtitle) in zip(axes, metrics):
        values = [models[name][key] for name, _, _ in approaches]
        bars = ax.bar(labels, values, color=colors, width=0.54)
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
        ax.axhline(20, color="#ad3930", linestyle="--", linewidth=1.4, label="Target: below 20%")
        ax.set_ylim(0, max(115, max(values) * 1.13))
        ax.set_ylabel("Error rate (%)")
        ax.set_title(f"{title}\n{subtitle}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Matched FLEURS Khmer test: 114 clips", fontsize=14, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
