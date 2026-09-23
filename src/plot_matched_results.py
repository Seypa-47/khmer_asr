"""Plot only newly saved matched-run evidence; never reuse diagnostic scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required matched-run evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def series(state: dict, key: str) -> tuple[list[float], list[float]]:
    points = [(float(row["step"]), float(row[key])) for row in state["log_history"] if key in row and "step" in row]
    return [p[0] for p in points], [p[1] for p in points]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    directory = args.results_dir
    names = ("whisper", "mms")
    states = {name: read_json(directory / f"{name}_matched_trainer_state.json") for name in names}
    metrics = {name: read_json(directory / f"{name}_matched_metrics.json") for name in names}

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"whisper": "#26336b", "mms": "#187c75"}
    for name in names:
        step, value = series(states[name], "eval_cer")
        if not step:
            raise ValueError(f"No validation CER history was saved for {name}")
        axes[0].plot(step, value, marker="o", label=name.upper(), color=colors[name])
        step, value = series(states[name], "loss")
        if not step:
            raise ValueError(f"No training loss history was saved for {name}")
        axes[1].plot(step, value, label=name.upper(), color=colors[name])
    axes[0].set(title="Validation CER during training", xlabel="Optimizer step", ylabel="CER (%)")
    axes[1].set(title="Training loss (different objectives)", xlabel="Optimizer step", ylabel="Training loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(directory / "matched_learning_curves.png", dpi=180)
    plt.close(figure)

    values = [float(metrics[name]["test_cer"]) for name in names]
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = axis.bar(["Whisper-Tiny", "MMS-1B CTC"], values, color=[colors[name] for name in names])
    axis.bar_label(bars, fmt="%.2f%%", padding=3)
    axis.set(title="CER on the identical held-out FLEURS rows", ylabel="Character error rate (%)")
    axis.set_ylim(0, max(values) * 1.16)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(directory / "matched_metrics_comparison.png", dpi=180)
    plt.close(figure)
    print(f"Saved matched learning curves and test comparison in {directory}")


if __name__ == "__main__":
    main()
