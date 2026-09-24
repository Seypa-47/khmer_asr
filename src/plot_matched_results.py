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


def plot_completed(directory: Path) -> None:
    """Plot the completed matched comparison and saved MMS learning history."""
    whisper = read_json(directory / "whisper_matched_metrics.json")
    mms = read_json(directory / "mms_matched_best300_summary.json")
    if whisper["test_examples"] != mms["test_examples"]:
        raise ValueError("Comparison requires the same number of test examples")
    if mms["status"] != "completed_training_run_evaluated_from_best_validation_checkpoint":
        raise ValueError("MMS training must be complete before plotting final figures")

    values = [float(whisper["test_cer"]), float(mms["held_out_test_cer_percent_whitespace_removed"])]
    figure, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    bars = axis.bar(
        ["Whisper-Tiny", "MMS-1B CTC\n(best validation checkpoint)"],
        values,
        color=["#26336b", "#187c75"],
    )
    axis.bar_label(bars, fmt="%.2f%%", padding=3)
    axis.set(
        title=f"CER on the same {whisper['test_examples']} held-out FLEURS clips",
        ylabel="Character error rate (%) - lower is better",
    )
    axis.set_ylim(0, max(values) * 1.16)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(directory / "matched_test_cer.png", dpi=180)
    plt.close(figure)

    groups = [
        ("References with Latin letters", mms["latin_reference_examples"], mms["latin_reference_cer_percent"]),
        ("References without Latin letters", mms["no_latin_reference_examples"], mms["no_latin_reference_cer_percent"]),
    ]
    figure, axis = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    bars = axis.barh(
        [f"{label} (n={count})" for label, count, _ in groups],
        [value for _, _, value in groups],
        color=["#ad5a34", "#187c75"],
    )
    axis.bar_label(bars, labels=[f"{value:.2f}%" for _, _, value in groups], padding=4)
    axis.set(
        title="MMS test errors by reference script",
        xlabel="Corpus character error rate (%) - lower is better",
    )
    axis.set_xlim(0, max(value for _, _, value in groups) * 1.18)
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(directory / "mms_error_subgroups.png", dpi=180)
    plt.close(figure)

    history = read_json(directory / "mms_matched_learning_history.json")
    loss = history["train_loss"]
    steps = list(range(10, 10 * (len(loss) + 1), 10))
    if len(loss) != 100:
        raise ValueError("Expected 100 saved MMS training-loss points")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(steps, loss, color="#187c75", alpha=0.65, linewidth=1.4)
    axes[0].set(title="MMS training loss", xlabel="Optimizer step", ylabel="CTC loss")
    val_steps = history["validation_steps"]
    val_cer = history["validation_cer_percent"]
    axes[1].plot(val_steps, val_cer, marker="o", markersize=3, color="#26336b")
    best = history["best_validation_checkpoint_step"]
    axes[1].axvline(best, color="#ad5a34", linestyle="--", label=f"Best checkpoint: {best}")
    axes[1].legend()
    axes[1].set(title="MMS validation CER", xlabel="Optimizer step", ylabel="CER (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(directory / "mms_matched_learning_curves.png", dpi=180)
    plt.close(figure)
    print(f"Saved completed-run figures in {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--completed", action="store_true", help="Plot the completed matched comparison and MMS curves.")
    args = parser.parse_args()
    directory = args.results_dir
    if args.completed:
        plot_completed(directory)
        return
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

    values = [
        float(metrics[name].get("test_cer", metrics[name].get("held_out_test_cer_percent")))
        for name in names
    ]
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
