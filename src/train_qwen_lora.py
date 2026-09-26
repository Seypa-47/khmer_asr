"""Fine-tune a Khmer Qwen3-ASR LoRA adapter on the fixed FLEURS train split.

Only train and validation indices are read. The held-out test split is never
loaded by this program. Save checkpoints outside Git (typically Google Drive).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from jiwer import process_characters, process_words
from peft import LoraConfig, get_peft_model
from qwen_asr import Qwen3ASRModel

from matched_fleurs import load_manifest, normalize_text
from score_matched_cer_wer import cer_text, word_tokens


MODEL_ID = "seanghay/Qwen3-ASR-0.6B-Khmer"
OFFICIAL_HELPER_COMMIT = "7c6daf77a2421100f5fb066495372c00129d39ff"


def require_helpers(official_repo: Path):
    helper_dir = official_repo / "finetuning"
    if not (helper_dir / "qwen3_asr_sft.py").is_file():
        raise FileNotFoundError(f"Clone QwenLM/Qwen3-ASR at {OFFICIAL_HELPER_COMMIT} to {official_repo}")
    sys.path.insert(0, str(helper_dir))
    from qwen3_asr_sft import (  # noqa: PLC0415
        DataCollatorForQwen3ASRFinetuning,
        make_preprocess_fn_prefix_only,
        patch_outer_forward,
    )

    return DataCollatorForQwen3ASRFinetuning, make_preprocess_fn_prefix_only, patch_outer_forward


def make_audio_rows(split: str, indices: list[int], root: Path) -> list[dict]:
    """Extract only manifested audio; references come from FLEURS itself."""
    ds = load_dataset("google/fleurs", "km_kh", split=split)
    if max(indices) >= len(ds):
        raise ValueError(f"Manifest index exceeds FLEURS {split} split")
    audio_col = ds.data.column("audio")
    text_col = ds.data.column("transcription")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        wav_path = root / f"{split}_{index}.wav"
        if not wav_path.exists():
            raw = audio_col[index].as_py()["bytes"]
            if raw is None:
                raise ValueError(f"Missing FLEURS {split} audio {index}")
            waveform, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            sf.write(wav_path, waveform, sr, subtype="PCM_16")
        rows.append({
            "index": index,
            "audio": str(wav_path),
            "reference": normalize_text(text_col[index].as_py()),
        })
    return rows


def score_rows(rows: list[dict]) -> dict:
    chars = process_characters(
        [cer_text(r["reference"]) for r in rows],
        [cer_text(r["hypothesis"]) for r in rows],
    )
    words = process_words(
        [" ".join(word_tokens(r["reference"])) for r in rows],
        [" ".join(word_tokens(r["hypothesis"])) for r in rows],
    )
    return {
        "cer_percent": 100 * chars.cer,
        "wer_percent_icu_segmented": 100 * words.wer,
        "cer_reference_characters": chars.hits + chars.substitutions + chars.deletions,
        "wer_reference_words": words.hits + words.substitutions + words.deletions,
        "cer_errors": chars.substitutions + chars.deletions + chars.insertions,
        "wer_errors": words.substitutions + words.deletions + words.insertions,
    }


def evaluate(wrapper, model, rows: list[dict], output: Path, batch_size: int) -> dict:
    wrapper.model = model
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        results = wrapper.transcribe(audio=[r["audio"] for r in batch])
        if len(results) != len(batch):
            raise RuntimeError("Qwen returned an unexpected number of predictions")
        predictions.extend([
            {"validation_index": r["index"], "reference": r["reference"], "hypothesis": str(p.text)}
            for r, p in zip(batch, results)
        ])
        if (start // batch_size + 1) % 5 == 0:
            print(f"Validation generated {len(predictions)}/{len(rows)}", flush=True)
    result = {
        "purpose": "student-trained Qwen LoRA validation; checkpoint selection only",
        "split": "google/fleurs km_kh validation",
        "examples": predictions,
        "metrics": score_rows(predictions),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result["metrics"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("results/matched_fleurs_split.json"))
    ap.add_argument("--official-repo", type=Path, default=Path("/content/Qwen3-ASR"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/content/qwen_fleurs_audio"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for Qwen fine-tuning")
    if args.epochs < 1 or args.grad_acc < 1 or args.eval_batch_size < 1:
        ap.error("epochs, grad-acc, and eval-batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    manifest = load_manifest(args.manifest)
    train_rows = make_audio_rows("train", manifest["indices"]["train"], args.cache_dir)
    val_rows = make_audio_rows("validation", manifest["indices"]["validation"], args.cache_dir)
    config = {
        "base_model": MODEL_ID,
        "official_qwen_helper_commit": OFFICIAL_HELPER_COMMIT,
        "manifest": str(args.manifest),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_used": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "gradient_accumulation": args.grad_acc,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
        "selection_metric": "validation ICU Khmer WER",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    collator_class, preprocess_fn, patch_forward = require_helpers(args.official_repo)
    wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cuda:0",
        max_inference_batch_size=args.eval_batch_size,
        max_new_tokens=256,
    )
    base = wrapper.model
    patch_forward(base)
    base.gradient_checkpointing_disable()  # outer input-embedding API is absent on this model
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable LoRA parameters: {trainable}", flush=True)
    collate = collator_class(wrapper.processor)
    prepare = preprocess_fn(wrapper.processor)
    prepared = [prepare({"audio": r["audio"], "text": "language Khmer<asr_text>" + r["reference"], "prompt": ""}) for r in train_rows]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    steps_per_epoch = math.ceil(len(prepared) / args.grad_acc)
    global_step = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(prepared)))
        random.shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for position, index in enumerate(order, start=1):
            batch = collate([prepared[index]])
            batch = {k: v.to("cuda", dtype=torch.float16) if v.is_floating_point() else v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, example {position}")
            loss_sum += float(loss.detach())
            # Average the final partial group over its actual number of clips.
            group_start = ((position - 1) // args.grad_acc) * args.grad_acc
            group_size = min(args.grad_acc, len(order) - group_start)
            scaler.scale(loss / group_size).backward()
            if position % args.grad_acc == 0 or position == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 50 == 0:
                    interim = args.output_dir / f"step-{global_step}"
                    model.save_pretrained(interim)
                    print(f"Saved interim adapter: {interim}", flush=True)
                if global_step % 10 == 0 or global_step == epoch * steps_per_epoch:
                    print(f"Epoch {epoch}/{args.epochs} step {global_step}, train mean loss {loss_sum / position:.4f}", flush=True)

        checkpoint = args.output_dir / f"epoch-{epoch}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        metrics = evaluate(wrapper, model, val_rows, checkpoint / "validation_predictions.json", args.eval_batch_size)
        record = {"epoch": epoch, "optimizer_steps": global_step, "mean_train_loss": loss_sum / len(order), **metrics}
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Epoch {epoch} validation: CER {metrics['cer_percent']:.2f}%, ICU WER {metrics['wer_percent_icu_segmented']:.2f}%", flush=True)
    best = min(history, key=lambda item: item["wer_percent_icu_segmented"])
    (args.output_dir / "best_validation.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Best validation checkpoint: epoch {best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
