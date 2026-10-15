"""Run the validation-selected big-corpus Qwen adapter on local audio or held-out FLEURS."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

from matched_fleurs import load_manifest, normalize_text
from qwen_khmer_inference import decode_batch, transcribe_khmer
from score_qwen_rows import score_rows


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models/qwen_km_corpus_bigtrain_20261014_epoch1"
MANIFEST = ROOT / "results/diagnostic/qwen_train30_split_20261007.json"
MODEL_ID = "seanghay/Qwen3-ASR-0.6B-Khmer"
REVISION = "351f1cc594f084e4899ef6cf7bbb17ec64484080"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_wrapper(dtype: torch.dtype):
    wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype, device_map="cuda:0",
        max_inference_batch_size=1, max_new_tokens=512,
    )
    wrapper.model = PeftModel.from_pretrained(wrapper.model, str(ADAPTER), is_trainable=False)
    wrapper.model.eval()
    return wrapper


def check_adapter() -> dict:
    config = json.loads((ADAPTER / "run_config.json").read_text(encoding="utf-8"))
    best = json.loads((ADAPTER / "best_validation.json").read_text(encoding="utf-8"))
    if best["epoch"] != 1 or config["train_examples"] != 16720:
        raise ValueError("This is not the selected big-corpus epoch-1 adapter")
    # The Windows checkout uses CRLF; Colab saved this identical manifest with LF.
    manifest_bytes = MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(manifest_bytes).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Local split manifest differs from the training manifest")
    return config


def run_samples(wrapper, output: Path, dtype_name: str) -> None:
    result = {
        "purpose": "local transcription smoke test; not a held-out score",
        "adapter": str(ADAPTER), "dtype": dtype_name, "examples": [],
    }
    for path in sorted((ROOT / "samples").glob("*.wav")):
        waveform, sample_rate = sf.read(path, dtype="float32")
        started = time.perf_counter()
        hypothesis = transcribe_khmer(wrapper, waveform, sample_rate, beams=3)
        result["examples"].append({
            "audio": str(path), "hypothesis": hypothesis,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        })
        print(f"{path.name}: {hypothesis}", flush=True)
        save_json(output, result)
    print(f"Saved {output}", flush=True)


def run_heldout(wrapper, output: Path, dtype_name: str) -> None:
    indices = load_manifest(MANIFEST)["indices"]["test"]
    if len(indices) != 114:
        raise ValueError("Expected the fixed 114-clip held-out manifest")
    identity = {
        "model": MODEL_ID, "revision": REVISION, "checkpoint": "epoch-1",
        "adapter_sha256": hashlib.sha256((ADAPTER / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "test_indices": indices, "dtype": dtype_name,
        "decoding": {"beams": 3, "force_khmer": True, "greedy_retry_on_missing_eos": True},
    }
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["configuration"] != identity:
            raise ValueError("Existing held-out output has a different configuration")
    else:
        result = {
            "purpose": "one-time held-out evaluation of validation-selected epoch-1",
            "dataset": "google/fleurs km_kh test", "configuration": identity,
            "examples": [],
        }
    done = len(result["examples"])
    if [row["test_index"] for row in result["examples"]] != indices[:done]:
        raise ValueError("Saved predictions do not match the held-out manifest prefix")
    if done < len(indices):
        dataset = load_dataset("google/fleurs", "km_kh", split="test")
        audio_column = dataset.data.column("audio")
        reference_column = dataset.data.column("transcription")
        for index in indices[done:]:
            waveform, sample_rate = sf.read(io.BytesIO(audio_column[index].as_py()["bytes"]), dtype="float32")
            if sample_rate != 16000:
                raise ValueError(f"Unexpected FLEURS sample rate: {sample_rate}")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)
            prediction = decode_batch(wrapper, [waveform], beams=3, force_khmer=True)[0]
            result["examples"].append({
                "test_index": index,
                "reference": normalize_text(reference_column[index].as_py()),
                **prediction,
            })
            save_json(output, result)
            print(f"Held-out {len(result['examples'])}/{len(indices)}", flush=True)
    result["metrics"] = score_rows(result["examples"])
    save_json(output, result)
    metrics = result["metrics"]
    print(f"Held-out CER {metrics['cer_percent']:.2f}% | ICU WER {metrics['wer_percent_icu_segmented']:.2f}%", flush=True)
    print(f"Saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["samples", "heldout"])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_adapter()
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU is required")
    torch.set_num_threads(4)
    dtype = getattr(torch, args.dtype)
    wrapper = load_wrapper(dtype)
    if args.mode == "samples":
        run_samples(wrapper, args.output, args.dtype)
    else:
        run_heldout(wrapper, args.output, args.dtype)


if __name__ == "__main__":
    main()
