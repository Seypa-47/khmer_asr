#!/usr/bin/env python
"""Transcribe local recordings with a saved MMS checkpoint, without references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


def decode_mono_16k(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = next(stream for stream in container.streams if stream.type == "audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16_000)
        chunks = []
        for frame in container.decode(stream):
            frame.pts = None
            chunks.extend(sample.to_ndarray() for sample in resampler.resample(frame))
        chunks.extend(sample.to_ndarray() for sample in resampler.resample(None))
    if not chunks:
        raise ValueError(f"No audio decoded from {path}")
    return np.concatenate(chunks, axis=1).squeeze().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="+", help="Exact filenames; default: all .m4a files")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-id", default="facebook/mms-1b-all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = (
        [args.audio_dir / name for name in args.names]
        if args.names else sorted(args.audio_dir.glob("*.m4a"))
    )
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("No selected audio files, or at least one file is missing")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    processor = AutoProcessor.from_pretrained(args.processor_id, target_lang="khm")
    model = Wav2Vec2ForCTC.from_pretrained(
        str(args.model_dir), local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    rows = []
    for position, path in enumerate(paths, 1):
        waveform = decode_mono_16k(path)
        inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        prediction = processor.batch_decode(logits.argmax(-1))[0]
        rows.append({
            "filename": path.name,
            "duration_seconds": round(len(waveform) / 16_000, 3),
            "prediction": prediction,
        })
        print(f"{position}/{len(paths)} {path.name}: {prediction}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({
            "model_choice": "Meta MMS-1B Khmer CTC saved checkpoint",
            "model_checkpoint": str(args.model_dir),
            "method": "Whole recording, greedy CTC decoding, no reference correction",
            "reference_status": "No reference used for inference or saved here",
            "files": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} raw predictions to {args.output}")


if __name__ == "__main__":
    main()
