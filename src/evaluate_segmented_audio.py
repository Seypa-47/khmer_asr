#!/usr/bin/env python
"""Experiment with general, pause-aware audio segmentation before MMS inference.

This experiment uses audio only to choose cut points. It never sees reference
transcripts and never substitutes expected words into the output.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from gradio_client import Client, handle_file

from evaluate_user_audio import DEFAULT_MODEL


def decode_mono_16k(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = next(stream for stream in container.streams if stream.type == "audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16_000)
        frames = []
        for frame in container.decode(stream):
            frame.pts = None
            for resampled in resampler.resample(frame):
                frames.append(resampled.to_ndarray())
    return np.concatenate(frames, axis=1).squeeze().astype(np.float32)


def pause_aware_bounds(audio: np.ndarray, sample_rate: int = 16_000) -> list[tuple[int, int]]:
    """Choose low-energy 200 ms windows near 8-second targets, max ~9.5 sec."""
    end = len(audio)
    start = 0
    bounds = []
    while end - start > int(9.5 * sample_rate):
        earliest = start + int(6.5 * sample_rate)
        latest = min(start + int(9.3 * sample_rate), end - int(1.0 * sample_rate))
        window = int(0.2 * sample_rate)
        candidates = range(earliest, latest + 1, int(0.05 * sample_rate))
        cut = min(candidates, key=lambda point: float(np.mean(audio[point-window//2:point+window//2] ** 2)))
        bounds.append((start, cut))
        start = cut
    bounds.append((start, end))
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="+", help="Exact filenames to include; otherwise all .m4a files")
    parser.add_argument("--app-url", default="http://127.0.0.1:7860/")
    parser.add_argument("--model-choice", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("local_eval/user_voice_segmented_predictions.json"))
    args = parser.parse_args()

    paths = [args.audio_dir / name for name in args.names] if args.names else list(args.audio_dir.glob("*.m4a"))
    paths = sorted(paths, key=lambda path: path.name.casefold())
    if not paths:
        raise ValueError("No .m4a files selected")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio files missing: {missing}")

    client = Client(args.app_url)
    rows = []
    with tempfile.TemporaryDirectory(prefix="khmer_asr_segments_") as temporary:
        temp_dir = Path(temporary)
        for path in paths:
            audio = decode_mono_16k(path)
            bounds = pause_aware_bounds(audio)
            segments = []
            for index, (start, end) in enumerate(bounds):
                segment_path = temp_dir / f"{len(rows):02d}_{index:02d}.wav"
                sf.write(segment_path, audio[start:end], 16_000)
                text, elapsed, _ = client.predict(
                    handle_file(str(segment_path)), args.model_choice, api_name="/transcribe_audio"
                )
                segments.append({
                    "start_seconds": round(start / 16_000, 3),
                    "end_seconds": round(end / 16_000, 3),
                    "prediction": text,
                    "app_elapsed": elapsed,
                })
            joined = "".join(segment["prediction"] for segment in segments)
            rows.append({
                "filename": path.name,
                "duration_seconds": round(len(audio) / 16_000, 3),
                "prediction": joined,
                "segments": segments,
            })
            print(f"{len(rows)}/{len(paths)} {path.name}: {len(segments)} segments -> {joined}", flush=True)

    result = {
        "model_choice": args.model_choice,
        "method": "Audio-only low-energy cuts near 8 s (6.5-9.3 s search); each segment through unmodified app; concatenate predictions",
        "reference_status": "No references used to select segment boundaries or alter predictions",
        "files": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
