#!/usr/bin/env python
"""Save raw app predictions for a directory of user audio files.

This script does not alter predictions or infer reference transcripts.
The local Gradio app must already be running.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
from gradio_client import Client, handle_file


DEFAULT_MODEL = "🏆 Approach 2: Meta MMS-1B Khmer CTC (Version 2.0 · 15 Epochs)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="+", help="Exact filenames to include; otherwise all .m4a files")
    parser.add_argument("--app-url", default="http://127.0.0.1:7860/")
    parser.add_argument("--model-choice", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("local_eval/user_voice_raw_predictions.json"))
    args = parser.parse_args()

    paths = (
        [args.audio_dir / name for name in args.names]
        if args.names else list(args.audio_dir.glob("*.m4a"))
    )
    paths = sorted(paths, key=lambda path: path.name.casefold())
    if not paths:
        raise ValueError(f"No .m4a files found in {args.audio_dir}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio files missing: {missing}")

    client = Client(args.app_url)
    rows = []
    for path in paths:
        with av.open(str(path)) as container:
            duration = container.duration / av.time_base if container.duration else None
            stream = next(stream for stream in container.streams if stream.type == "audio")
            sample_rate = stream.codec_context.sample_rate
        prediction, elapsed, status = client.predict(
            handle_file(str(path)), args.model_choice, api_name="/transcribe_audio"
        )
        rows.append({
            "filename": path.name,
            "duration_seconds": round(duration, 3) if duration is not None else None,
            "source_sample_rate": sample_rate,
            "prediction": prediction,
            "app_elapsed": elapsed,
            "app_status": status,
        })
        print(f"{len(rows)}/{len(paths)} {path.name}: {prediction}", flush=True)

    result = {
        "model_choice": args.model_choice,
        "method": "Unmodified Gradio app output on each complete input file",
        "reference_status": "No filename-to-reference mapping assumed; no CER calculated",
        "files": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
