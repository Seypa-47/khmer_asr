#!/usr/bin/env python
"""
Test the fine-tuned Khmer Whisper model.

Usage:
  1. Default test (runs on a real cached Khmer speech sample):
     .\\.venv\\Scripts\\python.exe test_inference.py

  2. Test on your own local audio file (.wav, .mp3, .flac):
     .\\.venv\\Scripts\\python.exe test_inference.py --audio "my_audio.wav"
"""

import argparse
import io
import glob
import os
import re
import sys
import unicodedata
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Configure UTF-8 stdout so Windows console can display Khmer characters properly
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "whisper-tiny-khmer")
if not os.path.exists(DEFAULT_MODEL_PATH):
    DEFAULT_MODEL_PATH = r"D:\khmer_asr\khmer_asr\content\whisper-tiny-khmer"
DDD_PARQUET_GLOB = r"D:\hf_cache\hub\datasets--DDD-Cambodia--khmer-speech-dataset\snapshots\*\data\train-*.parquet"


def normalize_khmer_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transcribe_audio_array(model, processor, audio_array, sampling_rate=16000, device="cpu"):
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="khmer",
            task="transcribe",
            forced_decoder_ids=None
        )

    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return normalize_khmer_text(transcription)


def test_with_cached_khmer_sample(model, processor, device, sample_index=0):
    parquet_files = glob.glob(DDD_PARQUET_GLOB)
    if not parquet_files:
        print("No cached Khmer dataset found in D:\\hf_cache.")
        print("Please test with a local audio file: python test_inference.py --audio file.wav")
        return

    import pyarrow.parquet as pq

    print(f"\n--- Loading sample #{sample_index} from cached Khmer dataset ---")
    table = pq.read_table(parquet_files[0])
    row = table.to_pylist()[sample_index % len(table)]

    audio_bytes = row["audio"]["bytes"]
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    ground_truth = normalize_khmer_text(row.get("transcript", ""))

    print(f"Audio Duration: {round(len(audio_array) / sr, 2)} seconds")
    print("Transcribing with fine-tuned Whisper...")
    prediction = transcribe_audio_array(model, processor, audio_array, sampling_rate=sr, device=device)

    print("\n" + "=" * 60)
    print("GROUND TRUTH (Original Speech Transcript):")
    print(ground_truth)
    print("-" * 60)
    print("MODEL PREDICTION (Your Fine-Tuned Whisper):")
    print(prediction)
    print("=" * 60)


def test_with_local_file(model, processor, device, audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return

    import librosa
    print(f"\nLoading audio file: {audio_path}")
    audio_array, sr = librosa.load(audio_path, sr=16000)

    print(f"Audio Duration: {round(len(audio_array) / sr, 2)} seconds")
    print("Transcribing with fine-tuned Whisper...")
    prediction = transcribe_audio_array(model, processor, audio_array, sampling_rate=sr, device=device)

    print("\n" + "=" * 60)
    print(f"AUDIO FILE: {os.path.basename(audio_path)}")
    print("MODEL PREDICTION (Your Fine-Tuned Whisper):")
    print(prediction)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Fine-tuned Khmer Whisper Model")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained model directory")
    parser.add_argument("--audio", default=None, help="Path to a local audio file (.wav, .mp3, .flac)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to test (0 to 399)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {args.model_path}")
    print(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(args.model_path, language="Khmer", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()

    if args.audio:
        test_with_local_file(model, processor, device, args.audio)
    else:
        test_with_cached_khmer_sample(model, processor, device, sample_index=args.sample_idx)


if __name__ == "__main__":
    main()
