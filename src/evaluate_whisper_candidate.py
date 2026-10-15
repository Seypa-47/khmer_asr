"""Validation-only Khmer Whisper evaluation using the fixed ICU 77.1 scorer."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from matched_fleurs import load_manifest
from qwen_experiment_utils import score_rows


def save(path, result):
    result['metrics'] = score_rows(result['examples'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    metric = result['metrics']
    print(f"{len(result['examples'])}/{len(result['validation_indices'])}: "
          f"CER {metric['cer_percent']:.2f}%, WER {metric['wer_percent_icu_segmented']:.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, default=Path('results/matched_fleurs_split.json'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        ap.error('batch size and limit must be nonnegative')
    indices = load_manifest(args.manifest)['indices']['validation']
    if args.limit:
        indices = sorted(random.Random(20261007).sample(indices, min(args.limit, len(indices))))
    identity = {'model': args.model, 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'batch_size': args.batch_size, 'dtype': 'float16',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'versions': {n: version(n) for n in ('torch', 'transformers', 'datasets')}}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding='utf-8'))
        if result['configuration'] != identity or result['validation_indices'] != indices:
            raise ValueError('Existing output belongs to a different experiment')
    else:
        result = {'purpose': 'external Whisper candidate validation only; held-out test unused',
                  'test_used': False, 'dataset': 'google/fleurs km_kh validation',
                  'configuration': identity, 'hardware': torch.cuda.get_device_name(0),
                  'validation_indices': indices, 'examples': [], 'elapsed_seconds': 0}
    done = len(result['examples'])
    if [r['validation_index'] for r in result['examples']] != indices[:done]:
        raise ValueError('Saved predictions do not match the validation prefix')
    if done == len(indices):
        save(args.output, result); return
    processor = WhisperProcessor.from_pretrained(args.model, language='Khmer', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to('cuda').eval()
    forced_ids = processor.get_decoder_prompt_ids(language='Khmer', task='transcribe')
    ds = load_dataset('google/fleurs', 'km_kh', split='validation')
    audios, texts = ds.data.column('audio'), ds.data.column('transcription')
    for start in range(done, len(indices), args.batch_size):
        batch_indices = indices[start:start+args.batch_size]
        wavs = []
        for index in batch_indices:
            wav, sr = sf.read(io.BytesIO(audios[index].as_py()['bytes']), dtype='float32')
            if sr != 16000:
                raise ValueError(f'Unexpected sample rate {sr}')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wavs.append(np.ascontiguousarray(wav))
        features = processor.feature_extractor(wavs, sampling_rate=16000, return_tensors='pt', padding=True).input_features.to('cuda', torch.float16)
        started = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(features, forced_decoder_ids=forced_ids)
        result['elapsed_seconds'] += time.perf_counter()-started
        decoded = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        result['examples'].extend({'validation_index': index, 'reference': str(texts[index].as_py()), 'hypothesis': text}
                                  for index, text in zip(batch_indices, decoded))
        save(args.output, result)


if __name__ == '__main__':
    main()
