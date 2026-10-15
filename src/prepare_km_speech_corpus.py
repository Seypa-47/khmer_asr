"""Prepare an audited, bounded supplement from seanghay/km-speech-corpus.

Only the publisher's train split is read. Validation/test transcript hashes are
used solely to reject exact overlaps; their audio and text are never training rows.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset

from matched_fleurs import normalize_text
from score_matched_cer_wer import cer_text


SOURCE = 'seanghay/km-speech-corpus'
REVISION = '88709b3a5fffb4323cc560a49688ad67a15405c2'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--audio-dir', type=Path, default=Path('/content/km_speech_audio'))
    ap.add_argument('--exclusions', type=Path, default=Path('results/diagnostic/fleurs_evaluation_text_exclusion_hashes.json'))
    ap.add_argument('--limit', type=int, default=14000)
    args = ap.parse_args()
    if args.limit < 1:
        ap.error('limit must be positive')
    if not Path('/content').is_dir():
        raise RuntimeError('Run preparation in Colab')

    excluded = set(json.loads(args.exclusions.read_text(encoding='utf-8'))['sha256'])
    ds = load_dataset(SOURCE, split='train', revision=REVISION)
    ds = ds.cast_column('audio', Audio(decode=False))
    audio_column = ds.data.column('audio')
    text_column = ds.data.column('transcription')
    order = list(range(len(ds)))
    random.Random(20261014).shuffle(order)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    rows, seen_text, seen_audio = [], set(), set()
    counts = collections.Counter()

    for index in order:
        if len(rows) >= args.limit:
            break
        text = normalize_text(text_column[index].as_py())
        if not text or '[' in text or ']' in text or len(text) > 180:
            counts['empty_annotation_or_long_text'] += 1
            continue
        # Reject rows with too little Khmer content while retaining mixed
        # Khmer/Latin speech and numbers, which matter in this validation set.
        letters = [ch for ch in text if unicodedata.category(ch).startswith('L')]
        khmer = sum('\u1780' <= ch <= '\u17ff' for ch in letters)
        if len(letters) < 3 or khmer / len(letters) < .5:
            counts['little_khmer_text'] += 1
            continue
        text_hash = hashlib.sha256(cer_text(text).encode('utf-8')).hexdigest()
        if text_hash in excluded:
            counts['evaluation_transcript_overlap'] += 1
            continue
        if text_hash in seen_text:
            counts['duplicate_transcript'] += 1
            continue
        record = audio_column[index].as_py()
        try:
            if record['bytes'] is not None:
                wav, rate = sf.read(io.BytesIO(record['bytes']), dtype='float32')
            else:
                wav, rate = sf.read(record['path'], dtype='float32')
        except (RuntimeError, ValueError, OSError, TypeError) as error:
            counts['unreadable_audio'] += 1
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        seconds = len(wav) / rate
        if not 1.0 <= seconds <= 20.0 or not np.isfinite(wav).all() or not np.any(wav):
            counts['invalid_or_duration'] += 1
            continue
        if np.mean(np.abs(wav) >= .999) > .01:
            counts['clipping'] += 1
            continue
        if rate != 16000:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), rate, 16000).numpy()
        path = args.audio_dir / f'clip_{index:05d}.wav'
        sf.write(path, wav, 16000, subtype='PCM_16')
        audio_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if audio_hash in seen_audio:
            path.unlink()
            counts['duplicate_audio'] += 1
            continue
        seen_text.add(text_hash)
        seen_audio.add(audio_hash)
        rows.append({'audio': str(path.resolve()), 'reference': text,
                     'source_id': str(index), 'seconds': seconds,
                     'audio_sha256': audio_hash})
        if len(rows) % 1000 == 0:
            print(f'Prepared {len(rows)} clips / {len(ds)} inspected', flush=True)

    if not rows:
        raise RuntimeError('No eligible public training audio found')
    result = {'source': SOURCE, 'source_revision': REVISION,
              'source_url': 'https://huggingface.co/datasets/' + SOURCE,
              'license': 'CC BY 4.0', 'split': 'train',
              'exclusion_file_sha256': hashlib.sha256(args.exclusions.read_bytes()).hexdigest(),
              'selection_seed': 20261014, 'max_clips': args.limit,
              'clips': len(rows), 'hours': sum(r['seconds'] for r in rows)/3600,
              'excluded_counts': dict(counts), 'evaluation_used_for_training': False,
              'limitations': 'Exact transcript overlap and duplicate-audio checks; near duplicates and label correctness are not fully verified.',
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
