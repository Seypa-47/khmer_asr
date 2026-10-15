"""Audit official FLEURS training and fixed validation data without test access."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

from matched_fleurs import load_manifest, normalize_text


def main():
    manifest = load_manifest('results/diagnostic/qwen_expanded_train_split.json')
    datasets = {s: load_dataset('google/fleurs', 'km_kh', split=s) for s in ('train', 'validation')}
    records = {}
    for split, ds in datasets.items():
        rows = []
        indices = range(len(ds)) if split == 'train' else manifest['indices']['validation']
        for index in indices:
            raw = ds.data.column('audio')[index].as_py()['bytes']
            wav, sr = sf.read(io.BytesIO(raw), dtype='float32')
            text = normalize_text(ds.data.column('transcription')[index].as_py())
            duration = len(wav) / sr
            rows.append({'index': index, 'seconds': duration, 'sample_rate': sr,
                         'channels': 1 if wav.ndim == 1 else wav.shape[1],
                         'audio_sha256': hashlib.sha256(raw).hexdigest(),
                         'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'characters': len(text.replace(' ', '')),
                         'characters_per_second': len(text.replace(' ', '')) / max(duration, .001),
                         'rms': float(np.sqrt(np.mean(wav**2))),
                         'clipped_fraction': float(np.mean(np.abs(wav) >= .999)),
                         'all_finite': bool(np.isfinite(wav).all()),
                         'existing_training_selection': index in manifest['indices']['train'] if split == 'train' else False})
        records[split] = rows
    train, val = records['train'], records['validation']
    summary = {}
    for split, rows in records.items():
        summary[split] = {'clips': len(rows), 'hours': sum(r['seconds'] for r in rows) / 3600,
            'duration_quantiles_seconds': dict(zip(['min','median','p95','max'],
                                                 np.quantile([r['seconds'] for r in rows], [0,.5,.95,1]).tolist())),
            'empty_or_invalid_indices': [r['index'] for r in rows if not r['all_finite'] or r['rms'] == 0 or not r['characters']],
            'clipped_over_one_percent_indices': [r['index'] for r in rows if r['clipped_fraction'] > .01],
            'unique_audio': len({r['audio_sha256'] for r in rows}),
            'unique_transcripts': len({r['text_sha256'] for r in rows}),
            'transcript_rate_over_35_indices': [r['index'] for r in rows if r['characters_per_second'] > 35]}
    for field in ('audio_sha256', 'text_sha256'):
        overlap = {r[field] for r in train} & {r[field] for r in val}
        summary['train_validation_overlap_' + field] = len(overlap)
    summary['train_eligible_under_30_seconds'] = sum(r['seconds'] <= 30 and r['characters'] > 0 and r['rms'] > 0 for r in train)
    result = {'purpose': 'data audit; no test audio or references loaded', 'test_used': False,
              'summary': summary, 'records': records,
              'limitation': 'Signal statistics cannot verify that each spoken word matches its transcript.'}
    output = Path('results/diagnostic/qwen_data_audit_20261007.json')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
