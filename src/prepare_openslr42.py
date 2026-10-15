"""Prepare a bounded, auditable public Khmer training supplement in Colab."""
import argparse
import collections
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from matched_fleurs import normalize_text
from score_matched_cer_wer import cer_text

URL = 'https://openslr.elda.org/resources/42/km_kh_male.zip'


def prepare(archive_path, output_dir, exclusions, limit):
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = set(json.loads(exclusions.read_text(encoding='utf-8'))['sha256'])
    counts = collections.Counter()
    rows, seen_text, seen_audio = [], set(), set()
    with zipfile.ZipFile(archive_path) as archive:
        wavs = {}
        for name in archive.namelist():
            if name.lower().endswith('.wav'):
                key = Path(name).stem
                if key in wavs:
                    raise ValueError(f'Ambiguous audio ID: {key}')
                wavs[key] = name
        indexes = [n for n in archive.namelist() if Path(n).name == 'line_index.tsv']
        if len(indexes) != 1:
            raise ValueError(f'Expected one line_index.tsv; found {indexes}')
        lines = archive.read(indexes[0]).decode('utf-8-sig').splitlines()
        random.Random(20261009).shuffle(lines)
        for line in lines:
            if len(rows) >= limit:
                break
            fields = line.split('\t', 1)
            if len(fields) != 2:
                counts['malformed_tsv'] += 1
                continue
            key, text = Path(fields[0]).stem, normalize_text(fields[1])
            text_hash = hashlib.sha256(cer_text(text).encode()).hexdigest()
            if not text or key not in wavs:
                counts['empty_or_missing_audio'] += 1
                continue
            if text_hash in excluded:
                counts['evaluation_transcript_overlap'] += 1
                continue
            if text_hash in seen_text:
                counts['duplicate_transcript'] += 1
                continue
            wav, rate = sf.read(io.BytesIO(archive.read(wavs[key])), dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            seconds = len(wav) / rate
            if not (1 <= seconds <= 30) or not np.isfinite(wav).all() or not np.any(wav):
                counts['invalid_or_duration'] += 1
                continue
            if np.mean(np.abs(wav) >= .999) > .01:
                counts['clipping'] += 1
                continue
            if rate != 16000:
                wav = torchaudio.functional.resample(torch.from_numpy(wav), rate, 16000).numpy()
            # Use generated names; archive member paths are never extracted.
            path = output_dir / f'clip_{len(rows):05d}.wav'
            sf.write(path, wav, 16000, subtype='PCM_16')
            audio_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if audio_hash in seen_audio:
                counts['duplicate_audio'] += 1
                continue
            seen_audio.add(audio_hash); seen_text.add(text_hash)
            rows.append({'audio': str(path.resolve()), 'reference': text, 'source_id': key,
                         'seconds': seconds, 'audio_sha256': audio_hash})
    if not rows:
        raise ValueError('No eligible OpenSLR training examples')
    with archive_path.open('rb') as stream:
        archive_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'source': 'OpenSLR42', 'source_url': URL, 'license': 'CC BY-SA 4.0',
            'archive_sha256': archive_sha256,
            'exclusion_file_sha256': hashlib.sha256(exclusions.read_bytes()).hexdigest(),
            'selection_seed': 20261009, 'max_clips': limit, 'clips': len(rows),
            'hours': sum(r['seconds'] for r in rows)/3600, 'excluded_counts': dict(counts),
            'evaluation_used_for_training': False,
            'limitations': 'Exact transcript hash exclusions and within-source audio deduplication; near duplicates and transcription correctness still require review.',
            'rows': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('/content/openslr42'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=3000)
    ap.add_argument('--exclusions', type=Path, default=Path('results/diagnostic/fleurs_evaluation_text_exclusion_hashes.json'))
    args = ap.parse_args()
    if args.limit < 1:
        ap.error('limit must be positive')
    if not Path('/content').is_dir():
        raise RuntimeError('Run data preparation in Colab')
    args.root.mkdir(parents=True, exist_ok=True)
    archive = args.root/'km_kh_male.zip'
    if not archive.exists():
        tmp = archive.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=120) as response, tmp.open('wb') as out:
            total = 0
            while chunk := response.read(8*1024*1024):
                out.write(chunk); total += len(chunk)
                print(f'OpenSLR download: {total/1e6:.0f} MB', flush=True)
        with zipfile.ZipFile(tmp) as check:
            if check.testzip() is not None:
                raise ValueError('Downloaded archive failed integrity check')
        tmp.replace(archive)
    result = prepare(archive, args.root/'audio', args.exclusions, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
