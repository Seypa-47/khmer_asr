"""Apply the missing-EOS greedy retry policy to a complete beam validation run.

Completed beam predictions are retained exactly. Only predictions lacking an
end token are decoded again, without consulting their reference transcripts.
This avoids recomputing unchanged branches of the same decoding algorithm.
"""
import argparse
import hashlib
import io
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

from matched_fleurs import load_manifest
from qwen_experiment_utils import score_rows
from qwen_khmer_inference import decode_batch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--primary', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output file to preserve existing evidence')
    result = json.loads(args.primary.read_text(encoding='utf-8'))
    indices = load_manifest('results/matched_fleurs_split.json')['indices']['validation']
    if result['test_used'] or result['validation_indices'] != indices:
        raise ValueError('A complete fixed-validation primary run is required')
    if [r['validation_index'] for r in result['examples']] != indices:
        raise ValueError('Primary predictions are incomplete or in the wrong order')
    config = result['configuration']
    if config['beams'] <= 1:
        raise ValueError('This policy guards beam search')
    if config['adapter'] and hashlib.sha256((Path(config['adapter'])/'adapter_model.safetensors').read_bytes()).hexdigest() != config['adapter_sha256']:
        raise ValueError('Adapter differs from the primary experiment')
    if any(version(name) != value for name, value in config['versions'].items()):
        raise ValueError('Runtime versions differ from the primary experiment')
    retry_rows = [r for r in result['examples'] if not r['has_eos']]
    result['purpose'] = 'Full validation of beam search with greedy retry on missing EOS; test unused'
    result['primary_result'] = {'path': str(args.primary), 'sha256': hashlib.sha256(args.primary.read_bytes()).hexdigest()}
    result['primary_metrics'] = result.pop('metrics')
    config['greedy_retry_on_missing_eos'] = True
    config['retry_batch_size'] = 1
    config['guard_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['decoding_source_sha256'] = hashlib.sha256(Path(__file__).with_name('qwen_khmer_inference.py').read_bytes()).hexdigest()
    for row in result['examples']:
        row['primary_has_eos'] = row['has_eos']
        row['greedy_fallback'] = False
    if retry_rows:
        torch.set_num_threads(4)
        ds = load_dataset('google/fleurs', 'km_kh', split='validation')
        audio = ds.data.column('audio')
        wrapper = Qwen3ASRModel.from_pretrained(config['model'], revision=config['revision'],
                    dtype=getattr(torch, config['dtype']), device_map='cuda:0', max_new_tokens=config['max_new_tokens'])
        if config['adapter']:
            wrapper.model = PeftModel.from_pretrained(wrapper.model, config['adapter'], is_trainable=False)
        wrapper.model.eval()
        for row in retry_rows:
            wav, sr = sf.read(io.BytesIO(audio[row['validation_index']].as_py()['bytes']), dtype='float32')
            if sr != 16000:
                raise ValueError('Unexpected sample rate')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            started = time.perf_counter()
            retry = decode_batch(wrapper, [np.ascontiguousarray(wav)], beams=1,
                                 force_khmer=config['force_khmer'], fallback=False,
                                 max_new_tokens=config['max_new_tokens'])[0]
            result['elapsed_seconds'] += time.perf_counter()-started
            row.update(retry, primary_hypothesis=row['hypothesis'], primary_has_eos=False, greedy_fallback=True)
            print(f'Retried validation row {row["validation_index"]}: EOS={row["has_eos"]}', flush=True)
    result['metrics'] = score_rows(result['examples'])
    result['greedy_retry_count'] = len(retry_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['metrics']), flush=True)


if __name__ == '__main__':
    main()
