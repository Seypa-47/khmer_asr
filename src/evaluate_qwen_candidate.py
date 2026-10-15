"""Reproducible Qwen validation experiments; never loads held-out test audio.

Supports pinned public models and student adapters with a fixed manifest,
precision and decoding settings. References are used only after generation.
"""
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

from matched_fleurs import load_manifest
from qwen_experiment_utils import score_rows


def write_result(path, result):
    result['metrics'] = score_rows(result['examples'])
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
    m = result['metrics']
    print(f"{len(result['examples'])}/{len(result['validation_indices'])}: "
          f"CER {m['cer_percent']:.2f}%, WER {m['wer_percent_icu_segmented']:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='seanghay/Qwen3-ASR-0.6B-Khmer')
    parser.add_argument('--revision', default='351f1cc594f084e4899ef6cf7bbb17ec64484080')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path('results/matched_fleurs_split.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--beams', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Seed-20261007 subset for decoding pilots; 0 uses all validation rows')
    parser.add_argument('--force-khmer', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.beams < 1 or args.batch_size < 1 or args.limit < 0:
        parser.error('Invalid beam, batch, or subset size')
    torch.set_num_threads(4)
    indices = load_manifest(args.manifest)['indices']['validation']
    if args.limit:
        indices = sorted(random.Random(20261007).sample(indices, min(args.limit, len(indices))))
    identity = {
        'model': args.model, 'revision': args.revision,
        'adapter': str(args.adapter) if args.adapter else None,
        'adapter_sha256': hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() if args.adapter else None,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'dtype': args.dtype, 'beams': args.beams, 'batch_size': args.batch_size,
        'force_khmer': args.force_khmer, 'max_new_tokens': args.max_new_tokens,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {name: version(name) for name in ('torch','transformers','peft','qwen-asr','datasets')},
    }
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding='utf-8'))
        if result['configuration'] != identity or result['validation_indices'] != indices:
            raise ValueError('Existing output belongs to a different experiment')
    else:
        result = {'purpose': 'validation-only candidate selection', 'test_used': False,
                  'dataset': 'google/fleurs km_kh validation', 'configuration': identity,
                  'hardware': torch.cuda.get_device_name(0), 'validation_indices': indices,
                  'examples': [], 'elapsed_seconds': 0}
    done = len(result['examples'])
    if [r['validation_index'] for r in result['examples']] != indices[:done]:
        raise ValueError('Saved predictions do not match the validation prefix')
    if done == len(indices):
        write_result(args.output, result)
        return
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output
    from peft import PeftModel

    ds = load_dataset('google/fleurs', 'km_kh', split='validation')
    audios = ds.data.column('audio')
    references = ds.data.column('transcription')
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype),
        device_map='cuda:0', max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=args.batch_size)
    if args.adapter:
        wrapper.model = PeftModel.from_pretrained(wrapper.model, args.adapter, is_trainable=False)
    wrapper.model.eval()
    # The package's fixed language allow-list omits Khmer although this fine-tune
    # emits Khmer. Construct the publisher's language prefix directly.
    prompt = wrapper._build_text_prompt('', 'Khmer' if args.force_khmer else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for start in range(done, len(indices), args.batch_size):
        batch_indices = indices[start:start + args.batch_size]
        waveforms = []
        for index in batch_indices:
            record = audios[index].as_py()
            wav, sr = sf.read(io.BytesIO(record['bytes']), dtype='float32')
            if sr != 16000:
                raise ValueError(f'Unexpected FLEURS sample rate: {sr}')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            waveforms.append(np.ascontiguousarray(wav))
        inputs = wrapper.processor(text=[prompt] * len(waveforms), audio=waveforms,
                                   return_tensors='pt', padding=True)
        inputs = inputs.to('cuda').to(getattr(torch, args.dtype))
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = wrapper.model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                            num_beams=args.beams, do_sample=False)
        ids = output.sequences[:, inputs['input_ids'].shape[1]:]
        decoded = wrapper.processor.batch_decode(ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
        result['elapsed_seconds'] += time.perf_counter() - t0
        for index, text, tokens in zip(batch_indices, decoded, ids):
            _, hypothesis = parse_asr_output(text, user_language='Khmer' if args.force_khmer else None)
            result['examples'].append({'validation_index': index,
                'reference': str(references[index].as_py()), 'hypothesis': hypothesis,
                'generated_tokens_including_padding': int(tokens.shape[0]),
                'has_eos': bool(torch.isin(tokens, tokens.new_tensor([151645, 151643])).any())})
        write_result(args.output, result)


if __name__ == '__main__':
    main()
