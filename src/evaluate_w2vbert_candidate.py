"""Screen a pinned Khmer CTC model on cached validation audio only; no training."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2BertForCTC
from qwen_experiment_utils import score_rows

MODEL = 'seanghay/w2v-bert-2.0-khmer'
REVISION = 'e151f2836020b18cfa169bd46dc72f8f6fa33185'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=24)
    args = ap.parse_args()
    if args.limit < 0:
        ap.error('limit must be nonnegative')
    torch.set_num_threads(4)
    source = Path('results/diagnostic/qwen_base_bf16_guarded_validation_20261007.json')
    examples = json.loads(source.read_text(encoding='utf-8'))['examples']
    manifest = json.loads(Path('results/diagnostic/qwen_train30_split_20261007.json').read_text())
    indices = manifest['indices']['validation']
    if args.limit:
        indices = sorted(random.Random(20261007).sample(indices, min(args.limit, len(indices))))
    reference_by_index = {r['validation_index']: r['reference'] for r in examples}
    assert set(indices).issubset(reference_by_index)
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path; preserving existing evidence')
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Wav2Vec2BertForCTC.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    result = {'purpose': 'alternative architecture validation screening; no training',
              'test_used': False, 'model': MODEL, 'revision': REVISION,
              'precision': 'float32', 'decoding': 'greedy CTC',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'validation_indices': indices, 'examples': [],
              'provenance_limit': 'Publisher lists OpenSLR42 training; independently auditing its complete training corpus is not possible here.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in indices:
        wav, sr = sf.read(f'local_eval/qwen_fleurs_audio_expanded/validation_{index}.wav', dtype='float32')
        assert sr == 16000 and wav.ndim == 1
        inputs = processor(wav, sampling_rate=sr, return_tensors='pt').to(device)
        with torch.inference_mode():
            ids = model(**inputs).logits.argmax(dim=-1)
        hypothesis = processor.batch_decode(ids)[0]
        result['examples'].append({'validation_index': index, 'reference': reference_by_index[index],
                                   'hypothesis': hypothesis})
        result['metrics'] = score_rows(result['examples'])
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(args.output)
        print(f"{len(result['examples'])}/{len(indices)} WER {result['metrics']['wer_percent_icu_segmented']:.2f}%", flush=True)


if __name__ == '__main__':
    main()
