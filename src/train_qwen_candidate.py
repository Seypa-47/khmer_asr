"""Train a reproducible student Qwen LoRA candidate, using validation only.

Targets and masking follow QwenLM/Qwen3-ASR's Apache-2.0 fine-tuning example:
https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning/qwen3_asr_sft.py
Checkpoints retain optimizer, scheduler, and RNG state for continuation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import time
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from qwen_asr import Qwen3ASRModel
from transformers import get_cosine_schedule_with_warmup

from matched_fleurs import load_manifest, normalize_text
from qwen_experiment_utils import score_rows
from qwen_khmer_inference import decode_batch

MODEL_ID = 'seanghay/Qwen3-ASR-0.6B-Khmer'
REVISION = '351f1cc594f084e4899ef6cf7bbb17ec64484080'


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def load_rows(split, indices, cache):
    ds = load_dataset('google/fleurs', 'km_kh', split=split)
    audio = ds.data.column('audio')
    references = ds.data.column('transcription')
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        path = cache / f'{split}_{index}.wav'
        if not path.exists():
            wav, sr = sf.read(io.BytesIO(audio[index].as_py()['bytes']), dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000 or not np.isfinite(wav).all():
                raise ValueError('Invalid training audio')
            sf.write(path, wav, sr, subtype='PCM_16')
        rows.append({'index': index, 'audio': str(path), 'reference': normalize_text(references[index].as_py())})
    return rows


def evaluate(wrapper, rows, output, batch_size, beams, force_khmer):
    wrapper.model.eval()
    result = {'purpose': 'student candidate selection; validation only',
              'split': 'google/fleurs km_kh validation', 'test_used': False,
              'decoding': {'beams': beams, 'force_khmer': force_khmer, 'greedy_retry_on_missing_eos': True}, 'examples': []}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        wavs = [sf.read(r['audio'], dtype='float32')[0] for r in batch]
        predictions = decode_batch(wrapper, wavs, beams=beams, force_khmer=force_khmer)
        result['examples'].extend({'validation_index': r['index'], 'reference': r['reference'],
                                   **p} for r, p in zip(batch, predictions))
        atomic_json(output, result)
        print(f'Validation {len(result["examples"])}/{len(rows)}', flush=True)
    result['metrics'] = score_rows(result['examples'])
    atomic_json(output, result)
    return result['metrics']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=Path('results/diagnostic/qwen_train30_split_20261007.json'))
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, default=Path('local_eval/qwen_fleurs_audio_expanded'))
    ap.add_argument('--extra-training-manifest', type=Path, help='Audited external training rows; never evaluation examples')
    ap.add_argument('--fleurs-repeat', type=int, default=1, help='Repeat official FLEURS training rows to balance external speech')
    ap.add_argument('--target-wer', type=float, default=20, help='Validation WER threshold for early success')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--patience', type=int, default=0, help='Stop after this many epochs without a lower validation WER; zero disables')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--grad-acc', type=int, default=8)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--targets', default='q_proj,v_proj',
                    help='Comma-separated linear projection names for LoRA')
    ap.add_argument('--eval-batch-size', type=int, default=4)
    ap.add_argument('--dtype', choices=['auto', 'bfloat16', 'float32'], default='auto',
                    help='Auto uses BF16 on supported GPUs and FP32 on Colab T4')
    ap.add_argument('--eval-beams', type=int, default=1)
    ap.add_argument('--eval-force-khmer', action='store_true')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--max-steps', type=int, default=0, help='Stop after saving a training smoke check; no validation selection')
    args = ap.parse_args()
    if args.patience < 0:
        ap.error('Patience cannot be negative')
    if min(args.epochs, args.grad_acc, args.rank, args.eval_batch_size, args.eval_beams, args.fleurs_repeat) < 1 or args.lr <= 0 or not 0 < args.target_wer < 100:
        ap.error('Training and decoding settings must be positive')
    targets = [name.strip() for name in args.targets.split(',') if name.strip()]
    if not targets:
        ap.error('At least one LoRA target is required')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab')
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    dtype_name = ('bfloat16' if native_bf16 else 'float32') if args.dtype == 'auto' else args.dtype
    if dtype_name == 'bfloat16' and not native_bf16:
        raise RuntimeError('This GPU does not support BF16; use --dtype float32')
    dtype = getattr(torch, dtype_name)
    manifest = load_manifest(args.manifest)
    extra_rows = []
    if args.extra_training_manifest:
        extra = json.loads(args.extra_training_manifest.read_text(encoding='utf-8'))
        if extra.get('source') not in {'OpenSLR42', 'seanghay/km-speech-corpus'} or extra.get('evaluation_used_for_training') is not False:
            raise ValueError('Expected audited external training supplement')
        for row in extra['rows']:
            audio_path = Path(row['audio'])
            if hashlib.sha256(audio_path.read_bytes()).hexdigest() != row['audio_sha256']:
                raise ValueError(f'External audio changed: {audio_path}')
            extra_rows.append({'index': extra['source']+':'+str(row['source_id']),
                               'audio': str(audio_path), 'reference': row['reference']})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_names = ['train_qwen_candidate.py', 'qwen_khmer_inference.py', 'qwen_experiment_utils.py',
                    'score_qwen_rows.py', 'score_matched_cer_wer.py', 'matched_fleurs.py']
    if args.extra_training_manifest:
        preparation_source = 'prepare_openslr42.py' if extra['source'] == 'OpenSLR42' else 'prepare_km_speech_corpus.py'
        source_names.append(preparation_source)
    source_bytes = {name: Path(__file__).with_name(name).read_bytes() for name in source_names}
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    config = {'base_model': MODEL_ID, 'base_revision': REVISION, 'manifest': str(args.manifest),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'train_examples': len(manifest['indices']['train'])*args.fleurs_repeat + len(extra_rows),
              'fleurs_repeat': args.fleurs_repeat, 'external_source': extra['source'] if args.extra_training_manifest else None,
              'validation_examples': len(manifest['indices']['validation']),
              'extra_training_manifest_sha256': hashlib.sha256(args.extra_training_manifest.read_bytes()).hexdigest() if args.extra_training_manifest else None,
              'test_used': False, 'epochs': args.epochs, 'patience': args.patience, 'target_wer': args.target_wer, 'learning_rate': args.lr,
              'gradient_accumulation': args.grad_acc, 'dtype': dtype_name, 'seed': 42,
              'lora': {'r': args.rank, 'alpha': args.rank*2, 'dropout': .05, 'targets': targets},
              'schedule': 'cosine, 5% warmup', 'weight_decay': .01,
              'selection': 'minimum full validation ICU 77.1 Khmer WER',
              'max_new_tokens': 512, 'hardware': torch.cuda.get_device_name(0),
              'validation_decoding': {'beams': args.eval_beams, 'force_khmer': args.eval_force_khmer,
                                     'greedy_retry_on_missing_eos': True},
              'decoding_source_sha256': hashlib.sha256(Path(__file__).with_name('qwen_khmer_inference.py').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_files_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()},
              'versions': {name: version(name) for name in ('torch','transformers','peft','qwen-asr','datasets')}}
    config_path = args.output_dir / 'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Existing output belongs to a different experiment configuration')
        if not args.resume and list(args.output_dir.glob('*/training_state.pt')):
            raise ValueError('Existing output requires matching configuration and explicit --resume')
    else:
        atomic_json(config_path, config)
    snapshot = args.output_dir / 'source'
    snapshot.mkdir(exist_ok=True)
    for name, data in source_bytes.items():
        saved = snapshot / name
        if saved.exists() and saved.read_bytes() != data:
            raise ValueError(f'Saved source differs: {name}')
        if not saved.exists():
            saved.write_bytes(data)
    fleurs_train = load_rows('train', manifest['indices']['train'], args.cache_dir)
    train = fleurs_train * args.fleurs_repeat
    train.extend(extra_rows)
    validation = load_rows('validation', manifest['indices']['validation'], args.cache_dir)
    wrapper = Qwen3ASRModel.from_pretrained(MODEL_ID, revision=REVISION, dtype=dtype,
                device_map='cuda:0', max_inference_batch_size=args.eval_batch_size, max_new_tokens=512)
    base = wrapper.model
    def forward(self, **kwargs):
        return self.thinker(**kwargs)
    base.forward = types.MethodType(forward, base)
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=args.rank*2, lora_dropout=.05,
                target_modules=targets, bias='none'))
    # Configure the actual decoder rather than the outer class, which has no
    # get_input_embeddings implementation. Non-reentrant checkpointing works
    # with frozen embeddings and trainable LoRA layers.
    base.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.thinker.audio_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    wrapper.model = model
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable parameters: {sum(p.numel() for p in params):,}', flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    total_steps = math.ceil(len(train)/args.grad_acc)*args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps*.05), total_steps)
    prefix = wrapper._build_text_prompt('', None)
    eos = wrapper.processor.tokenizer.eos_token
    history = []
    start_epoch, start_position, step = 1, 0, 0
    resume_order = None
    resume_loss, resume_count = 0., 0
    if args.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(args.resume / 'adapter_model.safetensors')))
        state = torch.load(args.resume / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['run_config'] != config:
            raise ValueError('Resume checkpoint does not match this experiment configuration')
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch, start_position, step = state['epoch'], state['position'], state['step']
        resume_order = state['order']; history = state['history']
        resume_loss, resume_count = state['epoch_loss_sum'], state['epoch_loss_count']
    if (args.output_dir / 'finished.json').exists():
        print('Experiment already finished; use a new output directory for a new recipe.', flush=True)
        return
    def checkpoint(path, epoch, position, order, loss_sum, loss_count):
        path.mkdir(parents=True, exist_ok=True)
        marker = path / 'complete.json'
        marker.unlink(missing_ok=True)
        model.save_pretrained(path)
        state = {'run_config': config, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'epoch': epoch, 'position': position, 'step': step, 'order': order, 'history': history,
                 'epoch_loss_sum': loss_sum, 'epoch_loss_count': loss_count}
        torch.save(state, path / 'training_state.pt')
        atomic_json(marker, {'step': step, 'epoch': epoch, 'position': position})
        print(f'Saved {path}', flush=True)
    t0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs+1):
        order = resume_order if resume_order is not None else random.sample(range(len(train)), len(train))
        resume_order = None
        model.train(); optimizer.zero_grad(set_to_none=True)
        running_loss, count = resume_loss, resume_count
        resume_loss, resume_count = 0., 0
        for position in range(start_position, len(order)):
            row = train[order[position]]
            wav, sr = sf.read(row['audio'], dtype='float32')
            target = 'language Khmer<asr_text>' + row['reference']
            full = wrapper.processor(text=[prefix+target+eos], audio=[wav], return_tensors='pt', padding=True)
            prompt = wrapper.processor(text=[prefix], audio=[wav], return_tensors='pt', padding=True)
            labels = full['input_ids'].clone()
            labels[:, :int(prompt['attention_mask'].sum())] = -100
            labels[full['attention_mask'] == 0] = -100
            full['labels'] = labels
            full = full.to('cuda').to(dtype)
            # Confirm BatchFeature.to(dtype) preserved integer IDs and labels.
            if full['labels'].dtype != torch.int64:
                raise RuntimeError('Labels must remain integer token IDs')
            output = model(**full, use_cache=False)
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at row {row["index"]}')
            running_loss += float(loss.detach()); count += 1
            group_start = position // args.grad_acc * args.grad_acc
            group_size = min(args.grad_acc, len(order)-group_start)
            (loss/group_size).backward()
            del full, output, loss, prompt, labels
            if (position+1) % args.grad_acc == 0 or position+1 == len(order):
                gradient_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                if gradient_norm <= 0:
                    raise RuntimeError('No trainable gradient reached the adapter')
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if step % 5 == 0 or step == 1:
                    print(f'Epoch {epoch}/{args.epochs} step {step}/{total_steps} clips {position+1}/{len(order)} '
                          f'loss {running_loss/count:.4f} lr {scheduler.get_last_lr()[0]:.7f} '
                          f'grad {float(gradient_norm):.4f} elapsed {time.perf_counter()-t0:.0f}s '
                          f'GPUpeak {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)
                if step % 50 == 0 or (args.max_steps and step >= args.max_steps):
                    checkpoint(args.output_dir / f'step-{step}', epoch, position+1, order, running_loss, count)
                if args.max_steps and step >= args.max_steps:
                    return
        start_position = 0
        path = args.output_dir / f'epoch-{epoch}'
        checkpoint(path, epoch, len(order), order, running_loss, count)
        torch.cuda.empty_cache()
        metrics = evaluate(wrapper, validation, path / 'validation_predictions.json', args.eval_batch_size,
                           args.eval_beams, args.eval_force_khmer)
        history.append({'epoch': epoch, 'optimizer_steps': step, 'mean_train_loss': running_loss/max(count,1), **metrics})
        atomic_json(args.output_dir / 'history.json', history)
        best = min(history, key=lambda r:r['wer_percent_icu_segmented'])
        atomic_json(args.output_dir / 'best_validation.json', best)
        checkpoint(path, epoch+1, 0, None, 0., 0)
        print(f'Epoch {epoch}: CER {metrics["cer_percent"]:.2f}%, WER {metrics["wer_percent_icu_segmented"]:.2f}%', flush=True)
        if metrics['cer_percent'] < 20 and metrics['wer_percent_icu_segmented'] < args.target_wer:
            atomic_json(args.output_dir / 'finished.json', {'reason': 'validation_target', 'epoch': epoch})
            print('Validation thresholds met. Hold out test pending fixed checkpoint selection.', flush=True)
            break
        if args.patience and epoch - best['epoch'] >= args.patience:
            atomic_json(args.output_dir / 'finished.json', {'reason': 'early_stopping', 'epoch': epoch})
            print('Stopping: validation WER did not improve. Keeping the best epoch.', flush=True)
            break
    else:
        atomic_json(args.output_dir / 'finished.json', {'reason': 'epoch_limit', 'epoch': args.epochs})


if __name__ == '__main__':
    main()
