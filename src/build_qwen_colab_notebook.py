"""Package the current experiment sources into a standalone Colab notebook."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--next-experiment', action='store_true')
parser.add_argument('--public-data', action='store_true')
parser.add_argument('--public-data-target25', action='store_true')
build_args = parser.parse_args()
if build_args.public_data_target25:
    build_args.public_data = True
if build_args.next_experiment and build_args.public_data:
    parser.error('Choose one experiment profile')
paths = [f'src/{name}.py' for name in (
    'train_qwen_candidate', 'qwen_khmer_inference', 'qwen_experiment_utils',
    'score_qwen_rows', 'score_matched_cer_wer', 'matched_fleurs')]
paths += ['results/diagnostic/qwen_train30_split_20261007.json',
          'results/diagnostic/qwen_base_bf16_guarded_validation_20261007.json']
if build_args.public_data:
    paths += ['src/prepare_openslr42.py', 'results/diagnostic/fleurs_evaluation_text_exclusion_hashes.json']
payload = {p: (ROOT/p).read_text(encoding='utf-8') for p in paths}
cells = []


def cell(kind, text):
    data = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        ast.parse(text)
        data.update(execution_count=None, outputs=[])
    cells.append(data)


cell('markdown', '''# Khmer Qwen: Colab improvement experiment
Upload this notebook to Google Colab. Select **Runtime → Change runtime type → GPU**, then **Run all**. Approve the Google Drive mount when prompted.

Target: **CER <20% and ICU Khmer WER <20%** on genuine held-out evaluation. The completed student baseline scored **12.49% CER / 28.81% WER** on 124 validation clips. This new recipe is an experiment, not a promise of reaching 20%.

This run starts from the pinned publisher Khmer Qwen base, with 1,663 official training clips, rank-16 LoRA on q/k/v/o attention projections, learning rate 5e-6, two epochs, and the same 124 validation clips. It tests whether more adapter capacity helps after the smaller adapter plateaued. It does not continue the older 883-clip adapter. Test audio is never loaded.

Training runs only in Colab. Checkpoints, predictions and logs are saved to Drive. T4 uses FP32 for compatibility; L4/A100 use BF16. Resume requires the same GPU type and package/source configuration. Setup downloads the model and training data and may take several minutes.
''')
cell('code', '''from pathlib import Path
import os, sys, subprocess, json, hashlib, shutil
if not Path('/content').is_dir():
    raise RuntimeError('Run this notebook in Google Colab.')
if shutil.which('nvidia-smi') is None:
    raise RuntimeError('No GPU is attached. In Colab choose Runtime > Change runtime type > GPU, reconnect, then rerun this cell.')
gpu_check = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
if gpu_check.returncode or not gpu_check.stdout.strip():
    raise RuntimeError('Colab has no usable NVIDIA GPU. Reconnect to a GPU runtime before installing packages.')
print(gpu_check.stdout.strip())
from google.colab import drive
drive.mount('/content/drive')
WORK = Path('/content/khmer_qwen_colab')
WORK.mkdir(exist_ok=True)
ENV = Path('/content/khmer_qwen_env')
if not (ENV/'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(ENV)], check=True)
PYTHON = str(ENV/'bin/python')
def run_setup(command, cwd=None):
    print('Running:', ' '.join(command), flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, cwd=cwd) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'Setup failed (exit {code}); see the full output above.')

run_setup([PYTHON, '--version'])
run_setup([sys.executable, '-m', 'pip', '--python', PYTHON,
           'install', '--upgrade', 'pip'])
run_setup([PYTHON, '-m', 'pip', 'install', 'torch==2.8.0', 'torchaudio==2.8.0',
    '--index-url', 'https://download.pytorch.org/whl/cu128'])
run_setup([PYTHON, '-m', 'pip', 'install', 'torch==2.8.0', 'torchaudio==2.8.0',
    'qwen-asr==0.0.6', 'transformers==4.57.6', 'peft==0.21.0', 'accelerate==1.12.0',
    'datasets==5.0.1', 'pyicu-wheels==2.15.2', 'jiwer==4.0.0'])
''')
cell('code', 'PAYLOAD = '+repr(payload)+'''
for relative, source in PAYLOAD.items():
    destination = WORK/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding='utf-8')
os.environ.pop('KHMER_SCORE_PYTHON', None)
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('HF_DATASETS_OFFLINE', None)
os.environ['HF_HOME'] = '/content/khmer_hf_cache'
os.environ['PYTHONUNBUFFERED'] = '1'
check = """
import sys, json, torch, icu
sys.path.insert(0, 'src')
from score_qwen_rows import score_rows
assert torch.cuda.is_available(), 'Select a GPU runtime'
assert icu.ICU_VERSION == '77.1', f'Unexpected ICU {icu.ICU_VERSION}'
with open('results/diagnostic/qwen_base_bf16_guarded_validation_20261007.json') as f:
    saved = json.load(f)
actual = score_rows(saved['examples'])
assert actual == saved['metrics'], (actual, saved['metrics'])
print('GPU:', torch.cuda.get_device_name(0))
print('Precision:', 'BF16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'FP32')
print('Scoring regression passed:', actual)
"""
run_setup([PYTHON, '-c', check], cwd=WORK)
''')
cell('markdown', '''## Train and resume
Keep the run name unchanged to resume after a disconnect. If you change the recipe, choose a new run name. Only completed checkpoints are eligible for automatic resume. Each epoch is scored on all 124 validation clips. A completed two-epoch run is not extended automatically.
''')
cell('code', '''RUN_NAME = 'qwen_attention_r16_colab_20261008'
OUT = Path('/content/drive/MyDrive/khmer_asr_runs')/RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
history_path = OUT/'history.json'
history = json.loads(history_path.read_text()) if history_path.exists() else []
done = (OUT/'finished.json').exists() or len(history) >= 2 or any(r['cer_percent'] < 20 and r['wer_percent_icu_segmented'] < 20 for r in history)
command = [PYTHON, 'src/train_qwen_candidate.py',
    '--manifest', 'results/diagnostic/qwen_train30_split_20261007.json',
    '--output-dir', str(OUT), '--cache-dir', '/content/khmer_qwen_audio',
    '--epochs', '2', '--lr', '5e-6', '--rank', '16',
    '--targets', 'q_proj,k_proj,v_proj,o_proj', '--grad-acc', '8',
    '--eval-beams', '3', '--eval-force-khmer', '--eval-batch-size', '1', '--dtype', 'auto']
candidates = []
for marker in OUT.glob('*/complete.json'):
    if (marker.parent/'training_state.pt').exists() and (marker.parent/'adapter_model.safetensors').exists():
        state = json.loads(marker.read_text())
        candidates.append(((state['step'], state['epoch']), marker.parent))
if candidates:
    latest = max(candidates, key=lambda item: item[0])[1]
    command += ['--resume', str(latest)]
    print('Resuming:', latest)
if done:
    print('Run already completed; showing saved results below.')
else:
    with (OUT/'training.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=WORK, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end=''); log.write(line); log.flush()
            code = process.wait()
            if code:
                raise RuntimeError(f'Training failed with exit code {code}; see {OUT}/training.log')
        except KeyboardInterrupt:
            process.terminate(); process.wait()
            raise
''')
cell('code', '''from pathlib import Path
import json, zipfile
from google.colab import drive, files
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/khmer_asr_runs/qwen_attention_r16_colab_20261008')
history_file = OUT/'history.json'
if not history_file.exists():
    raise FileNotFoundError(f'No training history at {history_file}. Check the Drive mount and run folder.')
history = json.loads(history_file.read_text())
if not history:
    raise RuntimeError(f'Training history is empty: {history_file}')
for row in history:
    print(f"Epoch {row['epoch']}: CER {row['cer_percent']:.2f}% | WER {row['wer_percent_icu_segmented']:.2f}%")
best = min(history, key=lambda row: row['wer_percent_icu_segmented'])
checkpoint = OUT/f"epoch-{best['epoch']}"
passed = best['cer_percent'] < 20 and best['wer_percent_icu_segmented'] < 20
print('Selected checkpoint:', checkpoint)
print('Validation target met:', passed)
print('Held-out test: not evaluated. Validation success is not a final test result.')
archive = OUT/'selected_adapter_and_results.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for name in ['adapter_model.safetensors', 'adapter_config.json', 'validation_predictions.json']:
        bundle.write(checkpoint/name, name)
    for name in ['run_config.json', 'history.json', 'best_validation.json', 'training.log']:
        if (OUT/name).exists(): bundle.write(OUT/name, name)
    for path in (OUT/'source').glob('*.py'):
        bundle.write(path, 'source/'+path.name)
print('Saved export:', archive)
files.download(str(archive))
''')
notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU',
    'colab': {'name': 'qwen_improve_colab.ipynb'},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 5}
for i, item in enumerate(cells):
    item['id'] = f'qwen-{i:02d}'
filename = 'qwen_improve_colab.ipynb'
if build_args.next_experiment:
    filename = 'qwen_improve_colab_v2.ipynb'
    replacements = {
        'qwen_attention_r16_colab_20261008': 'qwen_attention_r16_lr5e5_colab_v2_20261008',
        "'--epochs', '2', '--lr', '5e-6'": "'--epochs', '4', '--patience', '2', '--lr', '5e-5'",
        'len(history) >= 2': 'len(history) >= 4',
        'learning rate 5e-6, two epochs': 'learning rate 5e-5, at most four epochs with early stopping after two epochs without improvement',
        'It tests whether more adapter capacity helps after the smaller adapter plateaued.': 'It tests a higher learning rate after the previous rank-16 run reached 28.92% validation WER. This is a new experiment from the pinned base.',
        'A completed two-epoch run is not extended automatically.': 'A completed or early-stopped run is not extended automatically.',
    }
    for item in cells:
        source = ''.join(item['source'])
        # Embedded trainer sources must not be rewritten by recipe substitutions.
        if not source.startswith('PAYLOAD = '):
            for old, new in replacements.items():
                source = source.replace(old, new)
        if item['cell_type'] == 'code':
            ast.parse(source)
        item['source'] = source.splitlines(keepends=True)
notebook['metadata']['colab']['name'] = filename
if build_args.public_data:
    filename = 'qwen_public_data_colab.ipynb'
    notebook['metadata']['colab']['name'] = filename
    cells[0]['source'] = ['''# Khmer Qwen: public-data experiment
Train only in Colab. Choose a GPU and run cells in order.
This experiment adds up to 3,000 OpenSLR42 clips to 1,663 FLEURS training clips.
The actual retained count and hours are measured during preparation. It uses the
same 124 validation clips and the same ICU77.1 scoring. Final test audio is not evaluated.
One-way hashes of all FLEURS validation/test transcripts are used only to exclude
exact external-training overlaps. No evaluation transcripts become training targets.

OpenSLR42 contains manually checked Khmer recordings (errors can remain), under
CC BY-SA4.0: https://www.openslr.org/42/ . Retain source attribution and license.
The publisher Qwen base already used DDD Khmer data; this experiment uses a different
public source. It starts from the pinned base, rank16 attention LoRA, learning rate
1e-5, up to three epochs, patience1. This is an unproven experiment, not a promise
of WER below20%. Previous attention-only student runs remained near29% validation WER.
Checkpoints and the preparation manifest are saved to a separate Drive folder.
T4 uses FP32; native-BF16 GPUs use BF16. Preparation downloads about866MB.
''']
    training = ''.join(cells[4]['source'])
    training = training.replace('qwen_attention_r16_colab_20261008', 'qwen_public_openslr42_20261009')
    training = training.replace("'--epochs', '2', '--lr', '5e-6'", "'--epochs', '3', '--patience', '1', '--lr', '1e-5'")
    training = training.replace('len(history) >= 2', 'len(history) >= 3')
    training = training.replace("candidates = []", "command += ['--extra-training-manifest', str(EXTRA)]\ncandidates = []")
    cells[4]['source'] = training.splitlines(keepends=True)
    cells[3]['source'] = ['## Prepare data, then train\nRun preparation again after a runtime reset to restore the external audio files. Keep the same recipe and GPU type when resuming.\n']
    preparation = '''EXTRA = WORK/'results/openslr42_training.json'
run_setup([PYTHON, 'src/prepare_openslr42.py', '--output', str(EXTRA)], cwd=WORK)
import shutil
DATA_OUT = Path('/content/drive/MyDrive/khmer_asr_runs/qwen_public_openslr42_20261009')
DATA_OUT.mkdir(parents=True, exist_ok=True)
shutil.copyfile(EXTRA, DATA_OUT/'openslr42_training.json')
'''
    cells.insert(4, {'cell_type':'code', 'id':'public-data-preparation', 'metadata':{},
                     'source':preparation.splitlines(keepends=True), 'execution_count':None,'outputs':[]})
    export = ''.join(cells[-1]['source']).replace('qwen_attention_r16_colab_20261008', 'qwen_public_openslr42_20261009').replace("'best_validation.json', 'training.log'", "'best_validation.json', 'training.log', 'openslr42_training.json', 'finished.json'")
    cells[-1]['source'] = export.splitlines(keepends=True)
    for item in cells:
        if item['cell_type'] == 'code':
            ast.parse(''.join(item['source']))
if build_args.public_data_target25:
    filename = 'qwen_public_data_target25_colab.ipynb'
    notebook['metadata']['colab']['name'] = filename
    for item in cells:
        source = ''.join(item['source'])
        if source.startswith('PAYLOAD = '):
            continue
        source = source.replace('qwen_public_openslr42_20261009',
                                'qwen_public_openslr42_target25_20261012')
        source = source.replace("r['wer_percent_icu_segmented'] < 20",
                                "r['wer_percent_icu_segmented'] < 25")
        source = source.replace("best['wer_percent_icu_segmented'] < 20",
                                "best['wer_percent_icu_segmented'] < 25")
        source = source.replace('WER below20%', 'WER below25%')
        if item is cells[0]:
            source = source.replace('# Khmer Qwen: public-data experiment',
                                    '# Khmer Qwen: public-data experiment for WER below 25%')
            source = source.replace('Previous attention-only student runs remained near29% validation WER.',
                                    'The best saved student run scored 12.49% CER and 28.81% WER on 124 validation clips. A strict WER below 25% needs at most 667 word edits across 2,669 reference words, versus 769 in that run. Extra data may help, but success is not guaranteed.')
        if item['cell_type'] == 'code':
            ast.parse(source)
        item['source'] = source.splitlines(keepends=True)
target = ROOT/'notebooks'/filename
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(f'Built {target}: {len(cells)} cells, {target.stat().st_size:,} bytes')
