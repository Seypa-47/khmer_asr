"""Build the distinct public-corpus Colab run without changing prior notebooks."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT/'notebooks/qwen_public_data_target25_colab.ipynb').read_text(encoding='utf-8'))
cells = base['cells']
name = 'qwen_km_corpus_bigtrain_20261014'

def source(index):
    return ''.join(cells[index]['source'])

def replace(index, value):
    if cells[index]['cell_type'] == 'code':
        ast.parse(value)
    cells[index]['source'] = value.splitlines(keepends=True)
    if cells[index]['cell_type'] == 'code':
        cells[index]['execution_count'] = None
        cells[index]['outputs'] = []

replace(0, '''# Khmer Qwen: larger public corpus experiment

Train on a Colab GPU using the pinned Khmer Qwen base. The public
[seanghay/km-speech-corpus](https://huggingface.co/datasets/seanghay/km-speech-corpus)
has 14,943 train clips (10.4 hours; CC BY 4.0). This notebook filters audio,
duplicates, and exact FLEURS validation/test transcript overlaps, then trains
on up to 14,000 eligible external clips plus 1,663 FLEURS train clips repeated
twice for balance. Validation uses the unchanged 124 FLEURS clips and ICU 77.1
Khmer word segmentation. Held-out test is untouched.

The last public-data run scored 30.31% WER; the best saved earlier student
scored 28.81% WER. This distinct run uses rank-8 q/v LoRA, learning rate
1e-5, three epochs and patience 2. It selects by full validation WER and
stops on CER <20% and WER <25%. The target is a goal, not a guarantee.
Run cells in order. Training and checkpoints live in Colab and Drive.
Each restart can resume the same run if the GPU type and files match.
The first cell installs packages; data download is about 2.4 GB.
''')

payload_paths = [f'src/{stem}.py' for stem in (
    'train_qwen_candidate', 'prepare_km_speech_corpus', 'qwen_khmer_inference',
    'qwen_experiment_utils', 'score_qwen_rows', 'score_matched_cer_wer',
    'matched_fleurs')]
payload_paths += ['results/diagnostic/qwen_train30_split_20261007.json',
                  'results/diagnostic/qwen_base_bf16_guarded_validation_20261007.json',
                  'results/diagnostic/fleurs_evaluation_text_exclusion_hashes.json']
payload = {path: (ROOT/path).read_text(encoding='utf-8') for path in payload_paths}
old = source(2)
marker = 'for relative, source in PAYLOAD.items():'
assert marker in old
replace(2, 'PAYLOAD = '+repr(payload)+'\n'+old[old.index(marker):])

replace(3, '## Prepare data and train\nThe source files and dataset are pinned. Repeat preparation after a Colab reset so the local audio files exist before resuming.\n')
replace(4, f'''EXTRA = WORK/'results/km_speech_training.json'
run_setup([PYTHON, 'src/prepare_km_speech_corpus.py',
           '--output', str(EXTRA), '--limit', '14000'], cwd=WORK)
import shutil
DATA_OUT = Path('/content/drive/MyDrive/khmer_asr_runs/{name}')
DATA_OUT.mkdir(parents=True, exist_ok=True)
shutil.copyfile(EXTRA, DATA_OUT/'km_speech_training.json')
''')

training = source(5).replace('qwen_public_openslr42_target25_20261012', name)
training = training.replace("'--epochs', '3', '--patience', '1', '--lr', '1e-5'",
                            "'--epochs', '3', '--patience', '2', '--target-wer', '25', '--fleurs-repeat', '2', '--lr', '1e-5'")
training = training.replace("'--rank', '16'", "'--rank', '8'")
training = training.replace("'--targets', 'q_proj,k_proj,v_proj,o_proj'", "'--targets', 'q_proj,v_proj'")
assert "'--rank', '8'" in training and "'--fleurs-repeat', '2'" in training
replace(5, training)

export = source(6).replace('qwen_public_openslr42_target25_20261012', name)
export = export.replace("'openslr42_training.json'", "'km_speech_training.json'")
export = export.replace("files.download(str(archive))", "print('Download the saved ZIP from Drive when training completes:', archive)")
replace(6, export)

base['metadata']['colab']['name'] = 'qwen_km_corpus_bigtrain_colab.ipynb'
for index, cell in enumerate(cells):
    cell['id'] = f'bigtrain-{index:02d}'
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
target = ROOT/'notebooks/qwen_km_corpus_bigtrain_colab.ipynb'
target.write_text(json.dumps(base, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(target, target.stat().st_size)
