"""Build a listening review notebook from saved validation error diagnostics."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
audit = json.loads((root/'results/diagnostic/qwen_error_audit_20261009.json').read_text(encoding='utf-8'))
# The high-error sample is for diagnosis, not a representative evaluation subset.
rows = [{k:r[k] for k in ('validation_index','reference','hypothesis','publisher_hypothesis','word_errors')}
        for r in audit['review_queue'][:12]]
cells = []
def add(kind, text):
    cell = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True), 'id': f'review-{len(cells)}'}
    if kind == 'code':
        ast.parse(text)
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

add('markdown', '''# Khmer ASR listening review
This notebook plays 12 validation clips selected for high word-error counts.
It does not train, modify evaluation references, or load test audio.
Use a CPU runtime. These selected difficult clips are not representative of overall accuracy.

For each clip, listen before deciding whether the mismatch is recognition, spelling,
number formatting, an unspoken reference phrase, or uncertain. An English name in
the reference is not by itself proof of an annotation error. Keep all official
evaluation references unchanged. Any training-data cleanup must use training clips
and a general annotation policy, not corrections copied from these validation rows.
''')
add('code', '''import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==5.0.1', 'soundfile', 'ipywidgets'], check=True)
''')
add('code', 'REVIEW_ROWS = '+repr(rows)+'''
import io, json
from pathlib import Path
import soundfile as sf
from datasets import load_dataset
from IPython.display import display, Audio, Markdown
import ipywidgets as widgets

dataset = load_dataset('google/fleurs', 'km_kh', split='validation')
audio_column = dataset.data.column('audio')
transcript_column = dataset.data.column('transcription')
import unicodedata, re
controls = []
for row in REVIEW_ROWS:
    index = row['validation_index']
    original = unicodedata.normalize('NFC', transcript_column[index].as_py()).replace('\\u200b', '').replace('\\ufeff', '')
    original = re.sub(r'\\s+', ' ', original).strip()
    assert original == row['reference'], f'Reference mismatch at row {index}'
    waveform, rate = sf.read(io.BytesIO(audio_column[index].as_py()['bytes']), dtype='float32')
    display(Markdown(f"### Validation {index}: {row['word_errors']} word edits"))
    display(Audio(waveform, rate=rate))
    for label, key in [('Official reference', 'reference'), ('Student', 'hypothesis'), ('Publisher base', 'publisher_hypothesis')]:
        display(widgets.Textarea(value=row[key], description=label, disabled=True, layout=widgets.Layout(width='100%', height='85px')))
    category = widgets.Dropdown(options=['Not reviewed', 'Recognition error', 'Possible reference mismatch', 'Spelling or formatting difference', 'Mixed / uncertain'], description='Assessment')
    notes = widgets.Textarea(description='Listening notes', layout=widgets.Layout(width='100%'))
    display(category, notes)
    controls.append((index, category, notes))
''')
add('code', '''reviews = [{'validation_index': i, 'assessment': category.value, 'notes': notes.value}
           for i, category, notes in controls]
out = Path('/content/khmer_listening_review.json')
out.write_text(json.dumps({'purpose': 'human listening diagnosis only; references unchanged', 'reviews': reviews}, ensure_ascii=False, indent=2), encoding='utf-8')
from google.colab import files
files.download(str(out))
''')
nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':cells}
out = root/'notebooks/qwen_listening_review.ipynb'
out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(out)
