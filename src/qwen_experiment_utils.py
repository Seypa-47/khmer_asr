"""Shared experiment utilities, including the original Windows ICU scorer in WSL."""
import json
import os
import subprocess
from pathlib import Path


def score_rows(rows):
    interpreter = os.environ.get('KHMER_SCORE_PYTHON')
    if not interpreter:
        from score_qwen_rows import score_rows as local_score
        return local_score(rows)
    # WSL can run the existing Windows scorer without importing native Windows
    # training libraries. This preserves ICU 77.1 exactly across runtimes.
    script = str(Path(__file__).with_name('score_qwen_rows.py').resolve())
    if os.name != 'nt':
        script = subprocess.check_output(['wslpath', '-w', script], text=True).strip()
    completed = subprocess.run([interpreter, script], input=json.dumps(rows),
                               text=True, capture_output=True, check=True, timeout=120)
    return json.loads(completed.stdout)
