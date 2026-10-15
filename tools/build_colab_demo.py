"""Build the self-contained Colab inference notebook for the selected Qwen LoRA."""

from __future__ import annotations

import base64
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "qwen_bigtrain_live_demo_colab.ipynb"


def cell(kind: str, source: str, *, hidden: bool = False) -> dict:
    item = {
        "cell_type": kind,
        "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


sample_b64 = base64.b64encode((ROOT / "samples" / "sample_1.wav").read_bytes()).decode("ascii")
inference_source = (ROOT / "src" / "qwen_khmer_inference.py").read_text(encoding="utf-8")

cells = [
    cell("markdown", """
    # Khmer ASR live demo: Qwen3-ASR 0.6B + my big-corpus LoRA

    This notebook runs **Approach 3** from the presentation: the pinned
    `seanghay/Qwen3-ASR-0.6B-Khmer` base plus my selected epoch-1 LoRA adapter
    (16,720 effective training examples). It performs inference only; it does
    not retrain the model. It includes a short sample audio for a readiness test.

    **Before the presentation:** In Google Colab, select **Runtime → Change
    runtime type → GPU** (T4 or better), then run cells 1-4 in order. Approve
    Google Drive access if asked. The adapter is expected at
    `MyDrive/khmer_asr_runs/qwen_km_corpus_bigtrain_20261014/selected_adapter_and_results.zip`.
    If it is absent, cell 2 lets you upload the supplied backup ZIP. Wait for
    the green **READY** output. Keep this tab and runtime connected.

    **During the presentation:** Run the final upload cell and choose a short
    Khmer WAV, MP3, or M4A recording. The notebook displays the audio,
    transcription, and inference time. A 5-20 second utterance works well.

    Opening a Colab notebook alone cannot start a GPU or execute code; you must
    run the cells. The first dependency install and base-model download may
    take several minutes. Colab may disconnect or have no GPU available, so do the
    readiness test before presenting and keep the local demo as a backup.
    """),
    cell("code", """
    # 1. Runtime and dependencies. This must finish without errors.
    import shutil, subprocess, sys
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('No NVIDIA GPU. In Colab: Runtime > Change runtime type > GPU, then rerun.')
    print(subprocess.check_output(['nvidia-smi', '-L'], text=True).strip())
    if not shutil.which('ffmpeg'):
        subprocess.run(['apt-get', '-qq', 'update'], check=True)
        subprocess.run(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], check=True)
    # Match the successful training Colab environment before importing torch.
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'torch==2.8.0', 'torchaudio==2.8.0',
        '--index-url', 'https://download.pytorch.org/whl/cu128',
    ], check=True)
    packages = [
        'qwen-asr==0.0.6', 'transformers==4.57.6', 'peft==0.21.0',
        'accelerate==1.12.0', 'soundfile', 'scipy',
    ]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot access the GPU. Reconnect to a Colab GPU runtime.')
    print('GPU ready:', torch.cuda.get_device_name(0), '| torch', torch.__version__)
    """),
    cell("code", """
    # 2. Retrieve and verify the exact adapter used for the reported result.
    from pathlib import Path
    import hashlib, zipfile
    from google.colab import drive, files

    ARCHIVE = Path('/content/drive/MyDrive/khmer_asr_runs/qwen_km_corpus_bigtrain_20261014/selected_adapter_and_results.zip')
    EXPECTED_SHA256 = '44d1eff105a7f51647bb23f887335104340528aaae13165d5e9400024ba694d2'
    ADAPTER_DIR = Path('/content/qwen_khmer_epoch1_adapter')
    ADAPTER_DIR.mkdir(exist_ok=True)
    if not Path('/content/drive/MyDrive').exists():
        try:
            drive.mount('/content/drive')
        except Exception as exc:
            print('Drive mount failed; upload the backup ZIP instead:', exc)
    if not ARCHIVE.is_file():
        print('Select qwen_epoch1_adapter_backup.zip from your computer.')
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError('Upload exactly one adapter ZIP, then rerun this cell.')
        ARCHIVE = Path('/content') / next(iter(uploaded))
    with zipfile.ZipFile(ARCHIVE) as zf:
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            if name not in zf.namelist():
                raise ValueError(f'Adapter ZIP is missing {name}')
            (ADAPTER_DIR / name).write_bytes(zf.read(name))
    digest = hashlib.sha256((ADAPTER_DIR / 'adapter_model.safetensors').read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f'Wrong adapter (SHA256 {digest}). Use the supplied epoch-1 backup ZIP.')
    print('Exact selected adapter verified:', digest)
    """),
    cell("code", """
    # 3. Load once; subsequent demo uploads reuse this model in GPU memory.
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel
    BASE = 'seanghay/Qwen3-ASR-0.6B-Khmer'
    REVISION = '351f1cc594f084e4899ef6cf7bbb17ec64484080'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    wrapper = Qwen3ASRModel.from_pretrained(
        BASE, revision=REVISION, dtype=dtype, device_map='cuda:0',
        max_inference_batch_size=1, max_new_tokens=512,
    )
    wrapper.model = PeftModel.from_pretrained(wrapper.model, str(ADAPTER_DIR), is_trainable=False)
    wrapper.model.eval()
    print('Loaded:', BASE, '| selected LoRA epoch 1 |', dtype)
    """),
    cell("code", inference_source + "\n" + textwrap.dedent("""
    import time
    import soundfile as sf
    import subprocess
    from IPython.display import Audio, display

    def transcribe_file(path):
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 25_000_000:
            raise ValueError('Use an existing audio file smaller than 25 MB.')
        converted = Path('/content/khmer_demo_input_16k.wav')
        command = ['ffmpeg', '-loglevel', 'error', '-y', '-i', str(path),
                   '-ac', '1', '-ar', '16000', str(converted)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Could not decode audio: ' + result.stderr[-1200:])
        wav, rate = sf.read(converted, dtype='float32')
        seconds = len(wav) / rate
        if not (0.25 <= seconds <= 30):
            raise ValueError(f'Audio is {seconds:.1f}s; use 0.25-30 seconds for this demo.')
        start = time.perf_counter()
        text = transcribe_khmer(wrapper, wav, rate)
        elapsed = time.perf_counter() - start
        if not text.strip():
            raise RuntimeError('The model returned an empty transcript. Try a clearer recording.')
        return text, seconds, elapsed
    """)),
    cell("code", """
    # 4. Required readiness test: this sample is embedded in the notebook.
    import base64
    SAMPLE = Path('/content/khmer_demo_sample.wav')
    SAMPLE.write_bytes(base64.b64decode(SAMPLE_B64))
    display(Audio(str(SAMPLE)))
    sample_text, sample_seconds, sample_elapsed = transcribe_file(SAMPLE)
    print('Sample transcript:', sample_text)
    print(f'Audio {sample_seconds:.1f}s | inference {sample_elapsed:.1f}s')
    print('✅ READY — the selected model is loaded and produced a nonempty Khmer transcript.')
    """),
    cell("markdown", """
    ## Live presentation cell

    Run this cell when you want to demonstrate a new recording. Choose one
    audio file; upload a short Khmer utterance. It does not use the held-out
    test references and does not change the reported test metrics. You can
    rerun this cell for another clip while the runtime remains connected.
    """),
    cell("code", """
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Upload exactly one WAV, MP3, or M4A file, then rerun.')
    audio_path = Path('/content') / next(iter(uploaded))
    display(Audio(str(audio_path)))
    transcript, duration, elapsed = transcribe_file(audio_path)
    print('Khmer transcription:')
    print(transcript)
    print(f'Audio: {duration:.1f}s | inference: {elapsed:.1f}s')
    """),
    cell("markdown", """
    **Demo backup:** If Colab loses its GPU, use the same selected model in the
    already tested local app at `http://127.0.0.1:7860/`. Keep a short WAV
    on your laptop. The first Colab base-model download can take several
    minutes; run the readiness test well before the talk.
    """),
]

# Keep the binary sample in a hidden setup cell instead of depending on GitHub.
cells.insert(5, cell("code", "SAMPLE_B64 = (\n" + "\n".join(
    f"    '{sample_b64[i:i+120]}'" for i in range(0, len(sample_b64), 120)
) + "\n)\n", hidden=True))

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT, OUTPUT.stat().st_size)
