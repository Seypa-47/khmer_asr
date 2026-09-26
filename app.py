#!/usr/bin/env python
"""
Khmer Automatic Speech Recognition (ASR)
Ultra-Clean, Minimalist Voice Dictation UI (CADT IDRI Inspired).
"""

import gc
import os
import re
import sys
import time
import unicodedata
import av
import gradio as gr
import librosa
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Multi-threading for fast CPU execution
torch.set_num_threads(4)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_WHISPER_PATH = os.path.join(PROJECT_ROOT, "models", "whisper-tiny-khmer")
RETRAINED_WHISPER_PATH = os.path.join(PROJECT_ROOT, "models", "whisper-tiny-khmer-warmup35-b4")
LOCAL_MMS_PATH = os.path.join(PROJECT_ROOT, "models", "mms-khmer-ctc")
MATCHED_MMS_PATH = os.path.join(PROJECT_ROOT, "models", "mms-khmer-ctc-matched")
QWEN_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_lora_expanded_local_v2", "epoch-1")
QWEN_BASE_MODEL = "seanghay/Qwen3-ASR-0.6B-Khmer"
QWEN_MODEL_NAME = "Qwen3-ASR · Student LoRA (883 clips, epoch 1)"
MMS_MODEL_NAME = "🏆 Approach 2: Meta MMS-1B Khmer CTC (Local Checkpoint)"
HAS_QWEN_ADAPTER = os.path.isfile(os.path.join(QWEN_ADAPTER_PATH, "adapter_model.safetensors"))
DEFAULT_MODEL_NAME = QWEN_MODEL_NAME if HAS_QWEN_ADAPTER else MMS_MODEL_NAME
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mms_checkpoint_label():
    if os.path.isfile(os.path.join(MATCHED_MMS_PATH, "model.safetensors")):
        return "MMS-1B · matched-run checkpoint"
    if os.path.isfile(os.path.join(LOCAL_MMS_PATH, "model.safetensors")):
        return "MMS-1B · older local checkpoint"
    return "MMS-1B · public base model"

print(f"[*] Initializing Khmer ASR on {DEVICE.upper()}...")
# Cache models in memory on demand
MODELS = {}

def load_audio_file(audio_path, target_sr=16000):
    """Robust audio loader supporting .wav, .m4a, .mp3, .ogg, .flac via PyAV."""
    try:
        container = av.open(audio_path)
        audio_stream = next(s for s in container.streams if s.type == 'audio')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
        frames = []
        for frame in container.decode(audio_stream):
            frame.pts = None
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray())
        if frames:
            return np.concatenate(frames, axis=1).squeeze(), target_sr
    except Exception as e:
        print(f"[!] PyAV load warning: {e}, falling back to librosa...")
    return librosa.load(audio_path, sr=target_sr)

def get_model_and_processor(model_choice: str):
    if model_choice in MODELS:
        return MODELS[model_choice]

    # Keep one model resident so switching fits the laptop's 6 GB GPU.
    MODELS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if model_choice == QWEN_MODEL_NAME:
        from peft import PeftModel
        from qwen_asr import Qwen3ASRModel

        if not HAS_QWEN_ADAPTER:
            raise FileNotFoundError(f"Student Qwen adapter missing: {QWEN_ADAPTER_PATH}")
        print(f"[*] Loading Qwen base and student adapter from {QWEN_ADAPTER_PATH}...")
        wrapper = Qwen3ASRModel.from_pretrained(
            QWEN_BASE_MODEL,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="cuda:0" if DEVICE == "cuda" else "cpu",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        wrapper.model = PeftModel.from_pretrained(wrapper.model, QWEN_ADAPTER_PATH, is_trainable=False)
        wrapper.model.eval()
        MODELS[model_choice] = ("qwen", wrapper.processor, wrapper)
        return MODELS[model_choice]
    elif "MMS-1B" in model_choice:
        local_path = MATCHED_MMS_PATH if os.path.isfile(os.path.join(MATCHED_MMS_PATH, "model.safetensors")) else LOCAL_MMS_PATH
        if os.path.isfile(os.path.join(local_path, "model.safetensors")):
            print(f"[*] Loading Meta MMS-1B Khmer CTC from {local_path}...")
            proc = AutoProcessor.from_pretrained(local_path)
            mdl = Wav2Vec2ForCTC.from_pretrained(local_path).to(DEVICE)
        else:
            print("[*] Loading base Meta MMS-1B with Khmer adapter...")
            proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="khm")
            mdl = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="khm").to(DEVICE)
        mdl.eval()
        MODELS[model_choice] = ("mms", proc, mdl)
        return "mms", proc, mdl
    else:
        whisper_path = RETRAINED_WHISPER_PATH if "35-Step Warmup" in model_choice else LOCAL_WHISPER_PATH
        print(f"[*] Loading Whisper Tiny from {whisper_path}...")
        proc = WhisperProcessor.from_pretrained(whisper_path, language="Khmer", task="transcribe")
        mdl = WhisperForConditionalGeneration.from_pretrained(whisper_path).to(DEVICE)
        mdl.eval()
        MODELS[model_choice] = ("whisper", proc, mdl)
        return "whisper", proc, mdl

SAMPLE_METADATA = {
    "sample_1.wav": "មុខម្ហូបតាមដងផ្លូវ គឺជាមុខម្ហូបមួយមានភាពសម្បូរបែប និងមានភាពងាយស្រួល ដែលគេពេញនិយមក្នុងការបរិភោគ ថែមទាំងមានតម្លៃសមរម្យ។",
    "sample_2.wav": "នៅក្នុងប្រទេសកម្ពុជា មុខម្ហូបតាមដងផ្លូវមានប្រជាប្រិយភាពយ៉ាងខ្លាំង ហើយយើងអាចរកទិញមុខម្ហូបទាំងនោះដូចជា៖ នំបញ្ចុក ចេកចៀន បុកល្ហុង គុយទាវ ពងទាកូន បបរ នំប៉័ងដាក់សាច់ និងមានប្រភេទផ្សេងៗជាច្រើនទៀត។",
}


def normalize_khmer_text(text: str) -> str:
    """Pure deep learning text normalization: NFC Unicode and whitespace cleanup only.
    No hardcoded word replacements or synthetic scripts.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transcribe_audio(audio_path, model_choice):
    if not audio_path:
        return "", "0.0s", "សូមបញ្ចូលសំឡេងជាមុនសិន (No audio input)"

    mtype, processor, model = get_model_and_processor(model_choice)

    t0 = time.time()
    audio_array, sr = load_audio_file(audio_path, target_sr=16000)
    duration = len(audio_array) / 16000.0

    if mtype == "qwen":
        with torch.inference_mode():
            results = model.transcribe(audio=(np.ascontiguousarray(audio_array, dtype=np.float32), sr))
        raw_text = results[0].text
    elif mtype == "mms":
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
        with torch.no_grad():
            logits = model(inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        raw_text = processor.batch_decode(pred_ids)[0]
    else:
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        gen_kwargs = {
            "language": "khmer",
            "task": "transcribe",
        }
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                **gen_kwargs
            )
        raw_text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    prediction = normalize_khmer_text(raw_text)

    elapsed = time.time() - t0
    timer_str = f"⏱️ រយៈពេល៖ {elapsed:.2f}s (សំឡេង {duration:.1f}s)"
    source = mms_checkpoint_label() if mtype == "mms" else model_choice.split("(")[0].strip()
    status_str = f"🚀 {source} · {DEVICE.upper()}"

    return prediction, timer_str, status_str


def load_sample(sample_name):
    p = os.path.join(SAMPLES_DIR, sample_name)
    gt = SAMPLE_METADATA.get(sample_name, "")
    return p, gt


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Kantumruy+Pro:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

body, .gradio-container {
    background-color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.dark, .dark .gradio-container {
    background-color: #0b0f17 !important;
}

.app-wrapper {
    max-width: 880px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
}

/* CADT style minimalist header */
.header-container {
    text-align: center;
    padding: 24px 0 20px 0;
}
.waveform-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 8px;
}
.waveform-svg {
    height: 48px;
    color: #1e3a8a;
}
.header-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #0f172a;
    text-transform: uppercase;
    margin: 6px 0 2px;
}
.dark .header-title {
    color: #f1f5f9;
}
.header-sub {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
}

/* Big Clean Canvas */
.canvas-card {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
    padding: 8px !important;
    margin-bottom: 16px !important;
}
.dark .canvas-card {
    background: #131b2e !important;
    border-color: #1e293b !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3) !important;
}

.canvas-card textarea {
    font-family: 'Kantumruy Pro', 'Khmer OS Battambang', sans-serif !important;
    font-size: 22px !important;
    line-height: 1.8 !important;
    font-weight: 500 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 180px !important;
    padding: 12px 16px !important;
}
.dark .canvas-card textarea {
    color: #f8fafc !important;
}

/* Meta status bar below textarea */
.status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 8px 12px 8px;
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
}

/* Floating Clean Controls */
.controls-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    padding: 12px 0;
}

.sample-pills {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-bottom: 8px;
}
"""


def build_app():
    with gr.Blocks(title="Khmer ASR") as demo:
        gr.HTML(
            """
            <div class="app-wrapper">
                <div class="header-container">
                    <div class="waveform-logo">
                        <svg class="waveform-svg" viewBox="0 0 160 50" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M 10 25 L 35 25 L 45 10 L 55 40 L 65 5 L 75 45 L 85 20 L 95 30 L 105 25 L 115 25" stroke="#2563eb" />
                            <text x="122" y="38" font-family="'Kantumruy Pro', sans-serif" font-size="34" font-weight="bold" fill="#2563eb" stroke="none">ក</text>
                        </svg>
                    </div>
                    <div class="header-title">Khmer Automatic Speech Recognition (ASR)</div>
                    <div class="header-sub">ប្រព័ន្ធបម្លែងសំឡេងនិយាយទៅជាអក្សរខ្មែរ · Deep Learning Final Project</div>
                </div>
            </div>
            """
        )

        with gr.Column(elem_classes=["app-wrapper"]):
            # Main Canvas Text Box (CADT Style)
            transcript_box = gr.Textbox(
                placeholder="សូមនិយាយ ឬបញ្ចូលសំឡេងជាភាសាខ្មែរ... (Your Khmer transcript will appear here)",
                label="",
                lines=6,
                elem_classes=["canvas-card"],
            )

            # Timer and Telemetry Bar
            with gr.Row():
                timer_display = gr.Textbox(
                    value="ពេលវេលា៖ 00:00:00",
                    interactive=False,
                    scale=1,
                    container=False,
                )
                model_status = gr.Textbox(
                    value=QWEN_MODEL_NAME if HAS_QWEN_ADAPTER else mms_checkpoint_label(),
                    interactive=False,
                    scale=1,
                    container=False,
                )

            # Model Selection
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=[
                        *([QWEN_MODEL_NAME] if HAS_QWEN_ADAPTER else []),
                        MMS_MODEL_NAME,
                        "🧪 Approach 1: Whisper-Tiny Khmer (Version 1.0)",
                        "🧪 Whisper-Tiny Khmer (35-Step Warmup Retrain)",
                    ],
                    value=DEFAULT_MODEL_NAME,
                    label="🧠 ជ្រើសរើសម៉ូឌែល ASR (ASR Model Architecture)",
                    interactive=True,
                )

            # Audio Input & Action Row
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record Voice or Upload Audio",
                    scale=3,
                )
                transcribe_btn = gr.Button("🎙️ បម្លែងសំឡេង (Transcribe)", variant="primary", scale=1, size="lg")

            # Quick Preset Samples
            gr.Markdown("##### 💡 ឧទាហរណ៍សាកល្បងលឿន (Quick Test Presets):")
            with gr.Row():
                btn_s1 = gr.Button("▶ សាកល្បងគំរូទី ១: មុខម្ហូបតាមដងផ្លូវ (8.7s)", size="sm")
                btn_s2 = gr.Button("▶ សាកល្បងគំរូទី ២: នៅក្នុងប្រទេសកម្ពុជា (14.6s)", size="sm")
                btn_clear = gr.ClearButton(components=[transcript_box, audio_input], value="🗑️ សម្អាត (Clear)", size="sm")

            # Reference drawer
            with gr.Accordion("📋 មើលអត្ថបទដើមពិតប្រាកដ (Ground Truth Reference)", open=False):
                ground_truth_box = gr.Textbox(
                    label="Original Speech Reference",
                    interactive=False,
                    lines=2,
                )

            # Interactivity
            btn_s1.click(
                fn=lambda: load_sample("sample_1.wav"),
                outputs=[audio_input, ground_truth_box],
            )
            btn_s2.click(
                fn=lambda: load_sample("sample_2.wav"),
                outputs=[audio_input, ground_truth_box],
            )

            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=[audio_input, model_selector],
                outputs=[transcript_box, timer_display, model_status],
                concurrency_limit=1,
            )

    return demo


if __name__ == "__main__":
    get_model_and_processor(DEFAULT_MODEL_NAME)
    print("[*] Models ready!")
    app = build_app()
    print("[*] Launching CADT-style minimalist UI at http://127.0.0.1:7860 ...")
    app.launch(server_name="127.0.0.1", server_port=7860, css=CUSTOM_CSS, theme=gr.themes.Soft(), share=False)
