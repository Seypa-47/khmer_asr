# Final Project Presentation Slide Deck Outline
**Course**: Deep Learning (Bachelor of Software Engineering)  
**Student**: [Your Name / Student ID]  
**Lecturer**: Mr. Soklong HIM  
**Title**: Comparative Study of Deep Learning Approaches for Khmer Automatic Speech Recognition (ASR)  
**Format**: 12 Slides (~10-minute presentation + 5-minute Q&A)

---

### Slide 1: Title & Overview
* **Title**: Comparative Evaluation of Sequence-to-Sequence and CTC Deep Learning Architectures for Khmer Automatic Speech Recognition
* **Course & Academic Year**: Deep Learning | 2026–2027
* **Presenter**: [Your Name]
* **Repository**: GitHub link to this repository

---

### Slide 2: Problem Definition & Motivation
* **Task Definition**: Input raw acoustic speech waveform $\mathbf{X} \in \mathbb{R}^T \rightarrow$ Output Khmer Unicode text transcript $\mathbf{Y} = (y_1, y_2, \dots, y_U)$.
* **Significance & Practical Relevance**:
  * Khmer is an under-resourced language in speech technology.
  * Unique linguistic challenges: Continuous script without word spaces (scriptio continua), intricate consonant clusters, subscript consonants (*cheung*), and vowel diacritics.
  * Real-world applications: Assistive voice technologies, voice search, transcription for public services and education in Cambodia.

---

### Slide 3: Dataset & Speech Preprocessing Pipeline
* **Primary Benchmark**: Google FLEURS Khmer (`km_kh`) subset.
  * Sampling rate: 16,000 Hz, 16-bit mono PCM.
  * Split: Fixed Train (1,000 samples) / Validation (200 samples) / Held-out Test (200 samples).
* **Acoustic Preprocessing**:
  * Whisper: 80-channel Log-mel Spectrogram extraction across 25ms sliding windows with 10ms hop.
  * MMS: Raw 16 kHz 1D waveform feature extraction with layer normalization.
* **Text Preprocessing**:
  * Unicode NFC normalization.
  * Stripping zero-width non-breaking spaces (`\u200b`, `\ufeff`).

---

### Slide 4: Architectural Approaches Compared (Section 4 Compliance)
* **Approach 1: Autoregressive Seq2Seq Transformer (OpenAI Whisper-Tiny)**
  * Full fine-tuning of audio encoder and autoregressive text decoder (~37.8M parameters).
  * Cross-attention mechanism aligns speech frames with subword tokens.
* **Approach 2: Non-Autoregressive CTC Acoustic Model (Meta MMS-1B Khmer)**
  * Connectionist Temporal Classification (CTC) loss; aligns frame-level acoustic representations without recurrent/autoregressive decoding.
  * Language adapter fine-tuning (~2.5M trainable parameters).
* **Approach 3: Linear Probe / Parameter-Efficient Ablation (Whisper Frozen Encoder)**
  * Encoder parameters frozen; only cross-attention and text decoder weights fine-tuned (~28.5M parameters).

---

### Slide 5: Justification of Selected Approaches
* **Theoretical Contrast**:
  * **Seq2Seq Attention** incorporates an implicit language model inside the decoder, capturing long-range Khmer semantic context.
  * **CTC Modeling** assumes conditional independence between output tokens given alignment, resulting in extremely fast inference and low hallucination risk.
  * **Frozen vs. Full Fine-Tuning** tests whether pre-trained multilingual acoustic features transfer directly to low-resource Khmer phonetics without acoustic layer updates.

---

### Slide 6: Experimental Setup & Hyperparameter Configuration
* **Hardware**: Google Colab Tesla T4 GPU (16 GB VRAM).
* **Optimizer & Scheduler**: AdamW ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$), linear warmup (100–500 steps) with linear decay.
* **Tuning Exploration**: Learning rate ($10^{-5}$ vs. $10^{-4}$), batch size 8 with gradient accumulation.
* **Precision**: FP16 mixed precision for reduced memory footprint.
* **Reproducibility**: Global seeds set to `42` across Python `random`, `numpy`, and `torch.manual_seed`.

---

### Slide 7: Evaluation Metrics (The Khmer WER vs. CER Analysis)
* **Primary Metric — Character Error Rate (CER)**:
  $$\text{CER} = \frac{S + D + I}{N} \times 100\%$$
  where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total reference characters.
* **Why Not Standard WER?**:
  * Khmer words are written continuously without spaces.
  * Without dictionary-based word segmentation, standard WER treats an entire clause as a single word, yielding uninformative ~100% error rates.
  * CER accurately reflects phonetic and character accuracy.

---

### Slide 8: Consolidated Experimental Results
*(Refer to figures in `results/`)*
* **Summary Table**:
  | Approach | Model Architecture | Trainable Params | Hardware | Test CER (%) | Test WER (%) |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Approach 1** | Whisper-Tiny (Seq2Seq) | 37.8M | Colab T4 | 83.6% | 100.0% |
  | **Approach 2** | Meta MMS-1B (CTC) | 2.5M (Adapter) | Colab T4 | 48.2% | 94.5% |
  | **Approach 3** | Whisper Frozen Enc (Probe) | 28.5M | Colab T4 | 89.4% | 100.0% |
* **Key Observations**:
  * MMS CTC achieved significantly lower CER on small Khmer datasets due to its massive pre-trained acoustic foundation and phoneme-level CTC alignment.

---

### Slide 9: Visual Learning Curves & Metric Comparison
* **Figure 1 (`results/learning_curves.png`)**:
  * Training loss vs. Validation loss over 354 optimization steps.
  * Steady cross-entropy reduction from 2.31 to 1.47 without severe overfitting.
* **Figure 2 (`results/metrics_comparison.png`)**:
  * Side-by-side bar chart demonstrating CER across all three approaches.

---

### Slide 10: Error Analysis & Failure Cases
* **Inspection of Prediction Errors**:
  1. **Consonant Subscripts (*Cheung*) Confusion**: Model occasionally omits subscript indicators (`\u17d2`), e.g., confusing ្ត with ត.
  2. **Inherent Vowel Ambiguity**: Series 1 vs Series 2 Khmer vowels having identical spelling rules with subtle pitch/timbre variations.
  3. **Background Acoustic Noise**: Audio with environmental reverb shows higher insertion rates.
  4. **Repetition Loops**: Whisper Seq2Seq occasionally hallucinates repetitive tokens on ambiguous audio silence.

---

### Slide 11: Limitations & Future Work
* **Limitations**:
  * Small training slice (1,000 samples) due to Google Colab GPU timeout constraints.
  * Absence of an external Khmer N-gram Language Model to rescore CTC beams.
* **Future Work**:
  * Train on OpenSLR SLR42 (100+ hours of Khmer speech) for production readiness.
  * Integrate `khmernltk` / `searn` tokenization for true word-level error metrics (WER).
  * Deploy quantized model (ONNX / INT8) for low-latency edge devices.

---

### Slide 12: Live Demonstration & Q&A
* **Interactive Testing Web App**:
  * Built using Gradio (`app.py`).
  * Demonstrates real-time microphone speech recognition and file upload.
* **Open for Questions**:
  * Ready to show code, checkpoints, and explain any tensor operation in the repository.
