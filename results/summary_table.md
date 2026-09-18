# Model Approaches Comparison Table

Evaluated on identical held-out test split: Google FLEURS `km_kh` (200 test samples).

| Approach | Model Architecture | Strategy | Trainable Params | Hardware | Training Time | Test CER (%) | Test WER (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Approach 1 (Trained)** | Whisper-Tiny (Seq2Seq Transformer) | Full Fine-Tuning | 37.8M | Tesla T4 GPU (Google Colab) | ~15 minutes | **83.6%** | 100.0%* |
| **Approach 2 (Comparison)** | Meta MMS-1B Khmer (Acoustic CTC Model) | Adapter / Transfer Learning | 2.5M | Tesla T4 GPU (Google Colab) | ~10 minutes | **48.2%** | 94.5%* |
| **Approach 3 (Ablation)** | Whisper-Tiny (Frozen Enc) (Seq2Seq Transformer) | Linear Probe / Decoder-Only | 28.5M | Tesla T4 GPU (Google Colab) | ~11 minutes | **89.4%** | 100.0%* |

> \* *Note on Khmer WER: Khmer text is written without word delimiters (scriptio continua). Standard unsegmented WER treats sentences as monolithic tokens resulting in inflated WER (~100%). Character Error Rate (CER) is the recognized primary evaluation metric for Khmer ASR.*
