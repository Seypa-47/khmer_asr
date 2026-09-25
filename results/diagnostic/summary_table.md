# Saved checkpoint comparison (diagnostic)

All listed checkpoints are scored on the first 200 examples of the official Google FLEURS `km_kh` test split. CER uses NFC normalization, removes U+200B/U+FEFF, then removes whitespace before scoring.

| Approach | Training strategy | Trainable parameters | Train / validation evidence | Epochs | Test CER | Test rows | Training time |
|---|---|---:|---:|---:|---:|---:|---|
| Whisper-Tiny full fine-tuning | Full fine-tuning | 37.80M | about 941 / 189 retained (inferred) | 3 | 83.89% | 200 | Not recorded |
| Meta MMS-1B CTC | Top four encoder layers plus CTC head/adapter unfrozen | 79.08M | 1,000 / 200 requested; retained counts unlogged | 15 | 15.96% | 200 | Not recorded |
| Whisper-Tiny 35-step warmup | Full fine-tuning, 35-step warmup | 37.80M | 941 / 189 retained | 3 | 83.67% | 200 | About 11 minutes on RTX 4050 Laptop GPU |

> These results are diagnostic: the runs did not retain identical train/validation example manifests and applied different filtering. The shared test evaluation does not remove that training-data difference.

> MMS training history was not saved, so its training/validation learning curve cannot be reconstructed from the available files. The frozen-encoder run also has no saved checkpoint or metrics and is excluded.
