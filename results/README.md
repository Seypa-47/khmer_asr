# Results guide

This directory contains the **final controlled comparison**. All three trained approaches used the same fixed FLEURS `km_kh` row manifest in `matched_fleurs_split.json`: 531 training, 124 validation, and 114 held-out test clips.

| File | Purpose |
|---|---|
| `matched_fleurs_split.json` | Exact selected row indices and eligibility rules. |
| `matched_cer_wer_summary.json` | Audited comparison on the same 114 clips: Whisper 86.76% CER / 106.27% WER; MMS 15.84% CER / 67.16% WER. Includes edit counts and tokenization policy. |
| `matched_cer_wer_three_approaches.json` | Final comparison including head-only MMS: 15.71% CER / 65.80% WER on the same 114 clips. |
| `whisper_matched_predictions.json` | Whisper reference and prediction pairs for all 114 public FLEURS test clips. |
| `mms_matched_best300_predictions.json` | MMS reference and prediction pairs for the same clips. |
| `mms_frozen_predictions.json` | Frozen-encoder MMS prediction pairs for the same clips. |
| `whisper_matched_metrics.json` | Original Whisper Trainer log: 86.93% CER from decoded labels. The paired comparison rescores the saved checkpoint against raw FLEURS references. |
| `mms_matched_best300_summary.json` | MMS best checkpoint and held-out CER summary. |
| `mms_matched_metrics.json` | MMS training configuration and summary. |
| `mms_frozen_metrics.json`, `mms_frozen_trainer_state.json` | Third-approach training configuration, parameter counts, validation history, and selected checkpoint. |
| `mms_tuning_summary.json` | MMS pilot tuning choices. |
| `mms_matched_learning_history.json` | Compact saved MMS training and validation series. |
| `mms_matched_best300_error_analysis.md` | Failure examples and limitations. |
| `mms_frozen_error_analysis.md` | Paired comparison of the two MMS strategies, including uncertainty intervals. |
| `matched_cer_wer_three_approaches.png`, `mms_frozen_learning_curves.png` | Final three-approach comparison and third-approach learning curves. |
| `matched_cer_wer.png`, `matched_test_cer.png`, `mms_error_subgroups.png`, `mms_matched_learning_curves.png` | Earlier two-approach figures retained for audit. |

CER is computed after NFC normalization and removal of U+200B/U+FEFF and whitespace. WER applies ICU `km_KH` dictionary word boundaries to both references and predictions after NFC normalization and removal of U+200B/U+FEFF, then excludes punctuation-only segments. Both rates are corpus edit distance divided by reference units. They are not percentages of fully correct sentences; WER may exceed 100% when there are many insertions. The frozen MMS Trainer state is saved here; the original two Trainer states remain in Google Drive. The compact top-four MMS learning history here is derived from its full state.

Recompute the final metrics from the prediction pairs with `python src/score_matched_cer_wer.py --whisper results/whisper_matched_predictions.json --mms results/mms_matched_best300_predictions.json --frozen-mms results/mms_frozen_predictions.json --output results/matched_cer_wer_three_approaches.json`. Generate the final figure with `python src/plot_cer_wer.py --summary results/matched_cer_wer_three_approaches.json --output results/matched_cer_wer_three_approaches.png`. Run `python src/analyze_third_approach.py` to regenerate the third approach's learning curves and paired analysis.

`diagnostic/` contains **earlier exploratory runs** that did not share identical training examples. Those files are retained for audit and are not the final architecture comparison. Personal recordings, draft transcripts, and their raw model outputs are excluded from the repository.

`diagnostic/qwen_public_validation_summary.json` records a separate external-model probe on the **validation** split. The publisher-trained checkpoint achieved 14.29% CER and 32.07% ICU Khmer WER on 124 clips. It is neither a student-trained approach nor a held-out test result, so it is excluded from the final three-approach table and figures. The new student LoRA run defined in `src/train_qwen_lora.py` must be evaluated separately before any result is added here.

`diagnostic/qwen_lora_validation_v1.json` records the first student-owned adapter's **validation** score: 14.45% CER and 33.12% ICU Khmer WER after one epoch on 531 train clips. Both are slightly worse than the publisher checkpoint. This adapter is retained for audit and is not promoted to the held-out test comparison.
