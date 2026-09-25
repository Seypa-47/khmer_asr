# Results guide

This directory contains the **final controlled comparison**. Both models used the same fixed FLEURS `km_kh` row manifest in `matched_fleurs_split.json`: 531 training, 124 validation, and 114 held-out test clips.

| File | Purpose |
|---|---|
| `matched_fleurs_split.json` | Exact selected row indices and eligibility rules. |
| `matched_cer_wer_summary.json` | Audited comparison on the same 114 clips: Whisper 86.76% CER / 106.27% WER; MMS 15.84% CER / 67.16% WER. Includes edit counts and tokenization policy. |
| `whisper_matched_predictions.json` | Whisper reference and prediction pairs for all 114 public FLEURS test clips. |
| `mms_matched_best300_predictions.json` | MMS reference and prediction pairs for the same clips. |
| `whisper_matched_metrics.json` | Original Whisper Trainer log: 86.93% CER from decoded labels. The paired comparison rescores the saved checkpoint against raw FLEURS references. |
| `mms_matched_best300_summary.json` | MMS best checkpoint and held-out CER summary. |
| `mms_matched_metrics.json` | MMS training configuration and summary. |
| `mms_tuning_summary.json` | MMS pilot tuning choices. |
| `mms_matched_learning_history.json` | Compact saved MMS training and validation series. |
| `mms_matched_best300_error_analysis.md` | Failure examples and limitations. |
| `matched_cer_wer.png`, `matched_test_cer.png`, `mms_error_subgroups.png`, `mms_matched_learning_curves.png` | Figures for the final comparison. |

CER is computed after NFC normalization and removal of U+200B/U+FEFF and whitespace. WER applies ICU `km_KH` dictionary word boundaries to both references and predictions after NFC normalization and removal of U+200B/U+FEFF, then excludes punctuation-only segments. Both rates are corpus edit distance divided by reference units. They are not percentages of fully correct sentences; WER may exceed 100% when there are many insertions. The full Trainer states remain in Google Drive. The compact MMS learning history here is derived from that state.

Recompute the metrics from the prediction pairs with `python src/score_matched_cer_wer.py --whisper results/whisper_matched_predictions.json --mms results/mms_matched_best300_predictions.json --output results/matched_cer_wer_summary.json`. Generate the comparison figure with `python src/plot_cer_wer.py`.

`diagnostic/` contains **earlier exploratory runs** that did not share identical training examples. Those files are retained for audit and are not the final architecture comparison. Personal recordings, draft transcripts, and their raw model outputs are excluded from the repository.
