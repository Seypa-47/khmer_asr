# Results guide

This directory contains the **final controlled comparison**. Both models used the same fixed FLEURS `km_kh` row manifest in `matched_fleurs_split.json`: 531 training, 124 validation, and 114 held-out test clips.

| File | Purpose |
|---|---|
| `matched_fleurs_split.json` | Exact selected row indices and eligibility rules. |
| `whisper_matched_metrics.json` | Whisper-Tiny test result: 86.93% CER. |
| `mms_matched_best300_summary.json` | MMS best checkpoint and test result: 15.84% CER. |
| `mms_matched_metrics.json` | MMS training configuration and summary. |
| `mms_tuning_summary.json` | MMS pilot tuning choices. |
| `mms_matched_learning_history.json` | Compact saved MMS training and validation series. |
| `mms_matched_best300_error_analysis.md` | Failure examples and limitations. |
| `matched_test_cer.png`, `mms_error_subgroups.png`, `mms_matched_learning_curves.png` | Figures for the final comparison. |

The 114 full reference/prediction pairs and complete Trainer states remain in the student's Google Drive and are not yet in this repository. The compact MMS learning history here is derived from that state. Scores are character error rates after the documented normalization; they are not percentages of fully correct sentences.

`diagnostic/` contains **earlier exploratory runs** that did not share identical training examples. Those files are retained for audit and are not the final architecture comparison. Personal recordings, draft transcripts, and their raw model outputs are excluded from the repository.
