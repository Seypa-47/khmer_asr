# Third approach: frozen MMS encoder with a trained CTC head

Both MMS runs used the same 531/124/114 FLEURS Khmer split. The third run trained only 194,712 head parameters; its 964,843,288-parameter model otherwise stayed frozen. Validation ICU WER selected checkpoint 100 of 335 optimizer steps across five epochs.

| Approach | Held-out CER | Held-out ICU WER |
|---|---:|---:|
| MMS, top four encoder layers and head trained | 15.84% | 67.16% |
| MMS, head only trained | 15.71% | 65.80% |

CER improvement (top-four tuned minus head-only): 0.13 percentage points. Paired utterance bootstrap 95% interval: -0.20 to 0.48 points (4,000 seed-42 resamples). Across clips: 42 improved, 38 worsened, 34 tied.
WER improvement (top-four tuned minus head-only): 1.37 percentage points. Paired utterance bootstrap 95% interval: -0.38 to 3.21 points (4,000 seed-42 resamples). Across clips: 45 improved, 33 worsened, 36 tied.

The small measured difference does not establish a reliable improvement if its bootstrap interval crosses zero. Both CER and WER use the documented shared normalization and ICU Khmer word breaks. The WER remains far above the lecturer's under-20% error target.

The saved prediction pairs contain public FLEURS references and model outputs. Personal recordings and the student's pasted test sentences were not used as training labels, test references, or hardcoded corrections.
