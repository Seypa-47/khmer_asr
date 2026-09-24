# MMS matched split: checkpoint-300 error analysis

The 15-epoch MMS run stopped after step 800 of 1005. The best saved validation checkpoint was step 300 (14.45% validation CER). We evaluated that checkpoint once on the 114 untouched FLEURS Khmer test clips selected by the shared manifest. Test CER was **15.84%** after NFC normalization, removal of U+200B/U+FEFF, and whitespace removal. This is a partial-run checkpoint result, not a completed-run result. The full 114 reference/prediction pairs are saved in Google Drive at `MyDrive/khmer_asr_final_runs/results/mms_matched_best300_predictions.json`.

Of the 114 clips, 34 had individual CER of at least 20%, and 10 had CER of at least 30%. These are observed text differences; the audio has not been reviewed for these examples, so acoustic causes remain hypotheses.

| FLEURS test index | Individual CER | Observed failure |
| --- | ---: | --- |
| 245 | 46.72% | The reference contains Latin-script names and titles such as `lakkha singh` and `raju khandelwal`; the hypothesis mixes Khmer approximations with Latin fragments. |
| 329 | 44.79% | A multiword personal name in Latin script is largely lost; surrounding Khmer words also contain substitutions. |
| 280 | 36.11% | The name `rolando mendoza` and `m16` are rendered inconsistently, with Latin characters embedded in Khmer text. |
| 211 | 33.33% | This mostly Khmer sentence still has several substitutions, showing that mixed-script names are not the only failure pattern. |
| 388 | 32.40% | A long sentence with repeated specialized terms and Latin-script transliterations accumulates errors. |

The strongest visible pattern is difficulty with names, foreign terms, numbers, and mixed Khmer/Latin references. Some ordinary Khmer words also differ. The decoder frequently omits spaces, so raw whitespace-based word error rate is unsuitable as the primary metric for this unsegmented Khmer output. Character error rate after a fixed normalization policy provides the reproducible comparison, while the examples show where it is insufficient to judge readability.

Next, listen to selected failed clips and classify the causes more carefully. Evaluate the user's separate recorded voices with verified transcripts before claiming performance outside FLEURS. More epochs alone are not supported by the validation curve: the best CER occurred at step 300 and later validation checks through step 800 did not improve it.
