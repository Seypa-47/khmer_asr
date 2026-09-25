# MMS matched split: checkpoint-300 error analysis

The 15-epoch MMS run was interrupted after step 800 and resumed to its planned step 1005. The best saved validation checkpoint remained step 300 (14.45% validation CER); the final step scored 14.70% validation CER. We evaluated checkpoint 300 once on the 114 untouched FLEURS Khmer test clips selected by the shared manifest. Test CER was **15.84%** after NFC normalization, removal of U+200B/U+FEFF, and whitespace removal. The full 114 reference/prediction pairs are saved in this directory as `mms_matched_best300_predictions.json` and in Google Drive under `MyDrive/khmer_asr_final_runs/results/`.

Of the 114 clips, 34 had individual CER of at least 20%, and 10 had CER of at least 30%. These are observed text differences; the audio has not been reviewed for these examples, so acoustic causes remain hypotheses.

Among the 26 references containing Latin letters, corpus CER was **23.75%**. Among the other 88 references, it was **13.03%**. The 10 references containing digits had **15.99%** CER. These are descriptive subgroups, not separate test sets or a new headline score. Latin-script content is associated with more errors, but the reference script itself can contribute to the measured difference when the model produces a Khmer transliteration.

| FLEURS test index | Individual CER | Observed failure |
| --- | ---: | --- |
| 245 | 46.72% | The reference contains Latin-script names and titles such as `lakkha singh` and `raju khandelwal`; the hypothesis mixes Khmer approximations with Latin fragments. |
| 329 | 44.79% | A multiword personal name in Latin script is largely lost; surrounding Khmer words also contain substitutions. |
| 280 | 36.11% | The name `rolando mendoza` and `m16` are rendered inconsistently, with Latin characters embedded in Khmer text. |
| 211 | 33.33% | This mostly Khmer sentence still has several substitutions, showing that mixed-script names are not the only failure pattern. |
| 388 | 32.40% | A long sentence with repeated specialized terms and Latin-script transliterations accumulates errors. |

The strongest measured pattern is difficulty with mixed Khmer/Latin references, especially names and foreign terms. The examples also show number and ordinary Khmer word errors, but the digit subgroup alone does not establish a strong number effect. The decoder frequently omits spaces, so raw whitespace-based WER is unsuitable for Khmer. The same ICU Khmer word-break policy now scores both models' saved predictions; see `matched_cer_wer_summary.json`. The MMS checkpoint has **67.16% segmented WER** (1,671 edits over 2,488 reference words), alongside **15.84% CER** (1,883 edits over 11,886 reference characters). These metrics capture different error units, so the character result alone does not establish accurate whole-word transcription.

Next, listen to selected failed clips and classify the causes more carefully. Separately recorded voices have no verified exact transcripts and therefore have no defensible accuracy percentage. More epochs alone are not supported by the complete validation curve: the best CER occurred at step 300 and no later validation check through step 1005 improved it.
