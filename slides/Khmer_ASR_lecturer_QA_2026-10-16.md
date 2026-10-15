# Khmer ASR: lecturer questions and model answers

Presentation rehearsal, 16 October 2026. This guide follows the current 12-slide deck. Learn the short answers first, then use the technical answers if the lecturer asks for detail.

## The answer to practise first: Why these three approaches? (Slide 6)

**Question. Why did you choose Whisper-Tiny, MMS-1B, and Qwen3-ASR?**

**Answer.** I chose three different ways to adapt a pretrained speech recognizer to Khmer. Whisper-Tiny gives a small encoder-decoder baseline with full fine-tuning. MMS-1B gives a multilingual speech encoder with CTC output, where I updated only its upper layers and output components. Qwen3-ASR gives a Khmer-adapted generative starting point for a parameter-efficient LoRA experiment. This choice lets me compare practical model families and adaptation methods on one held-out test set. It does **not** prove which architecture is inherently best: Qwen started from a Khmer-adapted checkpoint and used far more effective training examples.

**If pressed: why not compare them as a controlled experiment?** Whisper and MMS both used 531 training utterances, but Qwen used 16,720 effective examples and a different starting checkpoint. The common 114-utterance test makes the reported scores comparable as outcomes of these saved systems; it does not isolate architecture, model size, or LoRA as the cause.

**If asked why a head-only MMS run is not one of the three slides:** I also completed a separate frozen-encoder MMS experiment (five epochs; 15.71% CER and 65.80% WER on this test). It is useful for comparing MMS fine-tuning strategies. I selected Qwen for this presentation to cover a third model family and a Khmer-specific LoRA path. A strictly controlled adaptation study would compare the two MMS settings more directly.

## Project flow in about one minute

1. Define the user problem: making a usable first Khmer transcript from a short recording, reducing the burden of manual listening and typing. This project measures transcription errors, not minutes of human work saved.
2. Prepare a fixed FLEURS Khmer manifest with 531 training, 124 validation, and 114 held-out test utterances. Whisper and MMS use the 531 training utterances. Qwen also uses additional public Khmer speech; its per-epoch training list has 16,720 effective examples, including repeated FLEURS examples.
3. Load three existing pretrained checkpoints. Adapt Whisper by updating all weights, MMS by updating its top four encoder layers and output components, and Qwen by training a LoRA adapter on attention projections. None was trained from random weights.
4. Train, save checkpoints, and choose using validation only: Whisper ran three epochs; MMS ran 15 epochs and selected checkpoint step 300 by validation CER; Qwen ran three epochs and selected epoch 1 by validation ICU Khmer WER.
5. Run the selected systems on the same 114 held-out utterances. Normalize text consistently and calculate corpus CER and ICU-segmented Khmer WER from substitutions, deletions, and insertions.
6. Interpret the errors and limitations. Qwen had the lowest observed test errors, but the data and starting models differ. A live demo shows inference on one recording; it is separate from the test measurements.

## Training facts to memorize (Slides 3, 5, 7, and 8)

**Approach 1 — Whisper-Tiny.** Starting checkpoint: `openai/whisper-tiny`. Full fine-tuning of all model weights. The matched run trained on 531 FLEURS utterances for **3 completed epochs**. The saved model was evaluated on 114 held-out utterances: **86.76% CER, 106.27% ICU Khmer WER**. The slide does not state a selected Whisper checkpoint step, and the local summary does not preserve one; do not invent it.

**Approach 2 — MMS-1B CTC.** Starting checkpoint: `facebook/mms-1b-all`, with Khmer (`khm`) target. The matched run trained on the same 531 utterances for **15 completed epochs / 1,005 optimizer steps**. It updated the top four Transformer encoder layers, the CTC output head, and the model's adapter module if present; the lower encoder stayed frozen. **Step 300**, selected by the lowest validation CER (14.45%), produced **15.84% CER, 67.16% ICU Khmer WER** on the 114 held-out utterances. Step 300 is a checkpoint during the 15-epoch run, not the number of epochs.

**Approach 3 — Qwen3-ASR 0.6B + student LoRA.** Starting checkpoint: `seanghay/Qwen3-ASR-0.6B-Khmer`, already adapted to Khmer by its publisher. The student run trained a rank-8 LoRA adapter on `q_proj` and `v_proj` with **16,720 effective training examples per epoch** for **3 completed epochs**. FLEURS training examples were repeated twice and additional public Khmer speech was added; 16,720 is not a unique-utterance count. **Epoch 1** had the lowest validation ICU WER: **12.42% CER, 27.99% WER** on 124 utterances. That saved adapter scored **13.71% CER, 33.40% WER** on the separate 114-utterance test. Validation WER was 28.74% at epoch 2 and 28.55% at epoch 3, so more training did not improve the selected WER.

**If asked for training settings.** The documented matched Whisper run used learning rate `1e-5`, three epochs, and gradient accumulation 8. MMS used `3e-5`, 15 epochs, gradient accumulation 8, a cosine schedule, and SpecAugment; it selected by validation CER. Qwen used `1e-5`, three epochs, gradient accumulation 8, AdamW, a cosine schedule with 5% warmup, and rank-8 `q_proj`/`v_proj` LoRA; it selected by full-validation ICU WER. These are settings for the presented runs, not a controlled hyperparameter sweep.

## Likely questions by slide

### Slides 1–3: problem and data

**1. What is automatic speech recognition?** It converts an audio recording of speech into written text. Here the input is a short Khmer speech recording and the output is a Khmer transcript.

**2. What is the actual user problem?** People need Khmer transcripts for captions, interviews, and notes. Manual transcription takes repeated listening, typing, and proofreading. I measure whether an adapted model provides a better first transcript; I did not measure how much time users save.

**3. What is an utterance?** One short audio recording paired with its reference transcript. In this project, an utterance is the unit in the training, validation, or test split.

**4. Why FLEURS Khmer?** It provides Khmer audio with reference transcripts and separate official splits. I saved exact row indices so the approaches could be evaluated on the same 114 held-out utterances.

**5. What do training, validation, and test sets do?** Training updates parameters. Validation guides checkpoint choice without using the final test. The held-out test estimates error after the choice is fixed. The 124 validation utterances and 114 test utterances are different recordings.

**6. What does “16,720 effective examples” mean?** It is the number of entries in Qwen's per-epoch training list, not the number of unique recordings. It includes 1,663 eligible FLEURS training clips repeated twice plus 13,394 additional public Khmer clips. Qwen therefore had about 31.5 times as many effective entries per epoch as the 531-example Whisper and MMS runs.

**7. How was training data prepared?** The shared FLEURS manifest retained short clips of at most 15 seconds and labels within Whisper's token limit. The external Qwen corpus was filtered for usable Khmer labels, valid audio, duplicates, and exact normalized-transcript overlap with validation/test; audio was converted to mono 16 kHz. Exact-overlap filtering does not rule out every near duplicate or bad annotation.

### Slides 4–6: metrics and approaches

**8. What are CER and WER? Give the formula.** Both are `(substitutions + deletions + insertions) / number of reference units × 100%`. CER uses reference characters; WER uses reference word tokens. The project sums edits over the whole test corpus before dividing, so these are corpus error rates.

**9. Why use ICU Khmer word segmentation for WER?** Khmer spaces often separate phrases rather than every word. ICU supplies the same Khmer word-break rule for reference and prediction. Punctuation-only and whitespace segments are excluded. The chosen segmentation affects WER, so the policy must be reported.

**10. Why report CER as well as WER?** CER measures smaller spelling-level edits and avoids relying entirely on a word-break choice. WER shows whether complete Khmer word tokens match. They answer related but different questions; a low CER does not guarantee a low WER.

**11. Can WER be greater than 100%?** Yes. Insertions can make the number of word edits larger than the number of reference words. Whisper had 2,644 word edits over 2,488 reference words, giving 106.27% WER. This is an error rate, not “negative accuracy.”

**12. Did you train all three yourself or just download them?** I started from public pretrained checkpoints and adapted each: full fine-tuning for Whisper, partial fine-tuning for MMS, and a student-trained LoRA adapter for Qwen. I did not pretrain any of them from scratch. The Qwen starting checkpoint was already Khmer-adapted by its publisher.

**13. What is an epoch, an optimizer step, and a checkpoint?** An epoch is one pass through the run's training list. An optimizer step is one parameter update, possibly after several examples through gradient accumulation. A checkpoint stores a particular model state. MMS completed 15 epochs, but its best saved state was step 300 of the 1,005 total optimizer steps. At about 67 updates per epoch, step 300 falls during the fifth epoch.

**14. How do the model families differ?** Whisper and Qwen are generative speech-to-text models that decode text tokens. MMS uses an acoustic encoder and a CTC output that maps audio frames to token sequences. Their architectures and decoding are different, so CER/WER on the same references are the common outcome measures.

**15. What are full fine-tuning, partial fine-tuning, and LoRA?** Full fine-tuning updates all starting model weights. Partial fine-tuning freezes most weights and updates selected layers/head. LoRA freezes the base and learns small low-rank matrices added to chosen projections. In this Qwen run, the LoRA targets were `q_proj` and `v_proj` with rank 8.

**16. Why use LoRA for Qwen?** It makes a student adaptation practical by updating a small adapter instead of the full base checkpoint. The result still depends on the publisher's Khmer-adapted base and the larger training corpus; LoRA alone cannot be credited for the score difference.

**17. What loss did each approach train with?** Whisper and Qwen trained to predict transcript tokens with a generative token loss. The Qwen code masked the audio/prompt prefix so loss applied to target transcript tokens. MMS trained with CTC loss, which learns to align audio frames and text tokens without frame-by-frame labels.

### Slides 7–10: results and critical analysis

**18. What were the final shared-test results?** On the same 114 FLEURS Khmer utterances: Whisper-Tiny 86.76% CER / 106.27% WER; MMS-1B 15.84% / 67.16%; Qwen3-ASR LoRA 13.71% / 33.40%. Lower is better. These are error rates, not the percentage of fully correct utterances.

**19. Which checkpoint was selected, and by what rule?** MMS step 300 minimized validation CER (14.45%). Qwen epoch 1 minimized full-validation ICU WER (27.99%). Both choices were made using validation, before reporting the held-out test. The saved Whisper model followed its training run's validation-CER selection, but I do not quote an unsupported exact checkpoint step.

**20. Why choose Qwen epoch 1 if training continued for three epochs?** Its validation WER was lowest at epoch 1: 27.99%, compared with 28.74% and 28.55% later. Training loss decreased over the run, but the validation WER did not improve; more optimization of the training set did not yield a better validation score under this metric.

**21. Did Qwen meet the target?** No. The planned validation goal was WER below 25%. The selected epoch achieved 27.99% validation WER, 2.99 percentage points above that threshold; the separate test WER was 33.40%.

**22. Why is Whisper's result so poor?** The saved Whisper checkpoint made many edits on this test, so it is not reliable here. The experiment does not isolate a single cause. Limited data, training settings, and model capacity are possibilities to investigate, not proven explanations from these scores alone.

**23. How can MMS have 15.84% CER but 67.16% WER?** Character and word errors use different units. A few wrong characters or Khmer word boundaries can affect multiple whole-word matches. MMS made 1,671 word edits, including 605 insertions, under the documented ICU segmentation.

**24. What does Qwen's improvement mean?** It made 831 word edits, 840 fewer than MMS on the same 2,488 reference words, and had the lowest observed error of the presented systems. Qwen also used much more training data and a publisher-trained Khmer base, so the difference cannot be attributed to architecture or LoRA alone.

**25. What does the mixed-script result show?** Among MMS test references, the 26 containing Latin letters had 23.75% corpus CER, versus 13.03% on the other 88. That is an association in a small subgroup. Listening and a controlled error audit are needed before claiming the acoustic reason.

**26. Is this a fair comparison?** The test utterances and scoring policy are shared. Training data and starting checkpoints differ, especially for Qwen. I call it a shared-test practical comparison, not a controlled architecture experiment.

**27. Could data leakage explain Qwen's score?** The student training-data preparation removed exact normalized-transcript overlaps with validation/test and duplicate external audio, and the selected test was used after validation choice. These checks reduce leakage risk but cannot guarantee that near duplicates or annotation issues are absent. I also cannot fully audit the publisher's earlier training data for its Khmer base checkpoint.

**28. Can the 114-utterance test prove the model works on real interviews?** No. It measures this fixed FLEURS subset. Different microphones, speakers, background noise, speaking styles, and topics need additional evaluation.

**29. Are the model differences statistically significant?** This deck reports point estimates and edit counts, not a confidence interval or statistical test for the three presented systems. I can say Qwen had the lowest observed error on these 114 utterances, but not claim a population-wide or causal advantage.

### Slides 11–12 and the live demo

**30. What is the main conclusion?** Qwen3-ASR LoRA had the lowest observed shared-test error: 13.71% CER and 33.40% ICU Khmer WER. The <25% validation WER goal was missed. Unequal training data and different starting checkpoints limit causal conclusions.

**31. What would you do next?** Listen to difficult recordings and classify substitutions, deletions, insertions, names, numbers, and segmentation issues. Then run a matched-data experiment if the question is which architecture helps, and test on more realistic Khmer recordings if the question is deployment.

**32. What does the live Colab demo prove?** It shows that the saved epoch-1 adapter can produce a transcript from a new short recording. One example cannot establish CER or WER without an exact reference or replace the 114-utterance test. The Colab notebook checks the adapter hash and runs an embedded sample before a live upload.

**33. What are the sources and what was your contribution?** FLEURS supplies the audio/reference data. Whisper, MMS, and the publisher's Khmer Qwen are public starting checkpoints. My contribution is the documented adaptation, fixed split/scoring procedure, saved student checkpoints, evaluation, and interpretation. I should be able to explain every configuration and score I present.

## Five statements to keep precise

- Say **“three student-adapted pretrained checkpoints”**; none was trained from random initialization.
- Say **“16,720 effective training entries”**; this includes repeated FLEURS clips and is not a unique-audio count.
- Say **“MMS completed 15 epochs; validation selected step 300.”** A step number is not an epoch number.
- Say **“Qwen completed 3 epochs; validation selected epoch 1.”** Its 27.99% validation WER missed the <25% goal.
- Say **“lowest observed error on this shared test.”** The experiment does not isolate an architecture advantage or prove real-world deployment quality.

## Evidence to open if asked

- `slides/Khmer_ASR_problem_refined.pptx` — the current 12-slide presentation.
- `results/matched_fleurs_split.json` — exact 531/124/114 FLEURS row lists and eligibility rules.
- `results/whisper_matched_metrics.json` — Whisper completed three epochs.
- `results/mms_matched_best300_summary.json` and `results/mms_matched_learning_history.json` — MMS completion and selected step.
- `results/diagnostic/qwen_bigtrain_epoch1_evidence/run_config.json`, `history.json`, and `best_validation.json` — Qwen configuration and all three validation epochs.
- `results/matched_cer_wer_summary.json` and `results/diagnostic/qwen_bigtrain_heldout_epoch1_20261015.json` — final test rates and edit counts.

## Rapid self-test: answer these aloud without reading

1. Explain the user problem and the six-step project flow in one minute.
2. Give the 40-second answer to why you chose these three approaches.
3. State each run's completed epochs and the MMS and Qwen selected checkpoints.
4. Explain full fine-tuning, partial fine-tuning, LoRA, and CTC in plain language.
5. State the CER/WER formula and why Khmer WER uses ICU word breaks.
6. Explain why Qwen's 16,720 effective entries do not mean 16,720 unique utterances.
7. State the three test CER/WER pairs and explain why Qwen's lead is not a causal architecture result.
8. Explain why the live demo is separate from the held-out evaluation and give one honest next experiment.
