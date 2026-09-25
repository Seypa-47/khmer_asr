# Khmer Automatic Speech Recognition

Individual Deep Learning final project comparing a fine-tuned Whisper-Tiny Seq2Seq model with a Meta MMS CTC acoustic model on Khmer speech.

**Course:** Deep Learning Final Project, 2026-2027  
**Lecturer:** Mr. Soklong HIM  
**Student:** Khemrak Pasey

> **Experiment status:** Historical checkpoints tested on the first 200 FLEURS test rows remain diagnostic because their training selections differed. A controlled rerun used one saved 531/124/114 train/validation/test manifest for both architectures. The final matched comparison includes both CER and Khmer-segmented WER on all 114 held-out clips. The professor approved the topic and, according to the student, expects performance above 80%, but has not specified how that target is scored or certified these results.

## Problem

Given a 16 kHz Khmer speech waveform, predict its Khmer Unicode transcript. The final comparison reports both character error rate (CER) and word error rate (WER) on the same held-out clips. Khmer spaces mainly mark phrase boundaries, so WER uses ICU's Khmer word-break dictionary on both references and predictions. Lower error rates are better. The metric policy and exact edit counts are saved in `results/matched_cer_wer_summary.json`.

## Data

The project uses the public Google FLEURS `km_kh` configuration from [Hugging Face Datasets](https://huggingface.co/datasets/google/fleurs). The dataset card identifies the FLEURS source and licensing terms. The downloaded version in this workspace contains 1,675 training, 326 validation, and 771 test examples. FLEURS supplies official splits; no speaker-level or class distribution applies to this transcription task. Audio is mono at 16 kHz. Known limits include the small amount of Khmer speech relative to high-resource languages and variation in speaker, recording, and topic.

The original Whisper run did not save its exact selected row indices. Its trainer state records 354 updates over three epochs with batch size 8, consistent with 941 usable training rows. Applying its documented label filter to a seed-42 shuffled 1,000-row FLEURS selection leaves exactly 941 rows. Its validation throughput similarly matches 189 retained rows from the first 200. These are strong inferences, not a preserved split manifest. The saved MMS training code selected the first 1,000 training rows without shuffling and removed clips over 10 seconds; its actual retained row count and training history were not saved. The two runs therefore cannot be treated as trained on the same examples.

## Approaches

| Approach | Architecture | Saved training strategy |
|---|---|---|
| Whisper-Tiny | Encoder-decoder Transformer with cross-attention | Full fine-tuning |
| Meta MMS-1B Khmer | Wav2Vec 2.0 with CTC output | Top four Transformer layers plus CTC head/adapter unfrozen; SpecAugment enabled |

The MMS run is **not adapter-only tuning**. The saved MMS weights contain about 964.8M parameters, with approximately 79.1M trainable under the recorded top-four-layer training configuration. The frozen-encoder Whisper ablation is excluded because this workspace has no saved checkpoint or evaluation metrics for it.

The original Whisper training arguments show 3 epochs, learning rate `1e-5`, 500 warmup steps, batch size 8, and gradient accumulation 1. It ended after 354 updates, before warmup finished. The MMS arguments show 15 epochs, learning rate `5e-5`, 50 warmup steps, batch size 1, and gradient accumulation 8. A separate Whisper rerun with 35 warmup steps is documented below. The repository still does not retain a learning-rate and regularization sweep.

## Final controlled comparison

Both approaches used the same saved 531/124/114 FLEURS train/validation/test manifest. CER uses NFC text with U+200B/U+FEFF and whitespace removed. WER uses NFC text, removes U+200B/U+FEFF, applies [ICU `km_KH` dictionary word breaks](https://unicode-org.github.io/icu/userguide/boundaryanalysis/), and excludes punctuation-only segments. The same policy is applied to both models. The previously logged raw whitespace WER for Whisper is not comparable to this Khmer-segmented WER.

| Fine-tuned approach | Held-out test clips | Test CER | Khmer-segmented test WER |
|---|---:|---:|---:|
| Whisper-Tiny encoder-decoder | 114 | 86.76% | 106.27% |
| Meta MMS-1B encoder with CTC, best validation checkpoint | 114 | **15.84%** | **67.16%** |

Both rates divide edit operations by the reference length, measured in characters or ICU-segmented words. A WER above 100% is possible when a model inserts extra words. Whisper's earlier Trainer log reported 86.93% CER from decoded labels; the 86.76% above comes from rescoring its saved checkpoint against the same raw FLEURS references used for MMS. These are test-set error rates, not sentence accuracy or verified scores on personal recordings. Exact edit counts and prediction pairs are in [`results/`](results/README.md). Large checkpoints are linked below.

The lecturer evaluates both CER and WER. If the reported above-80% target applies to each metric as an error-rate threshold, each must be below 20%; the MMS checkpoint meets that threshold for CER but not WER. The completed MMS run selected its best checkpoint using validation CER. Its old Trainer `eval_wer` used whitespace and is not the Khmer-segmented WER above. The updated training script logs `eval_cer` and `eval_wer_icu` for future runs and can select a checkpoint using `--selection-metric wer_icu`; this change does not alter the completed checkpoint or its reported scores. Because training labels omit spaces, the validation WER uses word breaks on decoded labels, while final test WER uses the original FLEURS references. Keep this distinction in mind when comparing their values.

![Matched CER and Khmer-segmented WER](results/matched_cer_wer.png)

## Earlier diagnostic runs

All listed checkpoints were scored on the same first 200 examples of the FLEURS test split. CER applies NFC normalization, removes U+200B and U+FEFF, and removes whitespace before scoring. The 35-step-warmup Whisper rerun used the seed-42 shuffled first 1,000 training rows (941 retained after label filtering) and first 200 validation rows (189 retained).

| Approach | Trainable parameters | Training evidence | Epochs | Test CER (first 200) |
|---|---:|---|---:|---:|
| Original Whisper-Tiny | 37.8M | About 941 / 189 retained, inferred from run logs | 3 | **83.89%** |
| Whisper-Tiny, 35-step warmup rerun | 37.8M | 941 train / 189 validation retained | 3 | **83.67%** |
| Meta MMS-1B CTC, top four layers unfrozen | 79.1M | 1,000 train / 200 validation requested; retained counts unlogged | 15 | **15.96%** |

The Whisper rerun saved only 59 fewer character edits over 26,763 reference characters than the original: 22,392 versus 22,451. Among 200 clips, 92 improved, 84 worsened, and 24 tied. A paired utterance bootstrap interval for the CER improvement includes zero (`-0.49` to `+0.92` percentage points), so the observed `0.22`-point gain is too small to establish a reliable improvement. The result files retain both sets of Whisper predictions and the comparison details. MMS still has the much lower saved CER, while the differing training selections limit any architecture claim.

### Figures and learning-curve evidence

![Saved Whisper learning history](results/diagnostic/learning_curves.png)

![Shared-test CER diagnostic](results/diagnostic/metrics_comparison.png)

The original Whisper trainer state contains training loss and validation logs. Its historical validation CER retained spaces, while the post-hoc test CER removes whitespace; the two CER series therefore use different text policies. The warmup rerun's trainer state is saved in `results/diagnostic/whisper_warmup35_trainer_state.json`, with validation CER of 88.54%, 86.71%, and 83.62% after epochs one through three. MMS trainer history was not retained, so its learning curve cannot be reconstructed. The current figure shows the original Whisper history only.

`results/diagnostic/error_analysis.md` reports 13,424 substitutions, 8,741 deletions, and 286 insertions across the 200 Whisper examples, plus five high-error examples. Several outputs show repeated-token decoding failures. The report avoids assigning an acoustic cause without listening to each audio clip. MMS per-example predictions were not saved for this earlier run, so this diagnostic error analysis covers Whisper only.

## Training and evaluation

### Install

Use Python 3.10 or 3.11 with a CUDA GPU for training the MMS-1B model. Install the listed packages with:

```bash
pip install -r requirements.txt
```

The controlled rerun is complete. Use `notebooks/khmer_asr_experiments.ipynb` on a Google Colab GPU to reproduce it. The commands below prepare the same saved row list for both models. The historical scores above remain diagnostic.

### Matched rerun

```bash
# One shared list of eligible official FLEURS rows for both models.
python src/matched_fleurs.py --output results/matched_fleurs_split.json \
  --seed 42 --train-candidates 1000 --validation-candidates 200 \
  --test-start 200 --test-candidates 200 --max-duration-seconds 15

# Whisper-Tiny: train, validate, and test on the shared row list.
python src/finetune_whisper.py \
  --output-dir ./models/whisper-tiny-khmer-matched \
  --model-name openai/whisper-tiny --use-fleurs-train --skip-ddd \
  --split-manifest results/matched_fleurs_split.json \
  --metrics-output results/whisper_matched_metrics.json \
  --trainer-state-output results/whisper_matched_trainer_state.json \
  --seed 42 --num-train-epochs 3 --learning-rate 1e-5 --weight-decay 0 \
  --warmup-steps 35 --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 4 --gradient-accumulation-steps 8 \
  --eval-steps 67 --save-steps 67 --fp16

# MMS CTC: use the identical row list and a new checkpoint folder.
python src/train_mms.py \
  --output-dir ./models/mms-khmer-ctc-matched \
  --model-id facebook/mms-1b-all --target-lang khm \
  --split-manifest results/matched_fleurs_split.json \
  --metrics-output results/mms_matched_metrics.json \
  --trainer-state-output results/mms_matched_trainer_state.json \
  --seed 42 --num-train-epochs 15 --learning-rate 3e-5 --weight-decay 0 \
  --unfreeze-top-layers 4 --apply-spec-augment --lr-scheduler-type cosine \
  --warmup-steps 50 --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 --gradient-accumulation-steps 8 --fp16

# After both runs: save every prediction and create figures from the new evidence.
python src/evaluate_saved_whisper.py --model-dir models/whisper-tiny-khmer-matched \
  --split-manifest results/matched_fleurs_split.json \
  --output results/whisper_matched_predictions.json --device cuda
python src/evaluate_saved_mms.py --model-dir models/mms-khmer-ctc-matched \
  --processor-id facebook/mms-1b-all \
  --split-manifest results/matched_fleurs_split.json \
  --output results/mms_matched_best300_predictions.json --device cuda
# Compare both error rates from the saved predictions on the identical clips.
python src/score_matched_cer_wer.py \
  --whisper results/whisper_matched_predictions.json \
  --mms results/mms_matched_best300_predictions.json \
  --output results/matched_cer_wer_summary.json
python src/plot_cer_wer.py
python src/plot_matched_results.py --completed
```

The manifest removes recordings over 15 seconds and examples exceeding Whisper's label limit before either model trains. It records the exact retained indices. The new held-out test candidate rows start at index 200 because the earlier work repeatedly inspected rows 0–199. The matched output folders preserve the older local checkpoints. The Colab notebook includes three short MMS pilot runs: baseline, a lower learning rate, and weight decay. Pilot runs use validation only and save no large weights; the selected 3e-5 learning rate and zero weight decay went into the full MMS run. `--no-apply-spec-augment` is also available for a later regularization comparison. Final checkpoints and evidence are in Google Drive as shown in the notebook.

### Completed matched run: verified test evidence

The September 23–24 Colab run used one saved 531/124/114 train/validation/test manifest for both architectures. The manifest in `results/matched_fleurs_split.json` was regenerated locally and its train, validation, and test index hashes match the Drive copy exactly. Whisper-Tiny completed 3 epochs; its Trainer log reported 86.93% test CER. Rescoring the saved checkpoint against the shared raw references gave **86.76% CER and 106.27% WER**. The 15-epoch MMS run was interrupted at step 800, then resumed and completed at step 1005. Its best validation checkpoint remained step 300 (14.45% validation CER); it reached **15.84% test CER and 67.16% WER**. The final-step validation CER was 14.70%, so the extra training did not improve the selected model. WER was computed later from saved predictions with ICU Khmer word segmentation. Neither result is a percentage of fully correct sentences.

See `results/matched_cer_wer_summary.json` and `results/mms_matched_best300_error_analysis.md` for exact counts and observed failure patterns. The 26 MMS references containing Latin letters scored 23.75% corpus CER, versus 13.03% for the other 88; this is a descriptive failure breakdown. Both sets of 114 public FLEURS reference/prediction pairs are retained in this repository and Drive. To repeat MMS inference, use `src/evaluate_saved_mms.py` with `--processor-id facebook/mms-1b-all`, the best checkpoint as `--model-dir`, and the shared split manifest. The lecturer's above-80% target needs a named metric: if it means CER below 20%, MMS meets it on FLEURS; if it means WER below 20%, neither model meets it under the stated segmentation policy.

![Matched held-out CER](results/matched_test_cer.png)

![MMS test error by reference script](results/mms_error_subgroups.png)

![MMS training and validation history](results/mms_matched_learning_curves.png)

These completed-run figures can be regenerated with `python src/plot_matched_results.py --completed`. The plotted history is a four-decimal copy of the full Trainer state in Drive. The Whisper trainer state also remains in Drive for a combined curve figure before submission.

### Private voice checks

Personal recordings and their outputs are kept outside Git. They do not have verified exact spoken references, so no accuracy percentage is claimed for them. For a private check, run `src/evaluate_saved_mms_user_audio.py` with `--audio-dir`, `--model-dir`, and `--output`; score only recordings with verified references using `src/score_user_audio.py`.

### Inference demo

Run `python app.py`, then open `http://127.0.0.1:7860`. The Windows launcher is `run_app.bat`. The app prefers `models/mms-khmer-ctc-matched` when installed; this workspace currently has that model file, so a newly started local demo loads the matched-run model. Other clones need to download the shared weights first. Saved model files are ignored by Git.

## Repository map

- `src/finetune_whisper.py`: Whisper training and evaluation pipeline.
- `src/train_mms.py`: MMS CTC training pipeline.
- `src/matched_fleurs.py`: Saves one fixed FLEURS row selection for both approaches.
- `src/evaluate_saved_whisper.py`: Auditable Whisper scoring on the first N test rows.
- `src/evaluate_saved_mms.py`: Auditable MMS scoring on the same rows.
- `src/score_matched_cer_wer.py`: Matched corpus CER and ICU Khmer-segmented WER from saved predictions.
- `src/plot_cer_wer.py`: CER and WER figure from the audited score summary.
- `src/evaluate_saved_mms_user_audio.py`: Raw checkpoint inference on private recordings.
- `src/plot_matched_results.py`: Figures from completed matched runs.
- `src/evaluate_and_plot.py`: Regenerates figures from the earlier diagnostic runs.
- `notebooks/khmer_asr_experiments.ipynb`: Colab workflow.
- `results/`: Final matched-run evidence and figures, with earlier runs in `results/diagnostic/`.
- `slides/khmer_asr_presentation_cer_wer.pptx`: 12-slide final project presentation in the student's chosen design.
- `app.py`: Gradio inference interface.

## Model weights

Large model files are excluded from Git by `.gitignore`. The matched-run MMS score comes from checkpoint-300 in the [shared MMS Google Drive folder](https://drive.google.com/drive/folders/1gU93D2CA4lgn8E480GMy-UZhDzvstvJM); the [matched Whisper checkpoint is shared separately](https://drive.google.com/drive/folders/1TOEafns7lSfMVkHpXhklCO_xSqZ_m9MZ). Link viewers have read access to both folders. The older diagnostic checkpoints are stored locally under `models/`. After training completed, Trainer loaded the best validation weights into the top-level Drive folder `mms-khmer-ctc-matched`, which also contains the processor files. To use the matched model in the app, download the top-level model and processor files from that folder into `models/mms-khmer-ctc-matched`; the `checkpoint-*` training-state subfolders are not needed for inference. Cloning this repository alone does not reproduce the saved-model demo. The app falls back to the public `facebook/mms-1b-all` base model if no local MMS folder exists. The older [Hugging Face Whisper page](https://huggingface.co/Seypa-47/whisper-tiny-khmer) returned HTTP 401 during this review; use the shared Drive folder for the matched Whisper weights.

## References

1. Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML.
2. Pratap, V. et al. (2023). *Scaling Speech Technology to 1,000+ Languages*. Meta AI Research.
3. Conneau, A. et al. (2023). *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*. IEEE SLT.
4. Graves, A. et al. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML.
5. Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP.

## AI assistance disclosure

**Tools used:** Antigravity AI Assistant and Codex AI assistant. **Scope:** project review, debugging support, evaluation/error-analysis code, and documentation/presentation edits. **Verification:** the student must personally verify the training runs, understand and be able to modify the submitted code, and add any other tools used. Training results are reported only when backed by saved logs or predictions.

## Submission readiness checklist

- [x] Topic approval: confirmed by the lecturer (per student).
- [x] Two distinct trained deep learning architectures are present.
- [x] Run both approaches using the same train, validation, and test subsets; both runs completed.
- [x] Save curves and trainer state for both approaches in Drive; the MMS curve and summary are also in this repository.
- [x] Retain prediction pairs and compute matched CER and Khmer-segmented WER for both approaches.
- [x] Run and document MMS learning-rate and weight-decay pilot comparisons on validation data.
- [x] Share the matched MMS and Whisper checkpoint folders through viewer links.
- [x] Add the student name to the project materials.
- [x] README, requirements, source code, results, and slides are present.
- [ ] Rehearse the 10-minute presentation/Q&A and confirm the professor's exact 80% metric definition.
