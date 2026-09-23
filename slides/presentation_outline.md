# Khmer ASR final presentation outline

**Format:** 13 slides, designed for a 10-minute presentation plus Q&A.  
**Student:** `[Student Name / Student ID]`

## 1. Title

Comparative Study of Deep Learning Approaches for Khmer Automatic Speech Recognition. Add the student name and ID before submission.

## 2. Problem and motivation

Input: 16 kHz Khmer speech. Output: Khmer Unicode transcript. Explain why transcription is useful and why Khmer orthography makes evaluation and text normalization important.

## 3. Dataset and split

Google FLEURS `km_kh`; current dataset version has 1,675 train, 326 validation, and 771 test examples. The original Whisper run's 354 updates imply about 941 retained training rows, and its validation throughput matches 189 retained rows from the first 200; the original row manifest was not saved. The MMS code requested the first 1,000 training and 200 validation rows, then filtered long audio, without logging retained counts. Both checkpoints were scored on the first 200 test rows. The runs did not use identical training selections and filters.

## 4. Models compared

Whisper-Tiny uses a Seq2Seq Transformer and was fully fine-tuned. Meta MMS-1B uses Wav2Vec 2.0 with CTC; its saved run unfroze the top four encoder layers and CTC head/adapter, about 79.1M trainable parameters. The frozen-encoder ablation is excluded because its checkpoint and metrics are unavailable.

## 5. Why compare Seq2Seq and CTC?

Seq2Seq predicts tokens autoregressively and conditions on previous text. CTC uses monotonic frame-to-token alignment and greedy decoding. These design differences motivate comparison; they do not by themselves explain the measured score gap.

## 6. Saved training setup

Whisper: 3 epochs, learning rate `1e-5`, 500 warmup steps, batch size 8, gradient accumulation 1. MMS: 15 epochs, learning rate `5e-5`, 50 warmup steps, batch size 1, gradient accumulation 8, top four layers unfrozen, SpecAugment enabled. Training was reported on a Colab T4; exact runtime was not retained. The repository does not retain a complete learning-rate/regularization sweep.

## 7. Metric

Report CER after NFC normalization, removal of U+200B/U+FEFF, and whitespace removal. Standard WER is not the primary metric because Khmer whitespace is not a word tokenizer.

## 8. Results

Show the generated table. Original Whisper scored 83.89% CER; the 35-step-warmup rerun scored 83.67% CER; MMS scored 15.96% CER on the first 200 test rows. The Whisper improvement is 0.22 percentage points, with 92 examples improved and 84 worsened. State that the saved architecture runs used different training selections and filters, so this is diagnostic.

## 9. Curves and comparison figure

Show the saved Whisper loss/validation-CER history and the shared-test CER chart. Explain that MMS trainer history was not retained, so no MMS learning curve is available.

## 10. Error analysis

The 200 Whisper predictions contain 13,424 substitutions, 8,741 deletions, and 286 insertions. Several high-error examples contain repeated output tokens. Show one or two examples from `results/error_analysis.md`. Do not attribute these to a specific acoustic cause without listening to the clips.

## 11. Limitations and next steps

Train both models on the same seeded 1,000-example subset, save both trainer states and per-example predictions, run and record a learning-rate and regularization sweep, and publish an accessible Whisper download. Rerun the frozen-encoder ablation only if time permits.

## 12. Demo and Q&A

Demonstrate the Gradio app with one sample clip. Be prepared to explain FLEURS selection, text normalization, Whisper decoding, CTC decoding, and the limitations of the comparison.

## 13. Q&A appendix

Show saved run parameters, trainable parameter counts, and references: Whisper (Radford et al., 2023), MMS (Pratap et al., 2023), FLEURS (Conneau et al., 2023), CTC (Graves et al., 2006), and Transformers (Wolf et al., 2020).
