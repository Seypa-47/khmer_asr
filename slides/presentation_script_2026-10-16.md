# Khmer ASR presentation rehearsal script

Presentation date: 16 October 2026. Target length: about 10 minutes.

## Slide 1

Good morning. I am Khemrak Pasey. This project transcribes Khmer speech into Khmer Unicode text. I will compare three trained model checkpoints: Whisper-Tiny, Meta MMS-1B, and a Qwen3-ASR LoRA model. I will explain the data and scoring policy first, then show the held-out results and their limits.

## Slide 2

The input is a single Khmer speech waveform sampled at 16 kilohertz. The output is a Khmer Unicode transcript. Speech recognition can fail at the acoustic, spelling, or text-generation stage, so I report measured edit errors rather than subjective accuracy on a few examples. A model that seems fluent on a demo clip can still make many word errors across a held-out test set.

## Slide 3

FLEURS supplies the Khmer km_kh speech data. Whisper and MMS used the same saved row manifest: 531 training clips, 124 validation clips, and 114 held-out test clips. The Qwen LoRA run used 16,720 effective training clips, including public data beyond that 531-clip selection. All three are scored on the same 114 held-out FLEURS clips. This gives a common test set, but it does not isolate architecture as the reason for performance differences.

## Slide 4

CER is the number of character substitutions, deletions, and insertions divided by the number of reference characters. We normalize Unicode and remove whitespace and zero-width marks for CER. Khmer spaces usually separate phrases rather than every word, so whitespace-based WER would misrepresent word errors. We use ICU Khmer word breaks for both references and predictions, then compute corpus-level edit distance. A WER over 100 percent can happen when there are enough insertions.

## Slide 5

Whisper-Tiny is an encoder-decoder Transformer that generates text. The project fully fine-tuned its checkpoint. Meta MMS-1B uses a Wav2Vec 2.0 speech encoder with a CTC output; the selected run tuned the top four Transformer layers and the CTC head or adapter. Qwen3-ASR 0.6B is a speech-to-text generation model; the student trained a LoRA adapter on a larger public corpus. These are the three checkpoints I am presenting today. The older head-only MMS ablation remains in the repository but is not part of this lineup.

## Slide 6

Whisper and MMS trained on the same 531 selected FLEURS training clips. The matched Whisper run completed three epochs. The MMS run tuned its top four Transformer layers and selected checkpoint step 300 by validation CER. The latest Qwen run used 16,720 effective clips with LoRA and selected epoch 1 based on the lowest validation WER. This difference in training data is a major limitation when comparing the models. The observed result reflects both model choice and training recipe.

## Slide 7

Validation is for model selection; the held-out test is for the final reported score. For Qwen, epoch 1 was the selected checkpoint because it had the lowest WER on the 124 validation clips. Its validation scores were 12.42 percent CER and 27.99 percent WER. After selecting it, the project ran one local held-out evaluation on the fixed 114 test clips. All three test scores use the same CER and ICU Khmer WER policy. We should not mix validation and test numbers in one ranking.

## Slide 8

This is the main result. On the same 114 held-out Khmer FLEURS clips, the Qwen LoRA checkpoint scored 13.71 percent CER and 33.40 percent word error rate. MMS-1B scored 15.84 percent CER and 67.16 percent WER. Whisper-Tiny scored 86.76 percent CER and 106.27 percent WER. Lower values mean fewer edit errors. The Qwen model had the lowest observed error on this test set, but it also had far more training examples, so the chart does not establish a pure architectural advantage.

## Slide 9

The Qwen checkpoint has the best observed scores in this selected lineup. The newer validation goal was WER below 25 percent, but the selected Qwen checkpoint reached 27.99 percent, so it missed that goal. All three held-out test WER values are above 20 percent. Two MMS and Qwen test CER values are below 20 percent; Whisper is not. The model comparison is useful descriptively, but Qwen had a much larger training corpus and a different training recipe, so the results do not isolate architecture as the cause.

## Slide 10

There are four main limits. First, Qwen used substantially more training data than Whisper and MMS. Second, the reported test scores measure a fixed set of 114 public FLEURS clips, which is a small slice of possible speakers, microphones, and topics. Third, Khmer word error rates depend on a consistent segmentation policy; the project uses ICU for both references and predictions. Fourth, the local demo can transcribe personal audio, but without a verified exact spoken reference we should not assign those recordings a score.

## Slide 11

The main takeaway is measured progress, with a clear remaining gap. Qwen3-ASR gave the lowest character and word errors among the three selected models on the fixed held-out test clips. Its CER was 13.71 percent and WER was 33.40 percent. The word-error target remains unmet. A next experiment should use a fixed evaluation policy, audit audio and transcript alignment, and change one justified training factor at a time. I can now show a short local demo, while keeping demo impressions separate from the measured test results.

## Slide 12

The data source is FLEURS. Whisper and MMS are described in their original research papers. The Qwen checkpoint is based on a published Khmer ASR model with a student-trained LoRA adapter. The project README discloses AI assistance in review, debugging, analysis, documentation, and presentation preparation. The student should verify and understand the code and all scores before presenting.

## Likely lecturer questions

**Why is Khmer WER different from space-separated WER?** Khmer spaces often mark phrases. This project uses ICU Khmer word boundaries for both references and predictions.

**Is Qwen architecturally better than MMS?** Qwen had lower observed error on the shared test clips, but it also used much more training data. This experiment cannot isolate architecture.

**Did any model meet the WER target?** No. Qwen validation WER was 27.99%, above the newer <25% goal. Its held-out test WER was 33.40%.

**Why can Whisper WER exceed 100%?** WER divides all word substitutions, deletions, and insertions by the number of reference words. Many insertions can push it above 100%.

**Can the demo prove real-world accuracy?** No. A demo shows an output for a recording. Scoring requires a verified exact spoken reference and a representative test set.
