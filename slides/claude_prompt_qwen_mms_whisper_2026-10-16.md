# Prompt for Claude Code

Create a polished PowerPoint for my 10-minute Deep Learning final-project presentation on October 16, 2026. Work in `D:\khmer_asr` and save the finished deck as `D:\khmer_asr\slides\khmer_asr_qwen_mms_whisper_2026-10-16.pptx`. Also save a concise rehearsal script and five likely lecturer Q&A answers in `D:\khmer_asr\slides\presentation_script_2026-10-16.md`.

My chosen three models are exactly:

1. **Approach 1: Whisper-Tiny Khmer**, the matched full fine-tuning checkpoint.
2. **Approach 2: Meta MMS-1B Khmer CTC**, with the top four Transformer layers and CTC head tuned.
3. **Approach 3: Qwen3-ASR 0.6B Khmer**, the big-corpus LoRA adapter trained on 16,720 effective clips, selected at epoch 1.

Read `README.md`, `results/README.md`, `results/matched_cer_wer_three_approaches.json`, `results/diagnostic/qwen_bigtrain_heldout_epoch1_20261015.json`, and the relevant notebooks before building the deck. The older final three-approach comparison includes a head-only MMS variant that I am **not** presenting as one of these three. The existing chart `results/matched_cer_wer_three_approaches.png` also contains that variant, so create a new chart for my chosen lineup. You may use the existing PowerPoint as a visual reference, but verify its claims before reusing them.

Use these exact held-out scores on the same 114 FLEURS Khmer test clips:

- Whisper-Tiny: **86.76% CER, 106.27% ICU Khmer WER**.
- MMS-1B top-four-layer tuning: **15.84% CER, 67.16% ICU Khmer WER**.
- Qwen3-ASR big-corpus LoRA: **13.71% CER, 33.40% ICU Khmer WER** using BF16 inference.

Qwen was selected on the fixed 124-clip validation set at **12.42% CER / 27.99% WER**. Keep validation and held-out test scores clearly separated. Whisper and MMS used the fixed 531-training-clip manifest; Qwen used substantially more training data, including external public speech. They share the held-out test clips and scoring policy, but this is **not** a controlled comparison of architecture alone. Do not claim that Qwen met the newer validation WER <25% goal. Do not call `1 − CER` general “accuracy,” and do not invent listening judgments, training runs, scores, or causal explanations.

Make 10–12 slides: title; problem and motivation; FLEURS data and evaluation policy; three model designs; training setup and the data-size difference; checkpoint selection; a clear CER/WER results chart; observed errors and limitations; conclusion and realistic next steps; references and AI-assistance disclosure. Explain that Khmer WER uses ICU word segmentation rather than spaces. Keep slide text short, visuals readable, and presenter notes detailed enough for a 10-minute talk. Include the student name Khemrak Pasey and lecturer Mr. Soklong HIM.

Render and inspect every slide for overflow, font size, and Khmer text display. Verify all numbers and model labels against the saved evidence before delivering the PowerPoint and rehearsal script.
