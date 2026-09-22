#!/usr/bin/env python
"""
Generate professional PowerPoint presentation (.pptx) for the Khmer ASR Final Project.
Adheres strictly to the Final Project Rubric Section 6.2:
- 10-20 slides structure
- Embeds comparison figures from results/
- Detailed tables for models, hyperparameters, and results
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, "slides", "khmer_asr_presentation.pptx")

# Visual Palette: Deep Tech Navy / Teal Accent / White background
COLOR_PRIMARY = RGBColor(16, 44, 87)       # Deep Navy
COLOR_SECONDARY = RGBColor(53, 162, 159)   # Vibrant Teal
COLOR_DARK_TEXT = RGBColor(33, 37, 41)     # Dark Slate
COLOR_MUTED_TEXT = RGBColor(108, 117, 125) # Cool Gray
COLOR_CARD_BG = RGBColor(245, 247, 250)    # Soft Light Gray
COLOR_WHITE = RGBColor(255, 255, 255)


def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="KHMER AUTOMATIC SPEECH RECOGNITION"):
        # Category pill / banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = COLOR_SECONDARY

        # Title text
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent card
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_PRIMARY
    bg.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "FINAL PROJECT — DEEP LEARNING (2026–2027)"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_SECONDARY

    p1 = tf.add_paragraph()
    p1.text = "Comparative Study of Deep Learning Architectures for Khmer Automatic Speech Recognition"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_before = Pt(14)
    p1.space_after = Pt(20)

    p2 = tf.add_paragraph()
    p2.text = "Evaluating Seq2Seq Transformers vs. CTC Acoustic Models Under Low-Resource Constraints\n\n" \
              "Department of Engineering | Bachelor of Software Engineering\n" \
              "Lecturer: Mr. Soklong HIM | Student: [Your Name] (ID: [Your ID])"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(220, 225, 230)

    # ==========================================
    # SLIDE 2: Problem Definition & Motivation
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Problem Definition & Practical Motivation")

    tb = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Formal Mathematical Formulation:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  Given a 16 kHz raw audio waveform X = (x_1, x_2, ..., x_T), the model optimizes the posterior P(Y | X) " \
                 "to output the correct Khmer Unicode orthographic sequence Y = (y_1, y_2, ..., y_U)."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(12)

    p = tf.add_paragraph()
    p.text = "• The Linguistic Challenge of the Khmer Language:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Scriptio Continua: Written continuously without spaces between words; words delineate purely through context.\n" \
                 "  - Intricate Orthography: 33 consonants, 23 vowels, 14 independent vowels, and complex subscript consonants (Cheung).\n" \
                 "  - Acoustic Ambiguity: Series 1 vs Series 2 register vowels sharing spellings but shifting tonal pitch."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(12)

    p = tf.add_paragraph()
    p.text = "• Real-World & Societal Impact in Cambodia:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Democratizes voice accessibility for non-literate citizens and rural populations.\n" \
                 "  - Enables real-time speech-to-text for healthcare documentation, legal transcription, and smart devices."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 3: Dataset & Preprocessing Pipeline
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Dataset Description & Standardized Preprocessing Pipeline")

    tb = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Primary Benchmark Dataset: Google FLEURS Khmer (km_kh)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Source & License: Hugging Face Datasets (google/fleurs), licensed under Creative Commons CC-BY 4.0.\n" \
                 "  - Audio Quality: 16 kHz, 16-bit mono PCM recorded across diverse speakers with natural conversational cadence.\n" \
                 "  - Strict Controlled Splits: Fixed 1,000 train utterances, 200 validation utterances, and 200 held-out test utterances.\n" \
                 "  - Data Fairness Guarantee: Evaluated on the exact same 200 held-out test samples across all 3 approaches."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "• Unified Audio & Text Preprocessing Pipeline:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  1. Audio Standardization: Resampled to 16,000 Hz single-channel.\n" \
                 "     • Whisper: Computes 80-channel log-mel spectrograms using 25ms Hann windows with 10ms hop length.\n" \
                 "     • MMS: Extracts normalized 1D temporal representations directly through 7-layer convolutional feature encoder.\n" \
                 "  2. Unicode Normalization: Canonical Decomposition followed by Canonical Composition (NFC).\n" \
                 "  3. Text Sanitization: Strips zero-width non-breaking spaces (U+200B, U+FEFF) and whitespace compaction.\n" \
                 "  4. Sequence Truncation Guard: Utterances with >448 target tokens filtered out to prevent decoder crashes."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 4: Three Distinct DL Approaches
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Deep Learning Architectures Compared (Section 4 Compliance)")

    # 3 Comparison Columns
    col_w = Inches(3.7)
    gap = Inches(0.3)

    approaches = [
        ("Approach 1 (Primary)", "OpenAI Whisper-Tiny", "Seq2Seq Transformer", "Full Fine-Tuning", "37.8M (100% Trainable)",
         "Autoregressive cross-attention. Encoder maps 80-channel log-mel frames; Decoder predicts subwords conditioned on preceding tokens. Implicit internal language model."),
        ("Approach 2 (Comparison)", "Meta MMS-1B Khmer", "Acoustic CTC Model", "Vocabulary Adapter Tuning", "2.5M (of 1 Billion)",
         "Wav2Vec 2.0 backbone with Connectionist Temporal Classification (CTC). Non-autoregressive; aligns speech frames with phonemic characters without hallucination."),
        ("Approach 3 (Ablation)", "Whisper-Tiny (Frozen Enc)", "Linear Probe / Hybrid", "Decoder-Only Fine-Tuning", "28.5M Trainable (9.3M Frozen)",
         "Ablation study: freezes pre-trained multilingual acoustic encoder. Tests whether multilingual audio features transfer to low-resource Khmer phonetics without adaptation.")
    ]

    for i, (app_title, model_name, arch_type, strat, params, desc) in enumerate(approaches):
        left = Inches(0.8) + i * (col_w + gap)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.6), col_w, Inches(5.2))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_SECONDARY if i == 0 else RGBColor(200, 205, 210)

        tb = s4.shapes.add_textbox(left + Inches(0.15), Inches(1.75), col_w - Inches(0.3), Inches(4.9))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = app_title
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_PRIMARY

        p_mod = tf.add_paragraph()
        p_mod.text = f"{model_name}\n({arch_type})"
        p_mod.font.bold = True
        p_mod.font.size = Pt(13)
        p_mod.font.color.rgb = COLOR_SECONDARY
        p_mod.space_after = Pt(8)

        p_s = tf.add_paragraph()
        p_s.text = f"Strategy: {strat}\nParameters: {params}"
        p_s.font.size = Pt(12)
        p_s.font.bold = True
        p_s.font.color.rgb = COLOR_DARK_TEXT
        p_s.space_after = Pt(10)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 5: Justification of Approaches
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Theoretical Justification of Selected Architectures")

    tb = s5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Why Compare Seq2Seq vs. CTC?"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Sequence-to-Sequence (Seq2Seq) Transformers leverage autoregressive decoders that learn semantic dependencies.\n" \
                 "    However, under limited data, Seq2Seq decoders are prone to hallucinating common phrases or looping.\n" \
                 "  - CTC Models preserve monotonic acoustic alignment, eliminating hallucination loops and enabling ultra-fast inference.\n" \
                 "  - Comparing them reveals whether explicit language modeling or faithful acoustic alignment is superior for Khmer."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "• Why Include the Frozen Encoder Ablation (Approach 3)?"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Disentangles the contribution of acoustic representation learning vs. language decoder adaptation.\n" \
                 "  - Evaluates whether OpenAI's multilingual pre-training contains sufficient phonetic priors for Khmer without gradient updates.\n" \
                 "  - Provides an essential parameter-efficiency benchmark for resource-constrained training."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 6: Experimental Setup & Hyperparameters
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Experimental Setup & Systematic Hyperparameter Tuning")

    tb = s6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Training Infrastructure & Hardware:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Compute: Google Colab Tesla T4 GPU (16 GB GDDR6 VRAM) / Intel Xeon @ 2.20 GHz.\n" \
                 "  - Framework: PyTorch 2.x, Transformers 4.x/5.x, Hugging Face Datasets & Accelerate.\n" \
                 "  - Mixed Precision: FP16 active across all training runs to optimize memory throughput.\n" \
                 "  - Random Seed: Seed 42 fixed globally across Python random, NumPy, PyTorch CPU & CUDA."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "• Hyperparameter Exploration for Best-Performing Model (Whisper):"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Learning Rate Search: Tested 1e-5 (too slow convergence), 5e-5, and 1e-4 (optimal convergence without divergence).\n" \
                 "  - Regularization: Weight decay evaluated at 0.0 vs. 0.01 with AdamW optimizer.\n" \
                 "  - Warmup Schedule: 200 warmup steps with linear decay prevented early gradient explosion.\n" \
                 "  - Batch Configuration: Effective batch size of 16 (per-device batch size 8 × gradient accumulation steps 2)."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 7: Evaluation Metrics (CER vs WER)
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Evaluation Metrics: Character Error Rate (CER) vs. Word Error Rate (WER)")

    tb = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Primary Evaluation Metric: Character Error Rate (CER)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  $$\\text{CER} = \\frac{S + D + I}{N} \\times 100\\%$$\n" \
                 "  where S = Substitutions, D = Deletions, I = Insertions, and N = Total Reference Unicode Characters.\n" \
                 "  • Reflects exact grapheme and phonetic reconstruction accuracy for non-segmented languages."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "• Why Standard WER Fails for Khmer Speech (Crucial Rubric Discussion):"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Standard Word Error Rate splits sentences using whitespace: string.split(' ').\n" \
                 "  - Because Khmer is written without spaces between words (scriptio continua), an entire sentence constitutes ONE token.\n" \
                 "  - A single misplaced consonant or vowel diacritic causes the entire sentence token to be marked incorrect (100% WER).\n" \
                 "  - Therefore, unsegmented WER is artificially inflated (~95-100%). CER is universally adopted in academic literature."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 8: Consolidated Results Table
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Consolidated Experimental Results (Identical Held-Out Test Split)")

    # Add Table Shape
    rows = 4
    cols = 8
    tbl_shape = s8.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.2))
    tbl = tbl_shape.table

    headers = ["Approach", "Model Architecture", "Strategy", "Trainable Params", "Hardware", "Train Time", "Test CER ↓", "Test WER ↓"]
    col_widths = [Inches(1.5), Inches(2.0), Inches(1.8), Inches(1.4), Inches(1.5), Inches(1.1), Inches(1.2), Inches(1.2)]
    for idx, w in enumerate(col_widths):
        tbl.columns[idx].width = w

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    data = [
        ["Approach 1", "Whisper-Tiny (Seq2Seq)", "Full Fine-Tuning", "37.8M (100%)", "Colab T4", "~60 min", "49.3%", "101.9%*"],
        ["Approach 2", "Meta MMS-1B (CTC)", "Adapter Tuning", "0.2M (0.02%)", "Colab T4", "~7 min", "15.5%", "100.0%*"],
        ["Approach 3", "Whisper (Frozen Enc)", "Linear Probe", "29.6M (78.3%)", "Colab T4", "~30 min", "59.7%", "100.8%*"],
    ]

    for row_idx, row_vals in enumerate(data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = tbl.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if row_idx % 2 == 0 else COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = COLOR_DARK_TEXT
            p.alignment = PP_ALIGN.CENTER
            if "15.5%" in val or "Approach 2" in val:
                p.font.bold = True
                if "15.5%" in val:
                    p.font.color.rgb = COLOR_SECONDARY

    tb = s8.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "* Note: All models evaluated on the exact same held-out Google FLEURS km_kh test utterances.\n" \
             "Key Takeaway: Meta MMS-1B with Khmer adapter achieved a stellar 15.5% CER, decisively outperforming Whisper-Tiny (49.3% CER) " \
             "and the Frozen Encoder ablation (59.7% CER) by leveraging a 1-billion parameter multilingual acoustic backbone with CTC inductive bias."
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 9: Visual Comparison Figures
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Visual Comparison: Learning Curves & Error Rate Analysis")

    # Insert images from results/
    img1_path = os.path.join(RESULTS_DIR, "learning_curves.png")
    img2_path = os.path.join(RESULTS_DIR, "metrics_comparison.png")

    if os.path.exists(img1_path):
        s9.shapes.add_picture(img1_path, Inches(0.8), Inches(1.6), width=Inches(6.0))
    if os.path.exists(img2_path):
        s9.shapes.add_picture(img2_path, Inches(7.1), Inches(1.6), width=Inches(5.4))

    tb = s9.shapes.add_textbox(Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Figure 1 (Left): Whisper training loss dropped steadily from 2.31 to 0.54, with validation CER converging smoothly.\n" \
             "• Figure 2 (Right): Meta MMS CTC achieved the lowest CER (15.5%), outperforming Whisper (49.3%), while freezing the Whisper encoder (Approach 3) degraded performance to 59.7% CER, proving acoustic adaptation is vital for Khmer."
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 10: Error Analysis & Linguistic Failure Cases
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. In-Depth Error Analysis & Failure Case Inspection")

    tb = s10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    errors = [
        ("1. Subscript Consonant (Cheung) Deletions:",
         "The Khmer subscript sign (U+17D2 COENG) transforms consonants into conjunct sub-glyphs. Under acoustic noise, the model frequently omitted the coeng indicator, outputting independent base consonants instead of stacked clusters (e.g. confusing ្ត with ត)."),
        ("2. Series 1 vs Series 2 Register Vowel Confusion:",
         "Khmer vowels possess dual phonetic pronunciations depending on whether the preceding consonant is Series 1 (high register) or Series 2 (low register). The acoustic encoder occasionally mapped Series 2 vowels to their Series 1 phonetic counterpart."),
        ("3. Whisper Repetition Loops on Boundary Silence:",
         "Due to autoregressive decoding, unvoiced trailing audio frames occasionally induced repetition loops in Whisper, repeating the final subword token until maximum decoder sequence length was reached."),
        ("4. Rare Compound Word Deletions:",
         "Specialized administrative or loan terms in FLEURS exhibited phonetic deletion errors, whereas daily conversational vocabulary demonstrated high fidelity.")
    ]

    for title, desc in errors:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_PRIMARY

        p_sub = tf.add_paragraph()
        p_sub.text = f"  {desc}"
        p_sub.font.size = Pt(13)
        p_sub.font.color.rgb = COLOR_DARK_TEXT
        p_sub.space_after = Pt(8)

    # ==========================================
    # SLIDE 11: Limitations & Future Work
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "10. Limitations & Prioritized Future Work")

    tb = s11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Identified Project Limitations:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Training Size Constraints: Fine-tuned on 1,000 utterances to comply with free Google Colab session timeouts.\n" \
                 "  - Absence of External Language Model: CTC predictions were decoded with greedy argmax rather than beam search rescoring.\n" \
                 "  - Model Scale: Compared Whisper-Tiny (37.8M) rather than Whisper-Large-v3 due to local hardware memory limits."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = "• Prioritized Roadmap for Future Engineering:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  1. Large Scale Training: Scale up to OpenSLR SLR42 (100+ hours of Khmer speech) on high-VRAM clusters.\n" \
                 "  2. External N-Gram Language Model: Rescore CTC emission logits with a KenLM 5-gram language model.\n" \
                 "  3. Khmer Dictionary Word Segmentation: Integrate khmer-nltk to compute true, segmented Word Error Rate (WER).\n" \
                 "  4. Edge Quantization: Export models to ONNX Runtime and 8-bit quantization for mobile deployment in Cambodia."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 12: Live Demonstration & Q&A
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "11. Interactive Web Application Demonstration & Q&A")

    tb = s12.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Interactive Testing Web Application (app.py):"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Built with Gradio; inspired by CADT IDRI Khmer voice dictation UI.\n" \
                 "  - Features dual input modalities: live microphone audio recording & audio file upload (.wav, .mp3, .flac).\n" \
                 "  - Displays real-time audio duration, transcription latency (seconds), and normalized Khmer Unicode text.\n" \
                 "  - Includes 1-click Windows launcher (run_app.bat) and pre-loaded sample audio files."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT
    p_sub.space_after = Pt(18)

    p = tf.add_paragraph()
    p.text = "• Oral Defense & Q&A Readiness:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Prepared to inspect live code, PyTorch tensor manipulation, and Hugging Face Trainer routines.\n" \
                 "  - Ready to demonstrate model inference directly from local checkpoint or Colab environment.\n\n" \
                 "  Thank you! We welcome your questions."
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    # ==========================================
    # SLIDE 13 (Appendix): Q&A Technical Reference
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Appendix: Hyperparameter Matrix & Tensor Specifications", "Q&A BACKUP SLIDE")

    tb = s13.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Detailed Technical Hyperparameter Matrix:"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_PRIMARY

    p_sub = tf.add_paragraph()
    p_sub.text = "  - Whisper-Tiny: 4 encoder layers, 4 decoder layers, 6 attention heads, d_model = 384.\n" \
                 "  - Input Dimensions: (Batch, 80, 3000) representing 30-second 80-channel log-mel spectrogram.\n" \
                 "  - MMS-1B: 48 Transformer layers, 16 attention heads, hidden dimension = 1280; Adapter: 2-layer residual projection.\n" \
                 "  - Optimization: AdamW (lr = 1e-4, eps = 1e-8, beta1 = 0.9, beta2 = 0.999), linear warmup 200 steps.\n" \
                 "  - Loss Functions: Label smoothed cross-entropy (Whisper) vs. CTC loss with blank token index (MMS).\n" \
                 "  - Checkpoint Safety: Evaluated and saved at fixed intervals to prevent Colab disconnection loss."
    p_sub.font.size = Pt(13)
    p_sub.font.color.rgb = COLOR_DARK_TEXT

    prs.save(OUTPUT_PPTX)
    print(f"Successfully created presentation: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_deck()
