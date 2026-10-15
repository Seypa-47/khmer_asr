"""Khmer decoding for the publisher's fine-tune and its student adapters."""
import numpy as np
import torch
from qwen_asr.inference.utils import parse_asr_output, split_audio_into_chunks


def decode_batch(wrapper, waveforms, beams=3, force_khmer=True, fallback=True, max_new_tokens=512):
    """Decode audio, retrying unfinished beam outputs individually with greedy search."""
    prompt = wrapper._build_text_prompt('', 'Khmer' if force_khmer else None)
    inputs = wrapper.processor(text=[prompt]*len(waveforms), audio=waveforms, return_tensors='pt', padding=True)
    inputs = inputs.to(wrapper.model.device).to(wrapper.model.dtype)
    with torch.inference_mode():
        output = wrapper.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=beams, do_sample=False)
    tokens = output.sequences[:, inputs['input_ids'].shape[1]:]
    decoded = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rows = []
    for i, (text, ids) in enumerate(zip(decoded, tokens)):
        has_eos = bool(torch.isin(ids, ids.new_tensor([151645, 151643])).any())
        row = {'hypothesis': parse_asr_output(text, user_language='Khmer' if force_khmer else None)[1],
               'has_eos': has_eos, 'primary_has_eos': has_eos, 'greedy_fallback': False,
               'generated_tokens_including_padding': int(ids.numel())}
        if fallback and beams > 1 and not has_eos:
            retry = decode_batch(wrapper, [waveforms[i]], beams=1, force_khmer=force_khmer,
                                 fallback=False, max_new_tokens=max_new_tokens)[0]
            row.update(retry, primary_has_eos=False, greedy_fallback=True,
                       primary_hypothesis=row['hypothesis'])
        rows.append(row)
    return rows


def transcribe_khmer(wrapper, waveform, sample_rate=16000, beams=3):
    """Use explicit Khmer decoding and bounded chunks on the laptop GPU.

    The upstream language validator omits Khmer, so build the fine-tune's
    supported language prefix directly. No reference text is supplied.
    """
    wav = np.ascontiguousarray(waveform, dtype=np.float32)
    if sample_rate != 16000 or wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
        raise ValueError('Qwen needs nonempty, finite, mono 16 kHz audio')
    chunks = split_audio_into_chunks(wav=wav, sr=sample_rate, max_chunk_sec=30)
    texts = []
    for chunk, _ in chunks:
        texts.append(decode_batch(wrapper, [chunk], beams=beams)[0]['hypothesis'])
    return ' '.join(texts).strip()
