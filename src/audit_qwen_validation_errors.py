"""Reproduce error diagnostics without modifying references or selecting test rows."""
import argparse
import collections
import hashlib
import json
import re
import zipfile
from pathlib import Path

from jiwer import process_words
from score_matched_cer_wer import word_tokens
from score_qwen_rows import score_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-arrow-dir', type=Path)
    args = ap.parse_args()
    source = Path('selected_adapter_and_results(1).zip')
    with zipfile.ZipFile(source) as archive:
        prediction = json.loads(archive.read('validation_predictions.json'))
        history = json.loads(archive.read('history.json'))
    metrics = score_rows(prediction['examples'])
    assert metrics == prediction['metrics']
    manifest = json.loads(Path('results/diagnostic/qwen_train30_split_20261007.json').read_text(encoding='utf-8'))
    assert [r['validation_index'] for r in prediction['examples']] == manifest['indices']['validation']
    baseline = json.loads(Path('results/diagnostic/qwen_base_bf16_guarded_validation_20261007.json').read_text(encoding='utf-8'))
    baseline_by_index = {r['validation_index']: r for r in baseline['examples']}
    totals = collections.Counter()
    pairs = collections.Counter()
    rows = []
    for row in prediction['examples']:
        reference = word_tokens(row['reference'])
        hypothesis = word_tokens(row['hypothesis'])
        edits = process_words(' '.join(reference), ' '.join(hypothesis))
        counts = {k: getattr(edits, k) for k in ('substitutions', 'deletions', 'insertions', 'hits')}
        totals.update(counts)
        alignment = []
        for chunk in edits.alignments[0]:
            if chunk.type == 'equal':
                continue
            ref = ' '.join(reference[chunk.ref_start_idx:chunk.ref_end_idx])
            hyp = ' '.join(hypothesis[chunk.hyp_start_idx:chunk.hyp_end_idx])
            pairs[(chunk.type, ref, hyp)] += 1
            alignment.append({'type': chunk.type, 'reference': ref, 'hypothesis': hyp})
        base = baseline_by_index[row['validation_index']]
        assert base['reference'] == row['reference']
        rows.append({**row, **counts, 'word_errors': edits.substitutions+edits.deletions+edits.insertions,
                     'reference_words': len(reference), 'word_edits': alignment,
                     'publisher_hypothesis': base['hypothesis'],
                     'audio': f"local_eval/qwen_fleurs_audio_expanded/validation_{row['validation_index']}.wav",
                     'listening_review_status': 'not_reviewed'})
    groups = {}
    for name, predicate in (
        ('latin_in_reference', lambda r: bool(re.search('[A-Za-z]', r['reference']))),
        ('digits_in_reference', lambda r: bool(re.search('[0-9]', r['reference']))),
        ('neither_latin_nor_digits', lambda r: not re.search('[A-Za-z0-9]', r['reference'])),
    ):
        selected = [r for r in rows if predicate(r)]
        groups[name] = score_rows(selected) if selected else None
    target_errors = (metrics['wer_reference_words']-1)//5
    report = {
        'purpose': 'validation-only error diagnosis; training strategy selection',
        'test_used': False, 'references_modified': False,
        'source_zip': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'verified_metrics': metrics, 'history': history, 'word_edit_totals': dict(totals),
        'max_word_errors_for_strictly_below_20': target_errors,
        'minimum_errors_to_remove': metrics['wer_errors']-target_errors,
        'cohorts': groups, 'cohort_note': 'Latin and digit cohorts overlap; descriptive association, not causal attribution.',
        'frequent_edits': [{'type': t, 'reference': r, 'hypothesis': h, 'occurrences': n}
                           for (t, r, h), n in pairs.most_common(40)],
        'review_queue': sorted(rows, key=lambda r: r['word_errors'], reverse=True),
        'limitations': ['No listening review has been completed.',
                       'Reference-based edits are diagnostic and must not become validation-specific correction rules.',
                       'Train loss fell while validation WER rose; extra epochs under this recipe did not help.'],
    }
    if args.train_arrow_dir:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        texts = []
        for path in sorted(args.train_arrow_dir.glob('fleurs-train-*.arrow')):
            with pa.memory_map(str(path), 'r') as mapped:
                texts.extend(ipc.open_stream(mapped).read_all().column('transcription').to_pylist())
        assert len(texts) > max(manifest['indices']['train']), 'Incomplete training metadata'
        training = [texts[i] for i in manifest['indices']['train']]
        vocabulary = collections.Counter(w for text in training for w in word_tokens(text))
        coverage = collections.Counter()
        for row in rows:
            reference = word_tokens(row['reference'])
            edits = process_words(' '.join(reference), ' '.join(word_tokens(row['hypothesis'])))
            for chunk in edits.alignments[0]:
                for word in reference[chunk.ref_start_idx:chunk.ref_end_idx]:
                    coverage[('seen' if word in vocabulary else 'unseen') +
                             ('_correct' if chunk.type == 'equal' else '_error')] += 1
        report['train_vocabulary_diagnostic'] = {
            'training_clips': len(training), 'unique_transcripts': len(set(training)),
            'unique_word_tokens': len(vocabulary), 'reference_token_alignment_counts': dict(coverage),
            'training_clips_with_latin': sum(bool(re.search('[A-Za-z]', t)) for t in training),
            'training_clips_with_digits': sum(bool(re.search('[0-9]', t)) for t in training),
            'limitations': 'ICU vocabulary coverage is an association, not an acoustic diagnosis; insertions have no reference token.'}
    out = Path('results/diagnostic/qwen_error_audit_20261009.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('verified_metrics', 'word_edit_totals', 'minimum_errors_to_remove', 'cohorts')}, indent=2))


if __name__ == '__main__':
    main()
