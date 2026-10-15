"""Score JSON reference/hypothesis rows from stdin using the fixed ICU policy."""
import json
import sys

import icu
from jiwer import process_characters, process_words

from score_matched_cer_wer import cer_text, word_tokens


def score_rows(rows):
    if icu.ICU_VERSION != '77.1':
        raise RuntimeError(f'Experiments require ICU 77.1; found {icu.ICU_VERSION}')
    chars = process_characters([cer_text(r['reference']) for r in rows],
                               [cer_text(r['hypothesis']) for r in rows])
    words = process_words([' '.join(word_tokens(r['reference'])) for r in rows],
                          [' '.join(word_tokens(r['hypothesis'])) for r in rows])
    return {'evaluated_examples': len(rows), 'cer_percent': 100 * chars.cer,
            'wer_percent_icu_segmented': 100 * words.wer,
            'cer_reference_characters': chars.hits + chars.substitutions + chars.deletions,
            'wer_reference_words': words.hits + words.substitutions + words.deletions,
            'cer_errors': chars.substitutions + chars.deletions + chars.insertions,
            'wer_errors': words.substitutions + words.deletions + words.insertions,
            'icu_version': icu.ICU_VERSION}


if __name__ == '__main__':
    print(json.dumps(score_rows(json.load(sys.stdin))))
