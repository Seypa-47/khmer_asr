# Saved Whisper prediction error analysis

Evaluated 200 examples from the first 200 rows of the FLEURS `km_kh` test split. CER ignores whitespace after NFC and zero-width-character cleanup.

- Character error rate: 83.89%
- Substitutions: 13,424
- Deletions: 8,741
- Insertions: 286
- Reference characters: 26,763

## Highest-error examples

| FLEURS test row | Example CER | Reference | Whisper output |
|---:|---:|---|---|
| 194 | 108.0% | នេះហៅថាកម្រិតphនៃសារធាតុគីមីអ្នកអាចបង្កើតសូចនាករមួយដោយប្រើទឹកស្ពៃខ្ដោបក្រហម | ន្នៅពីន្រប្រប់ប្រប់ប់ពីប់ប្រប់ប់ប្រប់ប់ប្រប់ប្រប់ប់ប្រប់ប់ប្រប់ប្រប់ប់ប្រប់ប់ប់ប្រប់ប់ប់ប់ប់ប� |
| 170 | 101.3% | នៅចន្លោះម៉ោង10:00-11:00យប់mdtអ្នកទោសនៅទីធ្លាខាងក្រោយក៏បានដុតភ្លើងឱ្យឆេះឡើង។ | បារបស្របារប់ប្រប់ប់ប្រប់ប់ប់ប្រប់ប់ប្រប់ប្រប់ប្រប់ប់ប់ប្រប់ប់ប្រប់ប់ប្រប់ប្រប់ប្រប់ប់ |
| 34 | 101.1% | ដោយមានទីតាំងនៅលើកំពូលភ្នំមួយនៅភាគខាងជើងនៃទីក្រុងmeccaល្អាងនេះនៅដាច់ឆ្ងាយពីពិភពលោកយ៉ាងខ្លាំង | ប្រុងប្រប្រប់បាន្រប់ពីប្រប់ប់ពីប្រប់ប់ពីប្រប់ប់ប់ប្រប់ប់ប់ប្រប់ប់ប់ប្រប់ប់ប់ប្រប់ប្រប់ប់ប់ប់ប់� |
| 66 | 98.3% | បទដ្ឋាន802.11nដំណើរការទាំងនៅលើប្រេកង់2.4ghzនិងប្រកង់5.0ghz | បានបានបានបានបានបានបានបានបានបានបានបានបានបានបានបានបានបានបានបានបានបាន |
| 167 | 97.2% | ទោះយ៉ាងណាក៏ដោយនោះមិនមែនជាការពិតនោះទេទោះបីជាមានការសរសេរនៅផ្នែកខាងក្រោយនៃឯកសារក៏ដោយក៏វាមិនមែនជាផែនទីកំណប់ដែរ | ប្រប់ប់ប់ប់ប់ប់ប់ប់ប់ប់បីបីបបីបបីបបីបបបបីបបីបបបីបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបបប |

These examples document the observed text differences. CER alone cannot establish whether a specific mismatch came from coeng clusters, vowel register, noise, or decoding. Listen to the corresponding public FLEURS clips before assigning an acoustic cause. MMS per-example predictions were not retained, so this error analysis covers Whisper only.
