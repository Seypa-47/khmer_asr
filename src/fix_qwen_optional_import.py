"""Defer Qwen 0.0.6's Japanese tokenizer import until Japanese alignment is used.

Run with the app's Python after installing qwen-asr. Khmer transcription does
not need nagisa/DyNet, whose native extension can be blocked on Windows.
This changes only the import location; Windows security settings are untouched.
"""

from importlib.metadata import distribution
from pathlib import Path


def main():
    package = distribution("qwen-asr")
    path = Path(package.locate_file("qwen_asr/inference/qwen3_forced_aligner.py"))
    source = path.read_text(encoding="utf-8")
    original = "    def tokenize_japanese(self, text: str) -> List[str]:\n        words = nagisa.tagging(text).words"
    replacement = "    def tokenize_japanese(self, text: str) -> List[str]:\n        import nagisa\n\n        words = nagisa.tagging(text).words"
    if replacement in source and "\nimport nagisa\n" not in source:
        print("Qwen optional Japanese import is already deferred.")
        return
    if source.count("\nimport nagisa\n") != 1 or source.count(original) != 1:
        raise RuntimeError("Qwen source differs from the supported layout; no files changed.")
    backup = path.with_suffix(".py.before_lazy_nagisa")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    source = source.replace("\nimport nagisa\n", "\n", 1).replace(original, replacement, 1)
    path.write_text(source, encoding="utf-8")
    print(f"Deferred the optional Japanese tokenizer import in {path}")


if __name__ == "__main__":
    main()
