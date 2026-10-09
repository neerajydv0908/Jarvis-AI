"""One-time download of the NLP data: NLTK corpora/tokenizers and the spaCy English model.

    python scripts/setup_nlp.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jarvis.nlp.preprocess import ensure_nltk_data, missing_nltk_data  # noqa: E402

SPACY_MODEL = "en_core_web_sm"


def main() -> int:
    print("Downloading NLTK data (tokenizers, stop words, WordNet, POS tagger)...")
    ensure_nltk_data(quiet=True)
    missing = missing_nltk_data()
    if missing:
        print(f"  Still missing: {missing}. If you are behind a proxy, NLTK 3.10+ refuses proxied downloads "
              "unless you set NLTK_ALLOW_PROXIED_URLOPEN=1.")
    else:
        print("  NLTK data ready.")

    try:
        import spacy

        spacy.load(SPACY_MODEL)
        print(f"spaCy model {SPACY_MODEL} already installed.")
    except OSError:
        print(f"Downloading spaCy model {SPACY_MODEL}...")
        result = subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
        if result.returncode != 0:
            print("  spaCy download failed; run it yourself: python -m spacy download en_core_web_sm")
            return 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
