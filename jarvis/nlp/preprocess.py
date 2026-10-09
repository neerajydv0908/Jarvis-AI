"""Text normalisation with NLTK (roadmap step 3).

Pipeline used for intent classification::

    raw text
      -> clean            lower-case, expand contractions, "p.m." -> "pm", "5pm" -> "5 pm"
      -> word_tokenize    NLTK Treebank-style tokens
      -> filter           drop punctuation, map numbers to <num>
      -> stop words       NLTK's English list, minus words that carry intent
                          (question words, you/your/me/my, negations, ...)
      -> lemmatise        WordNet lemmatiser driven by NLTK POS tags
                          (or Porter stemming with method="stem")

The same pipeline turns training patterns and live commands into tokens, so
the bag-of-words features always line up.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Dict, List, Optional

import nltk

log = logging.getLogger(__name__)

UNK = "<unk>"   # any word the model has never seen
NUM = "<num>"   # any number ("10", "5:30", "ten")
SLOT = "xxslot"  # placeholder used by templates in intents.json; always becomes <unk>

# NLTK renamed some resources in 3.9 (punkt -> punkt_tab, ..._tagger -> ..._tagger_eng);
# fetching both keeps old and new NLTK versions happy.
NLTK_RESOURCES: Dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

# Stop words that must survive filtering because they decide the intent:
# "who are you" vs "how are you", "what is your name", "turn off", "clear all ..."
KEEP_WORDS = {
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "i", "me", "my", "myself", "you", "your", "yours", "yourself",
    "no", "nor", "not", "off", "up", "all", "about", "can", "do", "does", "did",
}

# Filler words that never help classify an intent.
EXTRA_STOP_WORDS = {"please", "kindly", "um", "uh", "hmm"}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred",
}

_NUMBER_RE = re.compile(r"^\d+(?:[:.]\d+)?(?:st|nd|rd|th)?$")

# Ordered list of (pattern, replacement); applied after lower-casing.
_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"\blet's\b"), "let us"),
    (re.compile(r"\b(\w+)n't\b"), r"\1 not"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\b(\w+)'re\b"), r"\1 are"),
    (re.compile(r"\b(\w+)'ll\b"), r"\1 will"),
    (re.compile(r"\b(\w+)'ve\b"), r"\1 have"),
    (re.compile(r"\b(\w+)'d\b"), r"\1 would"),
    (re.compile(r"\b(what|where|who|how|when|why|that|there|here|it|he|she)'s\b"), r"\1 is"),
    # the same words typed without an apostrophe ("whats the time")
    (re.compile(r"\b(what|where|who|how|when|why|that|there)s\b"), r"\1 is"),
]


def _resource_present(path: str) -> bool:
    # The downloader leaves some corpora (wordnet, omw) zipped; newer NLTK
    # versions only find those under their ".zip" name.
    for candidate in (path, path + ".zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def missing_nltk_data() -> List[str]:
    return [name for name, path in NLTK_RESOURCES.items() if not _resource_present(path)]


def ensure_nltk_data(quiet: bool = True) -> List[str]:
    """Download any NLTK resources that are missing. Returns the ones it fetched."""
    fetched = []
    for name in missing_nltk_data():
        log.info("Downloading NLTK resource %s", name)
        if nltk.download(name, quiet=quiet):
            fetched.append(name)
    return fetched


def clean_text(text: str) -> str:
    """Lower-case and regularise a raw utterance (no tokenisation yet)."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    # "p.m." / "p. m." / "a.m" -> "pm" / "am"
    text = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", text)
    # "5pm" -> "5 pm", "10:30am" -> "10:30 am"
    text = re.sub(r"(\d)(am|pm)\b", r"\1 \2", text)
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def strip_wake_word(text: str, wake_word: str = "jarvis") -> str:
    """Remove the wake word where it only addresses Jarvis (start or end of the command).

    "hey jarvis, what's the weather in chicago" -> "what's the weather in chicago"
    "what time is it jarvis"                    -> "what time is it"
    """
    wake = re.escape(wake_word.lower())
    text = re.sub(rf"^\s*(?:(?:hey|ok|okay|yo|oh)[\s,]+)?{wake}\b[\s,.!?]*", "", text, flags=re.I)
    text = re.sub(rf"[\s,]+{wake}[\s.!?]*$", "", text, flags=re.I)
    return text.strip()


def normalize_wake_word(text: str, wake_word: str = "jarvis") -> str:
    """Version of a command for the intent classifier.

    A leading wake word is dropped; any other mention means "you":
    "hey jarvis, what time is it" -> "what time is it"
    "thanks jarvis"                -> "thanks you"
    "what does jarvis stand for"   -> "what does you stand for"
    """
    wake = re.escape(wake_word.lower())
    text = re.sub(rf"^\s*(?:(?:hey|ok|okay|yo|oh)[\s,]+)?{wake}\b[\s,.!?]*", "", text, flags=re.I)
    text = re.sub(rf"\b{wake}(?:'s)?\b", "you", text, flags=re.I)
    return text.strip()


def _wordnet_pos(treebank_tag: str) -> str:
    from nltk.corpus import wordnet

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


class TextPreprocessor:
    """Tokenise -> remove stop words -> lemmatise (or stem), using NLTK."""

    def __init__(self, method: str = "lemma", remove_stopwords: bool = True, auto_download: bool = True):
        if method not in {"lemma", "stem"}:
            raise ValueError("method must be 'lemma' or 'stem'")
        self.method = method
        self.remove_stopwords = remove_stopwords
        if auto_download:
            ensure_nltk_data()
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        self._lemmatizer = WordNetLemmatizer()
        self._stemmer = PorterStemmer()
        try:  # touch every resource once so a missing one fails here, not mid-conversation
            nltk.word_tokenize("checking the tokenizer")
            self.stop_words = (set(stopwords.words("english")) - KEEP_WORDS) | EXTRA_STOP_WORDS
            self._lemmatizer.lemmatize("checks")
        except LookupError as exc:
            raise RuntimeError(
                "NLTK data is missing (" + ", ".join(missing_nltk_data()) + "). "
                "Run `python scripts/setup_nlp.py` once with an internet connection."
            ) from exc

    # -- public API ----------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(clean_text(text))

    def sentences(self, text: str) -> List[str]:
        """Sentence-split a paragraph (used to keep spoken answers short)."""
        return nltk.sent_tokenize(text)

    def normalize(self, text: str) -> List[str]:
        """Full pipeline: returns the list of normalised tokens for ``text``."""
        tokens = [t for t in self.tokenize(text) if not self._is_punct(t)]
        if not tokens:
            return []
        tagged = self._pos_tag(tokens) if self.method == "lemma" else [(t, "") for t in tokens]
        out: List[str] = []
        for token, tag in tagged:
            if token == SLOT:
                out.append(UNK)
            elif _NUMBER_RE.match(token) or token in NUMBER_WORDS:
                out.append(NUM)
            elif self.remove_stopwords and token in self.stop_words:
                continue
            elif self.method == "lemma":
                out.append(self._lemmatizer.lemmatize(token, _wordnet_pos(tag)))
            else:
                out.append(self._stemmer.stem(token))
        return out

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _is_punct(token: str) -> bool:
        return token in {"'s", "'", "``", "''", "--", "..."} or all(ch in string.punctuation for ch in token)

    @staticmethod
    def _pos_tag(tokens: List[str]):
        try:
            return nltk.pos_tag(tokens)
        except LookupError:  # tagger data missing: lemmatise everything as nouns
            return [(t, "NN") for t in tokens]


_default: Optional[TextPreprocessor] = None


def default_preprocessor() -> TextPreprocessor:
    """Shared instance (loading WordNet once is noticeably faster)."""
    global _default
    if _default is None:
        _default = TextPreprocessor()
    return _default
