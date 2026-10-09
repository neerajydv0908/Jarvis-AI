"""Question answering from Wikipedia (answers 'who is ...', 'what is ...', 'tell me about ...').

Two calls to the public MediaWiki API, no key required:
  1. full-text search to turn a spoken topic into the best article title
  2. the plain-text intro of that article, cut to its first sentences with NLTK
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional

import requests

log = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "JARVIS-personal-assistant/1.0 (educational project; Python requests)"


class KnowledgeError(Exception):
    """Raised with a message that is safe to speak to the user."""


def _default_sentence_splitter(text: str) -> List[str]:
    try:
        import nltk

        return nltk.sent_tokenize(text)
    except LookupError:
        return re.split(r"(?<=[.!?])\s+", text)


def clean_for_speech(text: str) -> str:
    """Remove parentheticals (pronunciations, dates) and citation marks so it reads aloud well."""
    previous = None
    while previous != text:  # nested parentheses
        previous = text
        text = re.sub(r"\s*\([^()]*\)", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Wikipedia:
    def __init__(self, sentences: int = 2, timeout: float = 8.0, session: Optional[requests.Session] = None,
                 sentence_splitter: Callable[[str], List[str]] = _default_sentence_splitter):
        self.sentences = sentences
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.split = sentence_splitter

    def _query(self, **params) -> dict:
        try:
            response = self.session.get(API_URL, params={"format": "json", "action": "query", **params},
                                        timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Wikipedia request failed: %s", exc)
            raise KnowledgeError("I couldn't reach Wikipedia right now.") from exc

    def search(self, query: str, limit: int = 3) -> List[str]:
        data = self._query(list="search", srsearch=query, srlimit=limit)
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    def intro(self, title: str) -> str:
        data = self._query(prop="extracts", exintro=1, explaintext=1, redirects=1, titles=title)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "") or ""
        return ""

    def answer(self, query: str) -> str:
        titles = self.search(query)
        if not titles:
            raise KnowledgeError(f"I couldn't find anything about {query}.")
        for title in titles:
            extract = clean_for_speech(self.intro(title))
            if not extract or "may refer to" in extract[:200]:
                continue  # skip empty and disambiguation pages
            sentences = [s for s in self.split(extract) if s.strip()]
            return " ".join(sentences[: self.sentences])
        raise KnowledgeError(f"I found several meanings for {query}. Could you be more specific?")
