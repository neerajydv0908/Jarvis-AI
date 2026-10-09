"""Load the Cornell Movie-Dialogs Corpus as (prompt, response) pairs.

Supports both distributions of the corpus:

* ConvoKit's ``movie-corpus`` (``utterances.jsonl`` - the current official download)
* the original release (``movie_lines.txt`` + ``movie_conversations.txt``)
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

DOWNLOAD_URLS = [
    # current official source (ConvoKit); listed in ConvoKit's download_config.json
    "http://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip",
    # original 2011 release (may no longer be online)
    "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
]
SEPARATOR = " +++$+++ "

Pair = Tuple[str, str]


def clean_line(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)          # stray <u>/<i> tags in the scripts
    text = text.replace("--", " ").replace("\u0092", "'")
    return re.sub(r"\s+", " ", text).strip()


def _keep(text: str, min_words: int, max_words: int) -> bool:
    words = text.split()
    return min_words <= len(words) <= max_words and any(ch.isalpha() for ch in text)


# -- original release ------------------------------------------------------------------
def load_original(folder: Path, min_words: int = 1, max_words: int = 30) -> List[Pair]:
    lines: Dict[str, str] = {}
    with open(folder / "movie_lines.txt", encoding="iso-8859-1") as fh:
        for raw in fh:
            parts = raw.rstrip("\n").split(SEPARATOR)
            if len(parts) == 5:
                lines[parts[0].strip()] = clean_line(parts[4])
    pairs: List[Pair] = []
    with open(folder / "movie_conversations.txt", encoding="iso-8859-1") as fh:
        for raw in fh:
            parts = raw.rstrip("\n").split(SEPARATOR)
            if len(parts) != 4:
                continue
            try:
                ids = ast.literal_eval(parts[3].strip())
            except (ValueError, SyntaxError):
                continue
            for first, second in zip(ids, ids[1:]):
                prompt, response = lines.get(first, ""), lines.get(second, "")
                if _keep(prompt, min_words, max_words) and _keep(response, min_words, max_words):
                    pairs.append((prompt, response))
    return pairs


# -- ConvoKit release ----------------------------------------------------------------------
def load_convokit(folder: Path, min_words: int = 1, max_words: int = 30) -> List[Pair]:
    texts: Dict[str, str] = {}
    replies: List[Tuple[str, str]] = []
    with open(folder / "utterances.jsonl", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            utt = json.loads(raw)
            texts[utt["id"]] = clean_line(utt.get("text") or "")
            parent = utt.get("reply-to") or utt.get("reply_to")  # ConvoKit writes "reply-to"
            if parent:
                replies.append((parent, utt["id"]))
    pairs = []
    for parent, child in replies:
        prompt, response = texts.get(parent, ""), texts.get(child, "")
        if _keep(prompt, min_words, max_words) and _keep(response, min_words, max_words):
            pairs.append((prompt, response))
    return pairs


# -- discovery / download --------------------------------------------------------------------
def find_corpus(root: Path) -> Optional[Tuple[str, Path]]:
    """Return ('convokit' | 'original', folder) for the first corpus found under ``root``."""
    root = Path(root)
    if not root.exists():
        return None
    for path in sorted(root.rglob("utterances.jsonl")):
        return "convokit", path.parent
    for path in sorted(root.rglob("movie_lines.txt")):
        if (path.parent / "movie_conversations.txt").exists():
            return "original", path.parent
    return None


def load_pairs(root: Path, min_words: int = 1, max_words: int = 30) -> List[Pair]:
    found = find_corpus(root)
    if not found:
        raise FileNotFoundError(f"No Cornell Movie-Dialogs corpus found under {root}")
    kind, folder = found
    loader = load_convokit if kind == "convokit" else load_original
    pairs = loader(folder, min_words, max_words)
    if not pairs:
        raise ValueError(f"Found the {kind} corpus in {folder} but could not read any dialogue pairs from it.")
    log.info("Loaded %d pairs from the %s corpus in %s", len(pairs), kind, folder)
    return pairs


def download(dest: Path, urls: Iterable[str] = DOWNLOAD_URLS) -> Path:
    """Download and unzip the corpus into ``dest``. Returns ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        archive = dest / Path(url).name
        try:
            print(f"Downloading {url} ...")
            with urllib.request.urlopen(url, timeout=60) as response, open(archive, "wb") as out:
                shutil.copyfileobj(response, out)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            archive.unlink(missing_ok=True)
            if find_corpus(dest):
                return dest
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
            archive.unlink(missing_ok=True)
    raise RuntimeError("Could not download the corpus:\n  " + "\n  ".join(errors))
