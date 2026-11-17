"""Normalización y tokenización sin dependencias pesadas (reproducible en cualquier máquina)."""
from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": " - "})

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | {
    "s", "t", "ll", "d", "re", "ve", "m", "don't", "i'm", "it's", "that's", "oh", "yes",
    "cont'd", "contd", "v.o", "o.s", "you've", "we've", "i've", "they've", "you're", "we're",
    "they're", "i'll", "you'll", "we'll", "he's", "she's", "there's", "what's", "let's", "can't",
    "won't", "isn't", "aren't", "wasn't", "didn't", "doesn't", "couldn't", "wouldn't", "shan't", "i'd",
}


def normalize(text: str) -> str:
    """NFKC (resuelve ligaduras ﬁ/ﬂ), comillas tipográficas → ASCII, espacios colapsados."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str, *, remove_stopwords: bool = True, min_len: int = 2) -> list[str]:
    toks = _TOKEN.findall(strip_accents(normalize(text)).lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [t for t in toks if len(t) >= min_len]


def canonical_speaker(name: str) -> str:
    """'GALADRIEL (V.O.) (CONT'D)' -> 'GALADRIEL';  'Frodo ' -> 'FRODO'."""
    name = strip_accents(normalize(name))
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^A-Za-z' \-]", " ", name)
    return re.sub(r"\s+", " ", name).strip().upper()
