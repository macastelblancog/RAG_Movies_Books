"""Citas del libro con hablante atribuido (intervenciones del lado libro).

Las citas se etiquetan con patrones de atribución explícita junto a la cita
("…," said Frodo / Frodo asked, "…"). Solo se conservan citas con UN único hablante
detectado en la ventana, priorizando precisión sobre cobertura.
"""
from __future__ import annotations

import re

import pandas as pd

from ..config import saga_config
from ..schema import Unit
from .pipeline import BOOK_MENTIONS, FILM_MEDIA

SPEECH_VERBS = (r"said|asked|cried|answered|replied|shouted|whispered|muttered|called|exclaimed|"
                r"laughed|growled|snapped|added|began|went on|continued|murmured|roared|sighed|"
                r"demanded|insisted|repeated|agreed|interrupted|gasped|yelled|hissed")
QUOTE = re.compile(r'"([^"]{3,600})"')
# Comillas simples (estilo británico, p.ej. LOTR): el apóstrofo entre letras ("don't", "Frodo's")
# no cierra la cita; la apertura no puede ir pegada a una letra.
QUOTE_SINGLE = re.compile(r"(?<![\w'])'((?:[^'\n]|(?<=\w)'(?=\w)){3,600}?)'(?![\w])")


def quote_regex(text: str) -> re.Pattern:
    """Estilo de comillas del texto: dobles si aparecen con frecuencia razonable; si no, simples."""
    return QUOTE if text.count('"') >= 0.1 * text.count("'") else QUOTE_SINGLE


def _attribution_patterns(saga: str) -> dict[str, re.Pattern]:
    pats = {}
    for canon, names in BOOK_MENTIONS.get(saga, {}).items():
        alt = "|".join(re.escape(n) for n in names)
        pats[canon] = re.compile(
            rf"(?:\b(?:{SPEECH_VERBS})\s+(?:the\s+)?(?:{alt})\b)|(?:\b(?:{alt})\s+(?:{SPEECH_VERBS})\b)")
    return pats


def book_quotes(units: list[Unit], saga: str, min_words: int = 3, window: int = 60) -> pd.DataFrame:
    """Citas del libro con hablante inferido por atribución explícita CERCANA a la cita.

    Para cada cita se mira el texto narrativo adyacente (hasta `window` caracteres antes y después,
    sin cruzar la cita vecina). Se etiqueta solo si aparece exactamente UN personaje con un verbo de
    habla ("said Frodo", "Frodo asked"). No depende de los saltos de párrafo, que varían por PDF.
    """
    film_works = {u.work for u in units if u.medium in FILM_MEDIA}
    bff = saga_config(saga).get("book_for_film", {})
    adapted = {bff.get(w, w) for w in film_works}
    pats = _attribution_patterns(saga)
    rows = []
    for u in units:
        if u.medium != "book" or (adapted and u.work not in adapted):
            continue
        text = u.text
        ms = list(quote_regex(text).finditer(text))
        for k, m in enumerate(ms):
            lo = max(ms[k - 1].end() if k else 0, m.start() - window)
            hi = min(ms[k + 1].start() if k + 1 < len(ms) else len(text), m.end() + window)
            ctx = text[lo:m.start()] + " " + text[m.end():hi]
            who = [c for c, p in pats.items() if p.search(ctx)]
            q = m.group(1).strip(" ,")
            if len(who) == 1 and len(q.split()) >= min_words:
                rows.append({"text": q, "speaker": who[0], "group": u.unit_id, "work": u.work,
                             "source": "book"})
    return pd.DataFrame(rows, columns=["text", "speaker", "group", "work", "source"])
