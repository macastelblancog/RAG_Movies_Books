"""Herramienta cuantitativa: preguntas que se CALCULAN sobre el corpus completo, no se recuperan.

"¿Cuántos capítulos tiene…?", "¿Quién tiene más líneas en…?", "¿Cuál es el capítulo más largo…?"
ningún pasaje lo dice: hay que contar. El enrutador detecta la intención y la obra; el cálculo usa las
MISMAS funciones que el gold de la evaluación (rag_mb.eval.gold), así que hay una sola fuente de verdad.
Si falta la saga o la obra, devuelve None y la pregunta sigue por el RAG normal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..config import saga_config

INTENTS = {
    "top_speaker": re.compile(r"\b(most|more)\s+(lines|dialogue|dialog)\b|\bspeaks?\s+the\s+most\b|"
                              r"\bm[aá]s\s+l[ií]neas\b|\bhabla\s+m[aá]s\b|\bm[aá]s\s+di[aá]logos?\b", re.I),
    "n_chapters": re.compile(r"\bhow\s+many\s+chapters\b|\bnumber\s+of\s+chapters\b|"
                             r"\bcu[aá]ntos\s+cap[ií]tulos\b|\bn[uú]mero\s+de\s+cap[ií]tulos\b", re.I),
    "longest_chapter": re.compile(r"\blongest\s+chapter\b|\bchapter\b.*\blongest\b|"
                                  r"\bcap[ií]tulo\s+m[aá]s\s+largo\b", re.I),
}
SPANISH = re.compile(r"[¿áéíóúñ]|\b(cu[aá]ntos|qui[eé]n|cu[aá]l|tiene|libro|pel[ií]cula)\b", re.I)


@dataclass
class ToolAnswer:
    intent: str
    saga: str
    work: str
    text: str
    source: str


@lru_cache(maxsize=8)
def _units(saga: str):
    from ..ingest.build import units_for
    return units_for(saga)


def _book_work(saga: str, work: str) -> str:
    return saga_config(saga).get("book_for_film", {}).get(work, work)


def detect(question: str) -> str | None:
    return next((name for name, p in INTENTS.items() if p.search(question)), None)


def run_tool(question: str, saga: str | None, work: str | None) -> ToolAnswer | None:
    intent = detect(question)
    if not intent or not saga or not work:
        return None
    from ..eval.gold import GOLD_FNS
    target = work if intent == "top_speaker" else _book_work(saga, work)
    try:
        g = GOLD_FNS[intent](list(_units(saga)), saga, work=target)
    except (ValueError, StopIteration):      # obra sin datos de ese medio
        return None
    es = bool(SPANISH.search(question))
    if intent == "top_speaker":
        src = "conteo de líneas de diálogo del guion" if es else "count of dialogue lines in the screenplay"
        text = (f"Según el conteo sobre el corpus, quien más habla es {g['answer']}."
                if es else f"By counting over the corpus, the character with the most lines is {g['answer']}.")
    elif intent == "n_chapters":
        src = "capítulos detectados en el libro" if es else "chapters detected in the book"
        text = (f"Según el corpus, el libro tiene {g['answer']} capítulos." if es
                else f"According to the corpus, the book has {g['answer']} chapters.")
    else:
        src = "palabras por capítulo del libro" if es else "words per chapter in the book"
        text = (f"Según el corpus, el capítulo más largo es {g['answer']}." if es
                else f"According to the corpus, the longest chapter is {g['answer']}.")
    return ToolAnswer(intent, saga, target, text, f"{src} · {saga}/{target}")
