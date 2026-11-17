"""Respuestas de referencia calculadas desde el corpus (preguntas cuantitativas).

Estas preguntas miden algo que ni un LLM general ni un RAG puro hacen bien: contar sobre el
corpus completo. Sirven para mostrar el límite de la recuperación (top-k) frente al cómputo.
"""
from __future__ import annotations

from collections import Counter

from ..analysis.pipeline import ALIASES
from ..schema import Unit


def top_speaker(units: list[Unit], saga: str, work: str) -> dict:
    al = ALIASES.get(saga, {})
    c = Counter(al.get(ln.speaker, ln.speaker) for u in units
                if u.medium in ("screenplay", "dialogue") and u.work == work for ln in u.lines)
    (name, n), (second, n2) = c.most_common(2)
    return {"answer": f"{name.title()} ({n} lines; next: {second.title()} {n2})", "must": [[name.lower()]]}


def n_chapters(units: list[Unit], saga: str, work: str) -> dict:
    n = sum(1 for u in units if u.medium == "book" and u.work == work)
    words = {15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
             21: "twenty-one", 22: "twenty-two", 30: "thirty",
             36: "thirty-six", 37: "thirty-seven", 38: "thirty-eight"}
    return {"answer": str(n), "must": [[str(n)] + ([words[n]] if n in words else [])]}


def longest_chapter(units: list[Unit], saga: str, work: str) -> dict:
    u = max((u for u in units if u.medium == "book" and u.work == work), key=lambda u: u.n_words)
    return {"answer": f"{u.title} ({u.n_words} words)", "must": [[u.title.lower()]]}


GOLD_FNS = {"top_speaker": top_speaker, "n_chapters": n_chapters, "longest_chapter": longest_chapter}
