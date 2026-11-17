"""HyDE (Hypothetical Document Embeddings; Gao et al., 2022): el LLM escribe un pasaje HIPOTÉTICO que
respondería la pregunta, y se busca con él además de con la pregunta.

Ataca la brecha semántica pregunta↔escena: "¿Cómo muere Boromir?" no se parece al texto que narra la
muerte ("pierced with many black-feathered arrows"), pero un pasaje hipotético sí. El pasaje puede
contener errores: solo GUÍA la búsqueda; la respuesta se basa en los pasajes reales recuperados, y el
reranker juzga contra la pregunta original.

Caché en reports/cache/hyde.jsonl (clave: modelo + pregunta + saga + medio) → evaluación reproducible y
sin repetir llamadas al LLM.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from ..config import paths
from .llm import get_client

SAGA_NAME = {"lotr": "J.R.R. Tolkien's The Lord of the Rings and Peter Jackson's films",
             "hp": "J.K. Rowling's Harry Potter books and films",
             "dune": "Frank Herbert's Dune and Denis Villeneuve's films",
             "narnia": "C.S. Lewis's The Chronicles of Narnia and its films"}
SYSTEM_HYDE = ("You write a short passage (80-120 words), in English, that would directly answer the question, "
               "as it could appear in the source text. Use character and place names. No preamble, no hedging, "
               "no bullet points: just the passage.")


def _cache_file():
    f = paths().reports / "cache" / "hyde.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


@lru_cache(maxsize=1)
def _cache() -> dict:
    f = _cache_file()
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = d["text"]
    return out


def hypothetical(question: str, saga: str | None, medium: str | None, balanced: bool,
                 client: str = "local") -> str | None:
    llm = get_client(client)
    where = ("the book and the film" if balanced or medium is None
             else "the book" if medium == "book" else "the film screenplay")
    ctx = f"Context: {SAGA_NAME.get(saga, 'a fantasy book and its film adaptation')}; write it as {where}."
    key = hashlib.sha1(f"{llm.name}|{question}|{saga}|{medium}|{balanced}".encode()).hexdigest()
    cache = _cache()
    if key in cache:
        return cache[key]
    try:
        user = f"{ctx}\nQuestion: {question}\nPassage:"
        try:
            text = llm.complete(SYSTEM_HYDE, user, temperature=0.0, max_tokens=220)
        except TypeError:      # clientes sin parámetros de muestreo (OpenAI / Gemini)
            text = llm.complete(SYSTEM_HYDE, user)
    except Exception:          # sin LLM disponible: se busca solo con la pregunta
        return None
    text = (text or "").strip()
    if text:
        cache[key] = text
        with open(_cache_file(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "model": llm.name, "question": question, "saga": saga,
                                 "medium": medium, "text": text}, ensure_ascii=False) + "\n")
    return text or None
