r"""Métricas de recuperación y de respuesta.

Recuperación (relevancia binaria definida por `evidence` en el YAML):
  Recall@k   = |{e ∈ E : ∃ chunk relevante a e en top-k}| / |E|   (cobertura de evidencias:
               una pregunta de divergencia necesita evidencia del libro Y del guion)
  MRR        = 1 / rango del primer chunk relevante
  nDCG@k     = Σ_i rel_i / log2(i+1)  normalizado por el DCG ideal

Respuesta:
  keyword-match: correcto si contiene una alternativa de cada grupo `must`.
  abstención   : para `expect: abstain`, correcto si la respuesta rechaza la premisa / se abstiene.
  citas        : fracción de respuestas RAG que citan al menos un fragmento [n].
"""
from __future__ import annotations

import re

import numpy as np

from ..schema import Chunk
from ..text import strip_accents

# Abstención / corrección de premisa: frases que niegan EL HECHO PREGUNTADO, no cualquier negación.
# Una negación cualquiera no basta: "the slave is not drugged" no es una abstención.
ABSTAIN = re.compile(
    r"\b(?:passages?|context|text|sources?)\b[^.]{0,40}\b(?:do|does|did)\s+not\s+"
    r"(?:say|mention|show|state|contain|indicate|describe|include|provide)|"
    r"\bnot\s+(?:mentioned|shown|stated|described|provided)\b|"
    r"\b(?:does|do|did)\s+not\s+(?:appear|happen|occur|exist|take\s+place|feature|include)\b|"
    r"\b(?:doesn't|don't|didn't)\s+(?:appear|happen|occur|exist|say|mention)\b|"
    r"\bnever\s+(?:appears?|happens?|occurs?|takes?\s+place|comes?|goes?|exists?|is|was|does|did)\b|"
    r"\bnot\s+(?:in|part\s+of)\s+the\s+(?:book|film|movie|novel|screenplay|story|passages|context|series)\b|"
    r"\bfalse\s+premise\b|\bno\s+such\b|\bno\s+information\b|\bthere\s+is\s+no\b|"
    r"\b(?:cannot|can't)\s+(?:answer|determine|be\s+determined|find)\b|\bdon't\s+know\b|"
    r"\babsent\s+from\b|\bomitted\b|\bnot\s+featured\b|"
    r"\bno\s+(?:aparece|aparecen|ocurre|sucede|existe|lo\s+dicen|dicen|mencionan|muestran)\b|"
    r"\blos\s+pasajes\s+no\b|\bno\s+esta\s+en\b|\bpremisa\s+falsa\b|\bnunca\b", re.I)
CITATION = re.compile(r"\[\d+\]")


def _norm(s: str) -> str:
    return strip_accents(s).lower()


def chunk_relevant(chunk: Chunk, ev: dict) -> bool:
    if ev.get("medium") and chunk.medium != ev["medium"]:
        return False
    if ev.get("work") and chunk.work != ev["work"]:
        return False
    t = _norm(chunk.text)
    terms = ev["terms"]
    # terms: ["a", "b"] → todos;  [["a", "b"], ["c"]] → cualquiera de los conjuntos (frases partidas por chunking)
    sets = terms if terms and isinstance(terms[0], list) else [terms]
    return any(all(_norm(x) in t for x in st) for st in sets)


def retrieval_metrics(chunks: list[Chunk], evidence: list[dict], k: int) -> dict:
    if not evidence:
        return {}
    rel = np.array([any(chunk_relevant(c, e) for e in evidence) for c in chunks[:k]], float)
    covered = [any(chunk_relevant(c, e) for c in chunks[:k]) for e in evidence]
    first = np.flatnonzero(rel)
    dcg = float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())
    n_rel = int(rel.sum())   # nDCG relativo a lo recuperado: mide la CALIDAD DEL ORDEN en el top-k
    ideal = float((1 / np.log2(np.arange(2, n_rel + 2))).sum()) if n_rel else 0.0
    return {"recall@k": float(np.mean(covered)), "hit@k": float(rel.any()),
            "mrr": float(1 / (first[0] + 1)) if len(first) else 0.0,
            "ndcg@k": dcg / ideal if ideal else 0.0}


def score_answer(answer: str, must: list[list[str]] | None, expect: str | None = None) -> bool:
    a = _norm(answer or "")
    if expect == "abstain":
        return bool(ABSTAIN.search(a))
    if not must:
        return False
    return all(any(_norm(alt) in a for alt in group) for group in must)


def has_citation(answer: str) -> bool:
    return bool(CITATION.search(answer or ""))
