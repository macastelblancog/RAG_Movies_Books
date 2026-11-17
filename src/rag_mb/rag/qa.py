"""Pregunta → respuesta sobre el índice único (lo usa `rmb ask`).

    route (saga/medio/obra) → recuperación híbrida → reranker → ventana de contexto → LLM con citas [n]

Ventana de contexto: a los mejores pasajes se les une el fragmento anterior y el siguiente del MISMO
capítulo/escena (sin repetir el solapamiento), para que el LLM lea la escena completa y no un trozo.
Los pasajes contiguos de una misma unidad se fusionan en uno; la numeración [n] es la que ve el LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

from .index import Hit, RagIndex, rag_config
from .llm import SYSTEM_RAG, get_client, rag_prompt, split_evidence

# Si la pregunta no dice libro o película, el contexto trae ambos: responder por separado.
SPLIT_BY_MEDIUM = (" The question does not say whether it is about the book or the film and the context has "
                   "both: in the Answer write two short parts, one starting with 'Book:' and one with 'Film:' "
                   "(translated into the question's language), each citing only passages of its own medium. If "
                   "one medium has no relevant evidence, say so for that part.")


@dataclass
class Answer:
    question: str
    hits: list[Hit]                       # pasajes recuperados (ranking del reranker)
    passages: list[Hit] = field(default_factory=list)   # contexto expandido que lee el LLM / se muestra
    saga: str | None = None
    medium: str | None = None
    work: str | None = None
    query: str = ""
    text: str | None = None
    error: str | None = None
    saga_scores: dict = field(default_factory=dict)
    hyde: str | None = None
    balanced: bool = False
    evidence: str | None = None          # paso 1 del LLM: frases copiadas de los pasajes
    raw: str | None = None               # salida completa del LLM (evidencia + respuesta)
    tool: str | None = None              # fuente si respondió la herramienta cuantitativa
    sides: dict | None = None            # respuesta por medio (grafo): {"book": pasajes, "film": pasajes}


SIDE_TITLE = {"book": "Libro", "film": "Película"}


def passage_groups(ans: Answer) -> list[tuple[str, list[Hit]]]:
    """Pasajes para mostrar: agrupados por medio si la respuesta se generó por lado (cada lado numera
    sus citas [n] desde 1); si no, un solo grupo sin título."""
    if ans.sides:
        return [(SIDE_TITLE.get(k, k), v) for k, v in ans.sides.items()]
    return [("", ans.passages)]


PASSAGE_CHARS = 4000                     # tope de caracteres por pasaje en el prompt (tras expandir vecinos)
_SCENE_PREFIX = re.compile(r"^\[[^\]]*\]\s+")


def _merge(a: str, b: str, max_overlap: int = 80) -> str:
    """Une dos fragmentos consecutivos quitando el solapamiento (sufijo de a == prefijo de b)."""
    aw, bw = a.split(), _SCENE_PREFIX.sub("", b).split()
    for n in range(min(max_overlap, len(aw), len(bw)), 0, -1):
        if aw[-n:] == bw[:n]:
            return " ".join(aw + bw[n:])
    return " ".join(aw + bw)


def expand_context(idx: RagIndex, hits: list[Hit], window: int = 1, max_words: int = 2600) -> list[Hit]:
    """Cada pasaje recuperado + sus vecinos del mismo capítulo/escena, en orden de ranking, mientras
    quepan en `max_words` (el contexto del LLM local es de ~8k tokens)."""
    pos = getattr(idx, "_pos", None)
    if pos is None:
        pos = idx._pos = {c.chunk_id: i for i, c in enumerate(idx.chunks)}
    best: dict[int, Hit] = {}                         # posición → mejor hit que la reclama
    words = 0
    ranked = [(h, pos[h.chunk.chunk_id]) for h in hits if h.chunk.chunk_id in pos]
    for h, p in ranked:                               # primero, todos los pasajes recuperados
        best[p] = h
        words += len(idx.chunks[p].text.split())
    for h, p in ranked:                               # luego, vecinos por orden de ranking
        for q in range(p - window, p + window + 1):
            if q in best or not (0 <= q < len(idx.chunks)) or idx.chunks[q].unit_id != h.chunk.unit_id:
                continue
            extra = len(idx.chunks[q].text.split())
            if words + extra > max_words:
                continue
            best[q], words = h, words + extra
    runs, cur = [], []
    for q in sorted(best):                            # tramos contiguos dentro de la misma unidad
        if cur and (q != cur[-1] + 1 or idx.chunks[q].unit_id != idx.chunks[cur[-1]].unit_id):
            runs.append(cur)
            cur = []
        cur.append(q)
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        text = idx.chunks[run[0]].text
        for q in run[1:]:
            text = _merge(text, idx.chunks[q].text)
        lead = min((best[q] for q in run), key=lambda h: h.rank)
        first, last = idx.chunks[run[0]], idx.chunks[run[-1]]
        cid = first.chunk_id if len(run) == 1 else f"{first.chunk_id}..{last.chunk_id.split('#')[-1]}"
        out.append((lead.rank, lead.score, replace(first, text=text, chunk_id=cid)))
    out.sort(key=lambda t: t[0])
    return [Hit(i + 1, sc, ch) for i, (_, sc, ch) in enumerate(out)]


@lru_cache(maxsize=2)
def global_index(dense: str | None = None) -> RagIndex:
    """Un solo índice para todas las sagas; se carga una vez por proceso."""
    return RagIndex.load_or_build("all", dense)


def retrieve(question: str, k: int | None = None, saga: str | None = None, dense: str | None = None,
             hyde: bool | None = None) -> Answer:
    """Recuperación completa (sin LLM): enrutado → búsqueda → ventana de contexto. saga = forzarla (opcional)."""
    from .router import route
    k = k or int(rag_config().get("k", 6))
    hyde = bool(rag_config().get("hyde", False)) if hyde is None else hyde
    idx = global_index(dense)
    pre = route(question, saga)
    kb = int(rag_config().get("k_balanced", k))
    balanced_likely = pre.balanced or pre.medium is None
    hits = idx.search_routed(question, max(k, kb) if balanced_likely else k, balance_unspecified=True,
                             saga=saga, hyde=hyde)
    r = idx.last_route
    cfg = rag_config()
    passages = expand_context(idx, hits, max_words=int(cfg.get("context_words", 1800)))
    passages = passages[:int(cfg.get("llm_passages", 6))]
    return Answer(question, hits, passages, r.saga, r.medium, r.work, r.query,
                  saga_scores=r.saga_scores or {}, hyde=getattr(r, "hyde", None), balanced=r.balanced)


def tool_answer(question: str, saga: str | None = None) -> Answer | None:
    """Preguntas cuantitativas → cálculo sobre el corpus (sin recuperación ni LLM)."""
    from .router import route
    from .tools import detect, run_tool
    if not detect(question):
        return None
    r = route(question, saga)
    t = run_tool(question, r.saga, r.work)
    if t is None:
        return None
    return Answer(question, [], [], t.saga, None, t.work, r.query, text=t.text, tool=t.source)


def prompts(ans: Answer) -> tuple[str, str]:
    """(system, user) que recibe el LLM: los mismos en `rmb ask` y en el benchmark."""
    system = SYSTEM_RAG + (SPLIT_BY_MEDIUM if ans.balanced and ans.medium is None else "")
    return system, rag_prompt(ans.question, ans.passages, PASSAGE_CHARS, best_last=True)


def finish(ans: Answer, raw: str) -> Answer:
    ans.raw = raw
    ans.evidence, ans.text = split_evidence(raw)
    return ans


def ask(question: str, k: int | None = None, client: str | None = None, saga: str | None = None,
        dense: str | None = None, hyde: bool | None = None) -> Answer:
    tool = tool_answer(question, saga)
    if tool is not None:
        return tool
    if client and (rag_config().get("graph") or {}).get("orchestrate", False):
        try:                                   # mismo grafo que la condición rag_graph del benchmark
            from .graph import run as graph_run
            return graph_run(question, client, saga)
        except ImportError:                    # langgraph no instalado → pipeline lineal
            pass
        except Exception as e:                 # LLM caído: se muestran igual los pasajes
            ans = retrieve(question, k, saga, dense, hyde)
            ans.error = str(e)
            return ans
    ans = retrieve(question, k, saga, dense, hyde)
    if client:
        try:
            finish(ans, get_client(client).complete(*prompts(ans)))
        except Exception as e:   # el LLM es opcional: los pasajes se muestran igual
            ans.error = str(e)
    return ans
