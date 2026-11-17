"""Índice RAG ÚNICO: chunks de todas las sagas + recuperadores, con filtros por saga/medio/obra y citas.

    RagIndex.load_or_build()            → índice global (todas las sagas con chunks construidos)
    RagIndex.load_or_build("lotr")      → índice de una saga (se conserva para la evaluación por saga)

Pipeline de búsqueda:  filtros (saga/medio/obra) → BM25 ⊕ denso (RRF) → [reranker cross-encoder] → top-k
Reranker por defecto en modo "rrf": su orden se FUSIONA con el híbrido en vez de reemplazarlo; el
cross-encoder (entrenado en búsquedas web) entiende mal el formato de guion y a veces descarta el
pasaje correcto que el híbrido sí tenía arriba.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass

import numpy as np

from ..config import load_config, paths
from ..schema import Chunk, load_chunks
from .retrievers import BM25, CrossEncoderReranker, DenseLSA, DenseNeural, rrf

DEFAULT_MEDIA = ("book", "screenplay", "dialogue", "subtitles")   # dialogue_csv = control, no se indexa
FILM_MEDIA = ("screenplay", "dialogue", "subtitles")
SAGA_LABEL = {"lotr": "LOTR", "hp": "Harry Potter", "dune": "Dune", "narnia": "Narnia"}
MEDIUM_LABEL = {"book": "Libro", "screenplay": "Guion", "dialogue": "Diálogo película", "subtitles": "Subtítulos"}
# Etiqueta que ve el LLM (inglés, como el prompt de sistema): el medio en MAYÚSCULAS y sin "Capítulo" en los
# pasajes de película (el diálogo de HP viene agrupado por "capítulos" del dataset de Maven).
PROMPT_MEDIUM = {"book": "BOOK", "screenplay": "FILM — screenplay", "dialogue": "FILM — dialogue transcript",
                 "subtitles": "FILM — subtitles"}
_LOCATOR_EN = [(r"^Libro (\d+), Cap\. (\d+)", r"Part \1, Chapter \2"), (r"^Cap\. (\d+)", r"Chapter \1"),
               (r"^Escena (\d+)", r"Scene \1"), (r"^Capítulo (\d+)", r"Scene group \1")]


def rag_config() -> dict:
    return load_config().get("rag", {}) or {}


def context_header(c: Chunk) -> str:
    sc = load_config()["sagas"].get(c.saga, {})
    work = (sc.get("work_titles") or {}).get(c.work, c.work)
    medium = "book" if c.medium == "book" else "film screenplay" if c.medium == "screenplay" else "film dialogue"
    return f"{sc.get('title', c.saga)} · {medium} · {work} · {c.locator}"


def indexed_sagas() -> list[str]:
    return [s for s in load_config()["sagas"] if paths().chunks_file(s).exists()]


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk

    @property
    def citation(self) -> str:
        medium = MEDIUM_LABEL.get(self.chunk.medium, self.chunk.medium)
        saga = SAGA_LABEL.get(self.chunk.saga, self.chunk.saga)
        return f"{saga} · {medium} · {self.chunk.work} · {self.chunk.locator}"

    @property
    def prompt_label(self) -> str:
        """Fuente del pasaje tal como la lee el LLM (`citation` es la versión en español para mostrar)."""
        loc = self.chunk.locator
        for pat, rep in _LOCATOR_EN:
            loc = re.sub(pat, rep, loc)
        medium = PROMPT_MEDIUM.get(self.chunk.medium, self.chunk.medium)
        return f"{medium} · {SAGA_LABEL.get(self.chunk.saga, self.chunk.saga)} · {self.chunk.work} · {loc}"


class RagIndex:
    def __init__(self, saga: str = "all", dense: str | None = None, media: tuple[str, ...] = DEFAULT_MEDIA,
                 rerank: bool | None = None):
        cfg = rag_config()
        self.saga = saga
        self.dense_kind = dense or cfg.get("dense", "lsa")
        self.media = media
        self.rerank_pool = int(cfg.get("rerank_pool", 30))
        self.rerank_fusion = cfg.get("rerank_fusion", "rrf")
        self.contextual = bool(cfg.get("contextual_headers", False))
        use_rerank = (self.dense_kind == "neural") if rerank is None else rerank
        self.reranker = (CrossEncoderReranker(cfg.get("reranker_model") or
                                              "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
                         if use_rerank else None)
        self.last_route = None

    @property
    def sagas(self) -> list[str]:
        return indexed_sagas() if self.saga == "all" else [self.saga]

    def _dense_model(self):
        if self.dense_kind != "neural":
            return DenseLSA()
        cfg = rag_config()
        return DenseNeural(cfg.get("embedding_model", "intfloat/multilingual-e5-base"),
                           cfg.get("query_prefix", ""), cfg.get("passage_prefix", ""), cfg.get("max_seq_length"))

    def index_text(self, c: Chunk) -> str:
        """Texto que se indexa. Con encabezado contextual ("contextual retrieval"): un fragmento de mitad de
        capítulo no dice de qué obra ni capítulo es; el encabezado se lo da a BM25, al embedding y al reranker."""
        if not self.contextual:
            return c.text
        return f"{context_header(c)}\n{c.text}"

    def build(self) -> "RagIndex":
        self.chunks = [c for s in self.sagas for c in load_chunks(paths().chunks_file(s)) if c.medium in self.media]
        texts = [self.index_text(c) for c in self.chunks]
        self.bm25 = BM25().fit(texts)
        self.dense = self._dense_model().fit(texts)
        self._media = np.array([c.medium for c in self.chunks])
        self._works = np.array([c.work for c in self.chunks])
        self._sagas = np.array([c.saga for c in self.chunks])
        return self

    # ---- persistencia
    def path(self):
        ctx = "_ctx" if self.contextual else ""
        return paths().reports / "indexes" / f"{self.saga}_{self.dense_kind}{ctx}.pkl"

    def save(self):
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self, fh)
        return p

    @classmethod
    def load_or_build(cls, saga: str = "all", dense: str | None = None, rerank: bool | None = None,
                      rebuild: bool = False) -> "RagIndex":
        fresh = cls(saga, dense, rerank=rerank)
        p = fresh.path()
        newest = max(paths().chunks_file(s).stat().st_mtime for s in fresh.sagas)
        if p.exists() and not rebuild and p.stat().st_mtime > newest:
            with open(p, "rb") as fh:
                idx = pickle.load(fh)
            idx.reranker, idx.rerank_pool = fresh.reranker, fresh.rerank_pool   # reranker: decisión de uso
            idx.rerank_fusion = fresh.rerank_fusion
            idx.contextual = getattr(idx, "contextual", False)
            idx.last_route = None
            return idx
        idx = fresh.build()
        idx.save()
        return idx

    # ---- búsqueda
    def _mask(self, medium=None, work=None, saga=None) -> np.ndarray:
        mask = np.ones(len(self.chunks), bool)
        if saga:
            mask &= self._sagas == saga
        if medium:
            mask &= np.isin(self._media, [medium] if isinstance(medium, str) else list(medium))
        if work:
            mask &= self._works == work
        return mask

    def search(self, query: str, k: int = 8, mode: str = "hybrid", medium: str | None = None,
               work: str | None = None, saga: str | None = None, pool: int = 100,
               rerank: bool = True, expansion: str | None = None) -> list[Hit]:
        """expansion (HyDE): pasaje hipotético. BM25 busca con pregunta + pasaje; el denso promedia la
        similitud a la pregunta y al pasaje; el reranker juzga SIEMPRE contra la pregunta original."""
        mask = self._mask(medium, work, saga)
        lex = f"{query} {expansion}" if expansion else query

        def dense_scores():
            d = self.dense.scores(query)
            if expansion:
                d = (d + (self.dense.scores_passage(expansion) if hasattr(self.dense, "scores_passage")
                          else self.dense.scores(expansion))) / 2
            return d
        if mode == "bm25":
            s = self.bm25.scores(lex)
        elif mode == "dense":
            s = dense_scores()
        elif mode == "hybrid":
            s1, s2 = self.bm25.scores(lex), dense_scores()
            s1[~mask], s2[~mask] = -np.inf, -np.inf
            r1, r2 = np.argsort(-s1)[:pool], np.argsort(-s2)[:pool]
            s = rrf([r1, r2], len(self.chunks))
        else:
            raise ValueError(mode)
        s = np.where(mask, s, -np.inf)
        if rerank and self.reranker is not None:
            cand = [j for j in np.argsort(-s)[:max(k, self.rerank_pool)] if np.isfinite(s[j])]
            if cand:
                ce = self.reranker.scores(query, [self.index_text(self.chunks[j]) for j in cand])
                if self.rerank_fusion == "replace":
                    final = ce
                else:   # RRF de dos listas: orden híbrido (cand ya viene ordenado) y orden del reranker
                    r_ce = np.empty(len(cand))
                    r_ce[np.argsort(-ce)] = np.arange(1, len(cand) + 1)
                    final = 1 / (60 + np.arange(1, len(cand) + 1)) + 1 / (60 + r_ce)
                order = np.argsort(-final)[:k]
                return [Hit(i + 1, float(final[o]), self.chunks[cand[o]]) for i, o in enumerate(order)]
        top = np.argsort(-s)[:k]
        return [Hit(i + 1, float(s[j]), self.chunks[j]) for i, j in enumerate(top) if np.isfinite(s[j])]

    def search_routed(self, question: str, k: int = 8, mode: str = "hybrid",
                      balance_unspecified: bool = False, saga: str | None = None,
                      hyde: bool = False) -> list[Hit]:
        """Consulta enrutada: saga (si el índice es global), medio y obra deducidos de la pregunta.
        balance_unspecified=True: si la pregunta no nombra libro ni película, mitad y mitad (para
        responder por separado y no mezclar los dos relatos)."""
        from .router import route
        forced = saga                     # saga conocida (p. ej. evaluación "oráculo"): no se detecta
        r = route(question, forced or (None if self.saga == "all" else self.saga))
        self.last_route = r
        if self.saga == "all" and r.saga is None:
            r.saga = self.vote_saga(r.query, mode)
        saga = r.saga if self.saga == "all" else None
        exp = None
        if hyde:
            from .hyde import hypothetical
            exp = r.hyde = hypothetical(question, r.saga or self.saga, r.medium, r.balanced)
        if r.balanced:
            return self.search_balanced(r.query, k, mode, saga=saga, expansion=exp)
        medium = self.film_medium(saga) if r.medium == "screenplay" else r.medium
        work = r.work
        in_saga = self._mask(saga=saga)
        if work and medium is None:
            media_of_work = set(self._media[in_saga & (self._works == work)])
            if media_of_work and media_of_work.isdisjoint({"book"}):
                medium = self.film_medium(saga)  # "Dune: Part Two" solo existe como película
            else:
                work = None                      # obra ambigua sin medio explícito: sin filtro
        if (work and medium == "book" and not np.any(self._mask("book", work, saga))
                and np.any(self._mask(FILM_MEDIA, work, saga))):
            # "en el libro … y en Dune: Part Two": la obra solo existe como película → libro + esa película
            r.balanced = True
            return self.search_balanced(r.query, k, mode, saga=saga, expansion=exp, film_work=work)
        if work and not np.any(self._mask(medium, work, saga)):
            work = None          # p.ej. "Deathly Hallows" sin "Part": existe como libro, no como película
        if balance_unspecified and medium is None and work is None:
            r.balanced = True
            return self.search_balanced(r.query, k, mode, saga=saga, expansion=exp)
        return self.search(r.query, k, mode, medium=medium, work=work, saga=saga, expansion=exp)

    def vote_saga(self, query: str, mode: str = "hybrid", n: int = 20, min_share: float = 0.6) -> str | None:
        """Sin pistas en la pregunta: búsqueda preliminar en todo el índice y voto de los n mejores pasajes.
        Solo se filtra si una saga domina (≥ min_share); si no, se sigue buscando en todas."""
        hits = self.search(query, n, mode, rerank=False)
        if not hits:
            return None
        sagas, counts = np.unique([h.chunk.saga for h in hits], return_counts=True)
        top = int(np.argmax(counts))
        return str(sagas[top]) if counts[top] / len(hits) >= min_share else None

    def film_medium(self, saga: str | None = None) -> str:
        present = set(self._media[self._mask(saga=saga)])
        return next((m for m in FILM_MEDIA if m in present), "screenplay")

    def search_balanced(self, query: str, k: int = 8, mode: str = "hybrid", saga: str | None = None,
                        expansion: str | None = None, film_work: str | None = None) -> list[Hit]:
        """k/2 del libro + k/2 de la película: necesario para preguntas de DIVERGENCIA libro↔película."""
        hits = self.search(query, k // 2, mode, medium="book", saga=saga, expansion=expansion)
        # lado película: todos los medios fílmicos (guion en LOTR/Dune/Narnia, diálogo Maven en HP)
        hits += self.search(query, k - len(hits), mode, medium=FILM_MEDIA if saga is None else
                            self.film_medium(saga), work=film_work, saga=saga, expansion=expansion)
        hits.sort(key=lambda h: -h.score)
        return [Hit(i + 1, h.score, h.chunk) for i, h in enumerate(hits)]
