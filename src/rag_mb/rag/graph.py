"""El RAG como grafo (LangGraph): los nodos son las funciones de `rag.qa`; las ramas son DETERMINISTAS
(las decide el código, no el LLM). Mismo comportamiento que `qa.ask` salvo la rama de divergencia.

    pregunta ─► herramienta ──(conteo)──────────────────────────────► fin
                     │
                     └─► recuperar ─► generar ───────────────────────► fin
                              │
                              └─(libro Y película)─► libro ─► película ─► unir ─► fin

Rama de divergencia (config rag.graph.split_divergence): la recuperación balanceada ya trae pasajes
del libro y de la película; en vez de una sola respuesta que mezcla los dos relatos, el LLM responde
DOS veces, cada una solo con los pasajes de su medio, y el grafo une "Book: … / Film: …".
Cada lado arma su propio contexto (hasta rag.llm_passages pasajes de SU medio) desde los hits recuperados.
Las dos llamadas van en secuencia: llama-server atiende una petición a la vez, en paralelo no se gana nada.
Se activa si la pregunta nombra ambos medios o ninguno (no en preguntas de presencia).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, TypedDict

from .index import FILM_MEDIA, rag_config
from .llm import get_client
from .qa import Answer, expand_context, finish, global_index, prompts, retrieve, tool_answer
from .router import BOOK, FILM, PRESENCE

SIDES = {"book": ("book",), "film": FILM_MEDIA}
LABEL = {"book": "book", "film": "film"}


class RagState(TypedDict, total=False):
    question: str
    saga: str | None
    retrieved: Answer          # salida de la recuperación (pasajes ya expandidos)
    book: Answer               # rama de divergencia: respuesta solo con pasajes del libro
    film: Answer               # … y solo con pasajes de la película
    answer: Answer             # respuesta final


def split_enabled() -> bool:
    return bool((rag_config().get("graph") or {}).get("split_divergence", False))


def is_divergence(a: Answer) -> bool:
    """Se responde por medio (libro y película por separado) cuando la recuperación fue balanceada y:
      - la pregunta compara libro y película de forma EXPLÍCITA (o nombra una obra que solo existe como
        película), o
      - la pregunta NO nombra ningún medio: la respuesta igual sale como "Book / Film" y, con los pasajes
        mezclados, el modelo llenaba un lado con evidencia del otro.
    Nunca en preguntas de presencia ("¿aparece X en…?"), que necesitan ver ambos medios a la vez."""
    q = a.question
    if not a.balanced or PRESENCE.search(q):
        return False
    book, film = bool(BOOK.search(q)), bool(FILM.search(q))
    return (book and (film or bool(a.work))) or (not book and not film and a.medium is None)


def side_passages(a: Answer, side: str) -> list:
    """Contexto PROPIO de cada lado: se arma desde los hits recuperados de ese medio (no desde los pasajes
    ya recortados a rag.llm_passages, donde a veces solo sobrevive un medio)."""
    cfg = rag_config()
    hits = [h for h in a.hits if h.chunk.medium in SIDES[side]]
    if not hits:
        return []
    passages = expand_context(global_index(), hits, max_words=int(cfg.get("context_words", 1800)))
    return passages[:int(cfg.get("llm_passages", 6))]


def _side_answer(a: Answer, side: str, llm) -> Answer:
    passages = side_passages(a, side)
    q = f"{a.question} (Answer ONLY for the {LABEL[side]}: all passages below come from the {LABEL[side]}.)"
    sub = replace(a, question=q, passages=passages, balanced=False, medium=side)
    if not passages:
        sub.text, sub.evidence, sub.raw = f"the passages do not show the {LABEL[side]}", "none", ""
        return sub
    return finish(sub, llm.complete(*prompts(sub)))


def build(llm):
    """Grafo compilado para un cliente LLM (objeto con .complete(system, user))."""
    from langgraph.graph import END, START, StateGraph

    def tool(s: RagState):
        # siempre escribe la clave (LangGraph rechaza un nodo que no actualiza nada); None = no es de conteo
        return {"answer": tool_answer(s["question"], s.get("saga"))}

    def retrieve_node(s: RagState):
        return {"retrieved": retrieve(s["question"], saga=s.get("saga"))}

    def generate(s: RagState):
        a = s["retrieved"]
        return {"answer": finish(a, llm.complete(*prompts(a)))}

    def book(s: RagState):
        return {"book": _side_answer(s["retrieved"], "book", llm)}

    def film(s: RagState):
        return {"film": _side_answer(s["retrieved"], "film", llm)}

    def merge(s: RagState):
        a, b, f = s["retrieved"], s["book"], s["film"]
        return {"answer": replace(a, passages=b.passages + f.passages,   # lo que leyó cada lado
                                  sides={"book": b.passages, "film": f.passages},
                                  text=f"Book: {b.text}\nFilm: {f.text}",
                                  evidence=f"[book] {b.evidence} | [film] {f.evidence}",
                                  raw=f"[book]\n{b.raw}\n[film]\n{f.raw}")}

    g = StateGraph(RagState)
    for name, fn in [("tool", tool), ("retrieve", retrieve_node), ("generate", generate),
                     ("book", book), ("film", film), ("merge", merge)]:
        g.add_node(name, fn)
    g.add_edge(START, "tool")
    g.add_conditional_edges("tool", lambda s: "end" if s.get("answer") else "retrieve",
                            {"end": END, "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve",
                            lambda s: "split" if split_enabled() and is_divergence(s["retrieved"]) else "generate",
                            {"split": "book", "generate": "generate"})
    g.add_edge("book", "film")
    g.add_edge("film", "merge")
    g.add_edge("generate", END)
    g.add_edge("merge", END)
    return g.compile()


_APPS: dict[int, Any] = {}


def run(question: str, client: Any, saga: str | None = None) -> Answer:
    """client: nombre ("local:phi4mini") u objeto LLM ya creado. Devuelve el Answer final."""
    key = client if isinstance(client, str) else id(client)
    if key not in _APPS:
        _APPS[key] = build(get_client(client) if isinstance(client, str) else client)
    app = _APPS[key]
    return app.invoke({"question": question, "saga": saga})["answer"]


def mermaid() -> str:
    """Diagrama del grafo (Mermaid) para el informe / la presentación."""
    class _Dummy:
        def complete(self, system, user):
            return ""
    return build(_Dummy()).get_graph().draw_mermaid()
