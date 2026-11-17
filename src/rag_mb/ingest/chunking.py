"""Units → Chunks para recuperación. Cada chunk hereda la cita (locator) de su unidad.

No se corta a ciegas: Se corta por párrafos/líneas y se respeta el tamaño
máximo con solapamiento; fragmentos huérfanos muy cortos se fusionan con el anterior.
"""
from __future__ import annotations

from ..schema import Chunk, Unit


def _pieces(unit: Unit) -> list[str]:
    if unit.medium == "book":
        return [p for p in unit.text.split("\n\n") if p.strip()]
    return [p for p in unit.text.split("\n") if p.strip()]


def chunk_unit(unit: Unit, max_words: int = 220, overlap_words: int = 40, min_words: int = 25) -> list[Chunk]:
    """Ventanas de ≤ max_words que respetan párrafos; cada ventana nueva arranca con las últimas
    `overlap_words` de la anterior. Un fragmento SOLO se emite si trae palabras nuevas (`new` > 0):
    así el solapamiento nunca queda como fragmento propio (en libros sin saltos de párrafo cada
    capítulo es un bloque gigante y, sin esta regla, se emitirían restos de 40 palabras repetidas)."""
    chunks_words: list[list[str]] = []
    cur: list[str] = []
    new = 0                                        # palabras de `cur` aún no emitidas

    def flush():
        nonlocal cur, new
        chunks_words.append(cur)
        cur = cur[-overlap_words:] if overlap_words else []
        new = 0

    for pw in (p.split() for p in _pieces(unit)):
        if new and len(cur) + len(pw) > max_words:
            flush()                                # el párrafo no cabe: cerrar la ventana actual
        cur, new = cur + pw, new + len(pw)
        while len(cur) > max_words:                # párrafo gigante: ventana deslizante
            chunks_words.append(cur[:max_words])
            cur = cur[max_words - overlap_words:] if overlap_words else cur[max_words:]
            new = max(len(cur) - overlap_words, 0) if overlap_words else len(cur)
    if new:
        if chunks_words and new < min_words:
            chunks_words[-1] = chunks_words[-1] + cur[-new:]   # cola corta: se pega a la anterior
        else:
            chunks_words.append(cur)

    speakers = sorted({ln.speaker for ln in unit.lines if ln.speaker.strip()})
    out = []
    for k, ws in enumerate(chunks_words):
        text = " ".join(ws)
        if unit.medium == "screenplay" and k > 0:
            text = f"[{unit.title}] " + text       # contexto de escena en cada chunk
        out.append(Chunk(
            chunk_id=f"{unit.unit_id}#{k:02d}", unit_id=unit.unit_id, saga=unit.saga,
            medium=unit.medium, work=unit.work, order=unit.order, locator=unit.locator,
            text=text, speakers=[s for s in speakers if s.split()[0].lower() in text.lower()] or speakers,
        ))
    return out


def chunk_units(units: list[Unit], **kw) -> list[Chunk]:
    return [c for u in units for c in chunk_unit(u, **kw)]
