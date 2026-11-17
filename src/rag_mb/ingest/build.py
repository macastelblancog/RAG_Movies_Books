"""Orquestación de la ingesta: config → units.jsonl + chunks.jsonl por saga."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from ..config import load_config, paths, saga_config
from ..schema import Unit, load_units, write_jsonl
from .book_pdf import parse_book
from .chunking import chunk_units
from .screenplay_pdf import parse_screenplay
from .tabular import parse_dialogue_csv, parse_relational_dialogue, parse_srt

log = logging.getLogger(__name__)

PARSERS = {
    "book_pdf": parse_book,
    "screenplay_pdf": parse_screenplay,
    "dialogue_csv": parse_dialogue_csv,
    "srt": parse_srt,
    "relational_dialogue": parse_relational_dialogue,
}


def build_saga(saga: str, *, only: list[str] | None = None) -> dict:
    cfg, sc, P = load_config(), saga_config(saga), paths()
    units: list[Unit] = []
    report = {"saga": saga, "sources": {}}
    offsets: Counter = Counter()
    for src in sc["sources"]:
        if only and src["id"] not in only:
            continue
        fpath: Path = P.raw / src["path"]
        if not fpath.exists():
            log.warning("FALTA %s → %s (se omite)", src["id"], fpath)
            report["sources"][src["id"]] = "missing"
            continue
        kwargs = {"saga": saga, "source": src, "order_offset": offsets[src["medium"]]}
        if src["parser"] == "screenplay_pdf":
            kwargs["layout"] = cfg.get("screenplay_layout")
        us = PARSERS[src["parser"]](fpath, **kwargs)
        offsets[src["medium"]] += len(us)
        units.extend(us)
        report["sources"][src["id"]] = {"units": len(us), "words": sum(u.n_words for u in us)}

    ch = cfg.get("chunking", {})
    chunks = chunk_units(units, max_words=ch.get("max_words", 220), overlap_words=ch.get("overlap_words", 40))
    write_jsonl(units, P.corpus_file(saga))
    write_jsonl(chunks, P.chunks_file(saga))
    report.update(units=len(units), chunks=len(chunks))
    return report


def rechunk_saga(saga: str) -> dict:
    """Vuelve a trocear desde data/processed/<saga>_units.jsonl (sin re-leer PDFs ni OCR)."""
    ch = load_config().get("chunking", {})
    chunks = chunk_units(units_for(saga), max_words=ch.get("max_words", 220),
                         overlap_words=ch.get("overlap_words", 40))
    write_jsonl(chunks, paths().chunks_file(saga))
    return {"saga": saga, "chunks": len(chunks)}


def units_for(saga: str) -> list[Unit]:
    f = paths().corpus_file(saga)
    if not f.exists():
        raise FileNotFoundError(f"No existe {f}. Ejecuta primero: rmb build {saga}")
    return load_units(f)
