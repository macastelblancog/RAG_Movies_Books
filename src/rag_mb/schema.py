"""Esquema único de registros citables.

Unit  = unidad narrativa citable (capítulo de libro, escena de guion, línea de diálogo).
Chunk = fragmento de una Unit para recuperación; hereda la cita de su Unit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Line:
    """Línea de diálogo dentro de una escena (o atribuida en un capítulo)."""
    speaker: str
    text: str
    parenthetical: str | None = None


@dataclass
class Unit:
    unit_id: str          # p.ej. lotr_book:fotr:b1:c01  |  lotr_fotr_screenplay:s0042
    saga: str
    source_id: str
    medium: str           # book | screenplay | dialogue_csv | subtitles
    work: str             # fotr, ttt, dune_p1, ...
    order: int            # posición narrativa global dentro de (saga, medium)
    locator: str          # cita legible: "Libro 1, Cap. 1 — A Long-expected Party" | "Escena 42 — EXT. SHIRE - DAY"
    title: str
    text: str             # texto completo (acción + diálogo para guiones)
    lines: list[Line] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Unit":
        d = dict(d)
        d["lines"] = [Line(**ln) for ln in d.get("lines", [])]
        return cls(**d)


@dataclass
class Chunk:
    chunk_id: str
    unit_id: str
    saga: str
    medium: str
    work: str
    order: int
    locator: str
    text: str
    speakers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def write_jsonl(records: Iterable, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_units(path: Path) -> list[Unit]:
    return [Unit.from_dict(d) for d in read_jsonl(path)]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.from_dict(d) for d in read_jsonl(path)]
