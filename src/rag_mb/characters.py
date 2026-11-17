"""Personajes por saga (fuente única: configs/characters.yaml).

    aliases(saga)  → {variante de hablante: canónico}   (guion / CSV)
    mentions(saga) → {canónico: [formas en el texto]}    (libro, acotaciones, diálogo)
    notes(saga)    → {canónico: advertencia}
    mention_regex(names) → patrón compilado (límite de palabra, forma tal cual + TODO-MAYÚSCULAS)
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from .config import project_root


@lru_cache(maxsize=1)
def load_characters() -> dict:
    f = project_root() / "configs" / "characters.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else {}


def _saga(saga: str) -> dict:
    return load_characters().get(saga) or {}


def aliases(saga: str) -> dict[str, str]:
    return {cue: canon for canon, c in _saga(saga).items() for cue in (c or {}).get("cues", [])}


def mentions(saga: str) -> dict[str, list[str]]:
    return {canon: list((c or {}).get("names", [])) for canon, c in _saga(saga).items()}


def notes(saga: str) -> dict[str, str]:
    return {canon: c["nota"] for canon, c in _saga(saga).items() if c and c.get("nota")}


def mention_regex(names: list[str]) -> re.Pattern | None:
    """Alternancia más-larga-primero: 'Feyd-Rautha' se cuenta una vez, no también como 'Feyd'.
    Distingue mayúsculas (evita 'sam', 'ron' minúsculas) pero acepta la forma de presentación del guion."""
    forms = sorted({f for n in names for f in (n, n.upper())}, key=len, reverse=True)
    if not forms:
        return None
    return re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(f) for f in forms) + r")(?![\w-])")
