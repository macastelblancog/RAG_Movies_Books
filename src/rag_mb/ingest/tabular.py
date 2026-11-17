"""Fuentes tabulares y subtítulos: CSV de diálogos (Kaggle/Maven) y .srt (sin hablante)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..schema import Line, Unit
from ..text import canonical_speaker, normalize


def parse_dialogue_csv(path: Path, *, saga: str, source: dict, block: int = 15,
                       order_offset: int = 0) -> list[Unit]:
    """Agrupa líneas consecutivas de la misma película en bloques de `block` líneas.

    El CSV de Kaggle no trae escenas; los bloques son unidades de conveniencia para conteos
    por personaje. No se indexan en el RAG por defecto (el guion PDF es más rico).
    """
    cols = source["columns"]
    df = pd.read_csv(path, encoding="utf-8-sig", sep=source.get("sep", ","))
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}            # HP3 trae CHARACTER;SENTENCE en mayúsculas
    rename = {lower[v.lower()]: k for k, v in cols.items() if v.lower() in lower}
    df = df.rename(columns={k: {"speaker": "speaker", "text": "text", "work": "work"}[v] for k, v in rename.items()})
    if "work" not in df.columns:                          # un CSV por película
        df["work"] = source["work"]
    df["work"] = df["work"].astype(str).str.strip().map(lambda w: source.get("work_map", {}).get(w, w))
    df = df.dropna(subset=["text"])
    df["speaker"] = df["speaker"].fillna("UNKNOWN").astype(str).map(canonical_speaker)
    df["text"] = df["text"].astype(str).map(normalize)
    units: list[Unit] = []
    for work, g in df.groupby("work", sort=False):
        g = g.reset_index(drop=True)
        for b in range(0, len(g), block):
            part = g.iloc[b:b + block]
            lines = [Line(speaker=s, text=t) for s, t in zip(part["speaker"], part["text"])]
            units.append(Unit(
                unit_id=f"{source['id']}:{work}:k{b // block:04d}",
                saga=saga, source_id=source["id"], medium="dialogue_csv", work=work,
                order=order_offset + len(units), locator=f"{work} — filas {b}-{b + len(part) - 1}",
                title=f"{work} bloque {b // block}",
                text="\n".join(f"{ln.speaker}: {ln.text}" for ln in lines), lines=lines,
                meta={"rows": [int(b), int(b + len(part) - 1)]},
            ))
    return units


_SRT_BLOCK = re.compile(r"(\d+)\s*\n(\d\d:\d\d:\d\d[,.]\d+)\s*-->\s*(\d\d:\d\d:\d\d[,.]\d+)\s*\n(.*?)(?=\n\s*\n|\Z)", re.S)


def _secs(ts: str) -> float:
    h, m, s = ts.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(path: Path, *, saga: str, source: dict, gap_s: float = 8.0, max_words: int = 250,
              order_offset: int = 0) -> list[Unit]:
    """Subtítulos → pseudo-escenas: corte cuando hay silencio > gap_s o se excede max_words."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    subs = [(_secs(a), _secs(b), normalize(re.sub(r"<[^>]+>", "", t))) for _, a, b, t in _SRT_BLOCK.findall(raw)]
    units, cur, start, prev_end = [], [], None, None

    def flush():
        if cur:
            i = len(units)
            units.append(Unit(
                unit_id=f"{source['id']}:p{i:04d}", saga=saga, source_id=source["id"],
                medium="subtitles", work=source["work"], order=order_offset + i,
                locator=f"{source['work']} {int(start // 60):02d}:{int(start % 60):02d}",
                title=f"t={int(start)}s", text=" ".join(cur), meta={"t_start": start, "t_end": prev_end},
            ))

    for a, b, t in subs:
        if cur and (a - prev_end > gap_s or sum(len(x.split()) for x in cur) > max_words):
            flush()
            cur, start = [], None
        if start is None:
            start = a
        cur.append(t)
        prev_end = b
    flush()
    return units


def parse_relational_dialogue(path: Path, *, saga: str, source: dict, order_offset: int = 0) -> list[Unit]:
    """Esquema relacional (Maven Analytics, Harry Potter Movies):
        Dialogue(Chapter ID, Place ID, Character ID, Dialogue)
        ⋈ Characters(Character ID → Character Name) ⋈ Chapters(Chapter ID → Chapter Name, Movie ID, Movie Chapter)
        ⋈ Places(Place ID → Place Name)
    Unidad citable = capítulo de la película (equivalente a la escena de un guion).
    medium = "dialogue": transcripción con hablante, sin acotaciones.
    """
    t = source["tables"]
    base = path.parent
    d = pd.read_csv(path, encoding="utf-8-sig")
    ch = pd.read_csv(base / t["characters"], encoding="utf-8-sig")
    cp = pd.read_csv(base / t["chapters"], encoding="utf-8-sig")
    pl = pd.read_csv(base / t["places"], encoding="utf-8-sig") if t.get("places") else None
    d = d.merge(ch[["Character ID", "Character Name"]], on="Character ID", how="left") \
         .merge(cp, on="Chapter ID", how="left")
    if pl is not None:
        d = d.merge(pl[["Place ID", "Place Name"]], on="Place ID", how="left")
    d = d.dropna(subset=["Dialogue"]).sort_values(["Movie ID", "Movie Chapter", "Dialogue ID"])
    work_map = {int(k): v for k, v in source["work_map"].items()}
    units: list[Unit] = []
    for (movie, mch), g in d.groupby(["Movie ID", "Movie Chapter"], sort=True):
        work = work_map[int(movie)]
        lines = [Line(speaker=canonical_speaker(str(sp)), text=normalize(str(tx)))
                 for sp, tx in zip(g["Character Name"].fillna("UNKNOWN"), g["Dialogue"])]
        name = str(g["Chapter Name"].iloc[0])
        places = list(dict.fromkeys(g["Place Name"].dropna())) if pl is not None else []
        head = f"[{name}" + (f" — {', '.join(places[:3])}" if places else "") + "]"
        units.append(Unit(
            unit_id=f"{source['id']}:{work}:ch{int(mch):02d}", saga=saga, source_id=source["id"],
            medium="dialogue", work=work, order=order_offset + len(units),
            locator=f"Capítulo {int(mch)} — {name}", title=name,
            text=head + "\n" + "\n".join(f"{ln.speaker}: {ln.text}" for ln in lines), lines=lines,
            meta={"movie_id": int(movie), "movie_chapter": int(mch), "places": places},
        ))
    return units
