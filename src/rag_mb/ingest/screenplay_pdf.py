"""Guion (PDF en formato estándar de la industria) → escenas citables con hablantes.

Clasificación por geometría, no por heurísticas de texto: en un guion estándar la
coordenada x0 de cada línea determina su tipo (72 pt = 1"):

    acción ≈ 1.5"   diálogo ≈ 2.5"   acotación ≈ 3.1"   personaje ≈ 3.7"   transición ≈ 6"

Los umbrales están en configs/corpora.yaml (screenplay_layout) y se pueden ajustar por
documento. Las sluglines (INT./EXT.) abren escena; "(MORE)"/"(CONT'D)" se fusionan.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import Line, Unit
from ..text import canonical_speaker, normalize

log = logging.getLogger(__name__)

# Guiones de rodaje numerados: "2 EXT. ARRAKEEN TARMAC - MORNING 2", "1E ARRAKIS FROM SPACE 1E"
NUMBERED_SLUG = re.compile(r"^(\d+[A-Z]{0,2})\s+(.+?)\s+\1$")
SLUG = re.compile(r"^(?:\d+[A-Z]?\s+)?((?:INT|EXT|INT\./EXT|EXT\./INT|I/E)\.?\s.*?)(?:\s+\d+[A-Z]?)?$")
SKIP = re.compile(r"^\(?(CONTINUED|MORE)\)?:?(\s*\(\d+\))?$|^\d+[A-Z]?\.?$|^\(?CONT'D\)?$", re.I)
TRANSITION = re.compile(r"(?:CUT|DISSOLVE|SMASH CUT|MATCH CUT|FADE)(?: TO)?(?: BLACK)?[:.]?$|FADE (?:IN|OUT|UP)[:.]?", re.I)

DEFAULT_LAYOUT = {"action_max_x": 140, "dialogue_max_x": 205,
                  "parenthetical_max_x": 240, "character_max_x": 400}


@dataclass
class _Scene:
    heading: str
    action: list[str] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    body: list[str] = field(default_factory=list)   # orden original (acción y diálogo)


def _is_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.9


def extract_layout_lines(pdf_path: Path) -> list[tuple[float, str]]:
    """[(x0, texto)] para todas las páginas, sin cabecera/pie."""
    import pdfplumber

    out: list[tuple[float, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            h = float(page.height)
            for ln in page.extract_text_lines(keep_blank_chars=False):
                if ln["top"] < 0.06 * h or ln["bottom"] > 0.96 * h:
                    continue
                # marcas de revisión de Final Draft al final de la línea ("Mother. *")
                out.append((float(ln["x0"]), re.sub(r"\s*\*+$", "", normalize(ln["text"]))))
    return out


# ----------------------------------------------------------------------------- OCR (guiones escaneados)
def _ocr_tsv_to_lines(tsv: str, dpi: int, page_w_px: int, page_h_px: int,
                      min_conf: float = 30.0) -> list[tuple[float, str]]:
    """TSV de Tesseract → [(x0 en puntos PDF, texto)] por línea, sin cabecera/pie ni ruido de escaneo."""
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE))
    lines: dict = {}
    for r in rows:
        txt = (r.get("text") or "").strip()
        if r.get("level") != "5" or not txt:
            continue
        conf = float(r.get("conf") or -1)
        if conf < min_conf and not any(c.isalnum() for c in txt):
            continue                                   # manchas del escáner
        key = (r["block_num"], r["par_num"], r["line_num"])
        left, top = int(r["left"]), int(r["top"])
        lines.setdefault(key, []).append((left, top, int(r["height"]), txt))
    out = []
    for ws in lines.values():
        ws.sort()
        x0, top = ws[0][0], min(w[1] for w in ws)
        bottom = max(w[1] + w[2] for w in ws)
        if top < 0.075 * page_h_px or bottom > 0.96 * page_h_px:
            continue
        text = normalize(" ".join(w[3] for w in ws))
        if sum(c.isalnum() for c in text) < 1:
            continue
        out.append((top, x0 * 72.0 / dpi, text))
    return [(x, t) for _, x, t in sorted(out)]


def ocr_layout_lines(pdf_path: Path, dpi: int = 200) -> list[tuple[float, str]]:
    """OCR con pdftoppm + tesseract (binarios del sistema). Devuelve líneas con su x0 en puntos."""
    import shutil
    import subprocess
    import tempfile

    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        raise RuntimeError(
            f"{pdf_path.name} no tiene capa de texto (PDF escaneado). Se necesita OCR: instalar Tesseract y "
            f"Poppler, o colocar el archivo precalculado {pdf_path.name}.layout.json junto al PDF.")
    from PIL import Image

    out: list[tuple[float, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf_path), f"{tmp}/p"], check=True)
        for png in sorted(Path(tmp).glob("p-*.png")):
            w, h = Image.open(png).size
            tsv = subprocess.run(["tesseract", str(png), "stdout", "-l", "eng", "--psm", "4", "tsv"],
                                 check=True, capture_output=True, text=True).stdout
            out.extend(_ocr_tsv_to_lines(tsv, dpi, w, h))
    return out


def layout_lines(pdf_path: Path) -> list[tuple[float, str]]:
    """Fuente única de líneas con geometría: capa de texto → caché OCR (.layout.json) → OCR."""
    import json

    lines = extract_layout_lines(pdf_path)
    if sum(len(t) for _, t in lines) > 1000:
        return lines
    cache = pdf_path.with_name(pdf_path.name + ".layout.json")
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
    log.warning("%s: sin capa de texto, ejecutando OCR (lento)…", pdf_path.name)
    lines = ocr_layout_lines(pdf_path)
    cache.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
    return lines


def classify(x0: float, text: str, layout: dict) -> str:
    t = text.strip()
    if not t or SKIP.match(t):
        return "skip"
    if x0 > layout["character_max_x"]:
        return "transition"
    if x0 <= layout["action_max_x"]:
        if (SLUG.match(t) or NUMBERED_SLUG.match(t)) and _is_upper(t.split(" - ")[0]):
            return "slug"
        if TRANSITION.fullmatch(t):
            return "transition"
        return "action"
    if x0 <= layout["dialogue_max_x"]:
        return "dialogue"
    if x0 <= layout["parenthetical_max_x"]:
        if t.startswith("("):
            return "paren"
        return "cue" if _is_upper(t) else "dialogue"
    return "cue" if _is_upper(t) else "dialogue"


def parse_lines(layout_lines: list[tuple[float, str]], layout: dict | None = None) -> list[_Scene]:
    layout = {**DEFAULT_LAYOUT, **(layout or {})}
    scenes = [_Scene(heading="(SIN ENCABEZADO)")]
    speaker, paren, last = None, None, None
    for x0, text in layout_lines:
        kind = classify(x0, text, layout)
        sc = scenes[-1]
        if kind in ("skip", "transition"):
            continue
        if kind == "slug":
            t = text.strip()
            m = NUMBERED_SLUG.match(t)
            heading = m.group(2) if m else SLUG.match(t).group(1)
            scenes.append(_Scene(heading=heading.strip()))
            speaker, last = None, "slug"
        elif kind == "action":
            sc.action.append(text)
            sc.body.append(text)
            speaker, last = None, "action"
        elif kind == "cue":
            name = canonical_speaker(text)
            if not name:
                continue
            cont = "CONT" in text.upper()
            # (MORE)/(CONT'D) a través de salto de página: continuar la misma línea
            if cont and last in ("dialogue", "paren") and sc.lines and sc.lines[-1].speaker == name:
                speaker = name
            else:
                speaker = name
                sc.lines.append(Line(speaker=name, text=""))
            paren, last = None, "cue"
        elif kind == "paren":
            paren, last = text.strip("()"), "paren"
            if sc.lines and speaker:
                sc.lines[-1].parenthetical = paren
        elif kind == "dialogue":
            if speaker is None or not sc.lines:
                sc.action.append(text)          # texto huérfano: tratar como acción
                sc.body.append(text)
                continue
            sc.lines[-1].text = (sc.lines[-1].text + " " + text).strip()
            last = "dialogue"

    for sc in scenes:   # reconstruir cuerpo legible en orden aproximado
        sc.body = sc.action + [f"{ln.speaker}: {ln.text}" for ln in sc.lines if ln.text]
    return [s for s in scenes if s.action or s.lines]


def parse_screenplay(pdf_path: Path, *, saga: str, source: dict, layout: dict | None = None,
                     order_offset: int = 0) -> list[Unit]:
    scenes = parse_lines(layout_lines(pdf_path), {**(layout or {}), **source.get("layout", {})})
    units: list[Unit] = []
    for i, sc in enumerate(scenes):
        lines = [ln for ln in sc.lines if ln.text]
        action = " ".join(sc.action)
        dialogue = "\n".join(f"{ln.speaker}: {ln.text}" for ln in lines)
        text = (f"[{sc.heading}]\n" + action + ("\n" + dialogue if dialogue else "")).strip()
        units.append(Unit(
            unit_id=f"{source['id']}:s{i:04d}",
            saga=saga, source_id=source["id"], medium="screenplay", work=source["work"],
            order=order_offset + i,
            locator=f"Escena {i} — {sc.heading}",
            title=sc.heading, text=text, lines=lines,
            meta={"scene": i, "n_lines": len(lines), "action_words": len(action.split())},
        ))
    log.info("%s: %d escenas, %d líneas de diálogo", source["id"], len(units),
             sum(len(u.lines) for u in units))
    return units
