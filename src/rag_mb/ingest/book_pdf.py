"""Libro (PDF) → capítulos citables.

Pasos:
1. Extracción por página (pypdf, puro Python).
2. Limpieza de cabeceras corridas ("22 TH E L ORD OF THE RI NGS"), números de página y
   guiones de corte de línea ("mag-\\nniﬁcence" → "magniﬁcence").
3. Segmentación con regex configurables (configs/corpora.yaml):
   - part_regex     → "BOOK ONE" ... (mapea a la obra/película vía `parts`)
   - chapter_regex  → "Chapter 1"  (el título es la siguiente línea no vacía)
   - epigraph_regex → modo Dune: la novela no numera capítulos; cada uno abre con un epígrafe
                      "—from ..." y el corte se hace al inicio del párrafo del epígrafe.
   - stop_regex     → fin del texto narrativo (apéndices).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..schema import Unit
from ..text import normalize

log = logging.getLogger(__name__)

_WORDNUM = {w: i for i, w in enumerate(
    "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN "
    "SIXTEEN SEVENTEEN EIGHTEEN NINETEEN TWENTY".split(), start=1)}
_HEADER = re.compile(r"^\s*\d+\s+[A-Z][A-Z\s’'\-,.:;!?]*$|^[A-Z][A-Z\s’'\-,.:;!?]*\s+\d+\s*$|^\s*\d+\s*$")


def extract_pages(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader  # import local: dependencia solo para ingesta

    reader = PdfReader(str(pdf_path))
    return [(p.extract_text() or "") for p in reader.pages]


def clean_pages(pages: list[str], protect: re.Pattern | None = None) -> str:
    """Quita cabecera/pie de página y une páginas; resuelve cortes con guion.

    protect: líneas que NO deben tratarse como cabecera aunque lo parezcan
    (p.ej. "CHAPTER 1" al inicio de página tiene la forma "TEXTO número").
    """
    out: list[str] = []
    for page in pages:
        lines = [ln.rstrip() for ln in page.split("\n")]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and _HEADER.match(lines[0]) and not (protect and protect.match(lines[0].strip())):
            lines.pop(0)
        if lines and re.fullmatch(r"\s*\d+\s*", lines[-1]):
            lines.pop()
        out.append("\n".join(lines))
    text = "\n".join(out)
    # pypdf separa diacríticos como caracteres sueltos ("Khazad-Du ˆm", "Lo ́rien"): se eliminan
    # (solo espacios horizontales: \s también comería saltos de línea y pegaría encabezados)
    text = re.sub(r"[ \t]?[\u02c6\u02dc\u00b4\u00a8\u02d8\u02c7\u0300-\u036f][ \t]?", "", text)
    text = re.sub(r"(\w)-\n([a-zß-ÿ])", r"\1\2", text)   # de-hyphenation
    return text


def _roman_or_int(tok: str) -> int:
    tok = re.sub(r"\s+", "", tok).upper()
    if tok.isdigit():
        return int(tok)
    if tok in _WORDNUM:
        return _WORDNUM[tok]
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, prev = 0, 0
    for ch in reversed(tok):
        v = vals.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def fix_letterspaced(title: str) -> str:
    """'A u n t M a r g e ’ sB i g M i s t a k e' → 'Aunt Marge’s Big Mistake'.

    Si la mayoría de los tokens son letras sueltas, se colapsan los espacios y se recuperan las
    fronteras de palabra por las mayúsculas iniciales (los títulos vienen en Title Case).
    """
    toks = title.split()
    if not toks or sum(len(t) == 1 for t in toks) / len(toks) < 0.6:
        return title
    s = "".join(toks)
    s = re.sub(r"(?<=[a-z’'])(?=[A-Z])", " ", s)
    s = re.sub(r",(?=\S)", ", ", s)
    # palabras funcionales pegadas en minúscula: "Departmentof" → "Department of", "Eyeofthe" → "Eye of the"
    # (no se incluyen "on"/"at"/"in": romperían "Invitation", "Great", ...)
    for _ in range(3):
        s = re.sub(r"(?<=[A-Za-z]{3})(of|the|and|for|to)(?=(?:of|the|and|for|to)?(?:\s|$))", r" \1", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class _Chapter:
    part: int
    number: int
    title: str
    lines: list[str]


def _segment_numbered(lines, chapter_re, part_re, stop_re, extra_re=None,
                      stop_on_restart: bool = False, part_on_restart: bool = False) -> list[_Chapter]:
    """extra_re: encabezado sin número (p.ej. epílogo) → capítulo siguiente.
    stop_on_restart: si la numeración vuelve a empezar DESPUÉS de capítulos con cuerpo (adelanto del
    siguiente libro al final del PDF), se detiene. El índice no cuenta: sus entradas no tienen cuerpo."""
    chapters: list[_Chapter] = []
    part, cur, pending_title = 0, None, False
    for ln in lines:
        s = ln.strip()
        if stop_re and stop_re.match(s):
            break
        if part_re and (m := part_re.match(s)):
            part = _roman_or_int(m.group(1))
            continue
        m = chapter_re.match(s)
        if m or (extra_re and extra_re.match(s) and chapters):
            # Modo separador (regex sin grupos, p.ej. "= = = = = =" en Dune): numeración automática por parte
            separator = bool(m) and chapter_re.groups == 0
            if m and not separator:
                num = _roman_or_int(m.group(1))
            else:
                num = 1 + sum(1 for c in chapters if c.part == part) if separator else chapters[-1].number + 1
            real = [c.number for c in chapters if len(" ".join(c.lines).split()) >= 50]
            if stop_on_restart and m and real and num <= real[-1] and cur and len(" ".join(cur.lines).split()) >= 50:
                break
            # part_on_restart: varios libros en un PDF (ómnibus); cada reinicio en 1 abre un libro nuevo
            if part_on_restart and m and not separator and num == 1:
                part += 1
            title = (m.group(2) or "").strip() if m and m.lastindex and m.lastindex >= 2 else ("" if m else s)
            cur = _Chapter(part, num, fix_letterspaced(title), [])
            chapters.append(cur)
            pending_title = not title and not separator
            continue
        if cur is None:
            continue
        if pending_title and s:
            cur.title, pending_title = fix_letterspaced(s), False
            continue
        # Títulos en mayúsculas partidos en dos líneas: "THE BREAKING OF" / "THE FELLOWSHIP"
        if not cur.lines and cur.title.isupper() and s.isupper() and len(s.split()) <= 6:
            cur.title = f"{cur.title} {s}"
            continue
        cur.lines.append(ln)
    return chapters


def _segment_epigraphs(lines, epi_re, part_re, stop_re) -> list[_Chapter]:
    """Corta antes del bloque de epígrafe que precede a cada línea '—from ...'."""
    body: list[tuple[int, str]] = []
    part, started = 0, part_re is None
    for ln in lines:
        s = ln.strip()
        if stop_re and stop_re.match(s):
            break
        if part_re and (m := part_re.match(s)):
            part, started = _roman_or_int(m.group(1)), True
            body.append((part, "\x00PART"))
            continue
        if started:
            body.append((part, ln))

    cuts = []
    for i, (_, ln) in enumerate(body):
        if epi_re.match(ln.strip()):
            # Retroceder hasta el inicio del bloque del epígrafe: primera línea vacía hacia
            # atrás (máx. 12 líneas, los epígrafes de Dune son cortos).
            j = i
            while j > 0 and body[j - 1][1].strip() and body[j - 1][1] != "\x00PART" and (i - j) < 12:
                j -= 1
            cuts.append(j)
    cuts = sorted(set(cuts))
    chapters, counters = [], {}
    for k, start in enumerate(cuts):
        end = cuts[k + 1] if k + 1 < len(cuts) else len(body)
        seg = [ln for _, ln in body[start:end] if ln != "\x00PART"]
        p = body[start][0]
        counters[p] = counters.get(p, 0) + 1
        attribution = next((ln.strip() for ln in seg if epi_re.match(ln.strip())), "")
        chapters.append(_Chapter(p, counters[p], attribution[:80], seg))
    return chapters


def parse_book(pdf_path: Path, *, saga: str, source: dict, order_offset: int = 0) -> list[Unit]:
    pages = extract_pages(pdf_path)
    protect = re.compile(source["chapter_regex"]) if source.get("chapter_regex") else None
    text = normalize_lines(clean_pages(pages, protect))
    lines = text.split("\n")

    part_re = re.compile(source["part_regex"]) if source.get("part_regex") else None
    stop_re = re.compile(source["stop_regex"]) if source.get("stop_regex") else None
    if source.get("chapter_regex"):
        extra_re = re.compile(source["extra_heading_regex"]) if source.get("extra_heading_regex") else None
        chapters = _segment_numbered(lines, re.compile(source["chapter_regex"]), part_re, stop_re,
                                     extra_re, bool(source.get("stop_on_restart")),
                                     bool(source.get("part_on_restart")))
        # Con el índice al inicio, cada número aparece dos veces: se conserva la última aparición con cuerpo
        if source.get("stop_on_restart"):
            keep = {}
            for ch in chapters:
                if len(" ".join(ch.lines).split()) >= 50:
                    keep[(ch.part, ch.number)] = ch
            chapters = sorted(keep.values(), key=lambda c: (c.part, c.number))
    elif source.get("epigraph_regex"):
        chapters = _segment_epigraphs(lines, re.compile(source["epigraph_regex"]), part_re, stop_re)
    else:
        raise ValueError(f"{source['id']}: definir chapter_regex o epigraph_regex")

    part_to_work = {}
    for work, parts in (source.get("parts") or {}).items():
        for p in parts:
            part_to_work[p] = work
    default_work = source.get("work", saga)

    units: list[Unit] = []
    epi_re = re.compile(source["epigraph_regex"]) if source.get("epigraph_regex") else None
    for ch in chapters:
        if not ch.title and epi_re:
            attr = next((ln.strip() for ln in ch.lines[:40] if epi_re.match(ln.strip())), "")
            ch.title = re.sub(r"^[-—–\s]*from\s+", "", attr).split(" by ")[0].strip('" ')[:60]
    # renumerar tras descartar fragmentos vacíos (separador pegado al título de la parte)
    chapters = [c for c in chapters if len(" ".join(c.lines).split()) >= 50]
    if source.get("chapter_regex") and re.compile(source["chapter_regex"]).groups == 0:
        seen: dict = {}
        for c in chapters:
            seen[c.part] = seen.get(c.part, 0) + 1
            c.number = seen[c.part]
    for i, ch in enumerate(chapters):
        body = _reflow(ch.lines)
        if len(body.split()) < 50:      # basura: índices, páginas en blanco
            continue
        work = part_to_work.get(ch.part, default_work)
        part_tag = f"b{ch.part}" if ch.part else "b0"
        units.append(Unit(
            unit_id=f"{source['id']}:{work}:{part_tag}:c{ch.number:02d}",
            saga=saga, source_id=source["id"], medium="book", work=work,
            order=order_offset + i,
            locator=(f"Libro {ch.part}, Cap. {ch.number}" if ch.part else f"Cap. {ch.number}")
                    + (f" — {ch.title.title()}" if ch.title else ""),
            title=ch.title.title(), text=body,
            meta={"part": ch.part, "chapter": ch.number},
        ))
    log.info("%s: %d capítulos", source["id"], len(units))
    return units


def normalize_lines(text: str) -> str:
    return "\n".join(normalize(ln) if ln.strip() else "" for ln in text.split("\n"))


def _reflow(lines: list[str]) -> str:
    """Une líneas físicas en párrafos (línea vacía = corte de párrafo)."""
    paras, cur = [], []
    for ln in lines:
        if ln.strip() in {"", ".", "* * *"}:
            if cur:
                paras.append(" ".join(cur))
                cur = []
        else:
            cur.append(ln.strip())
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)
