"""Pruebas de ingesta con fixtures sintéticos (sin PDFs con derechos de autor)."""
from rag_mb.ingest.book_pdf import _segment_numbered, clean_pages
from rag_mb.ingest.chunking import chunk_unit
from rag_mb.ingest.screenplay_pdf import classify, parse_lines
from rag_mb.schema import Unit
from rag_mb.text import canonical_speaker
import re

LAYOUT = {"action_max_x": 140, "dialogue_max_x": 205, "parenthetical_max_x": 240, "character_max_x": 400}


def test_screenplay_geometry():
    lines = [
        (108, "EXT. SHIRE - DAY"),
        (108, "FRODO reads under a tree."),
        (270, "GANDALF"),
        (225, "(smiling)"),
        (180, "A wizard is never late,"),
        (180, "Frodo Baggins."),
        (468, "CUT TO:"),
        (108, "INT. BAG END - NIGHT"),
        (270, "FRODO (CONT'D)"),
        (180, "Is it secret?"),
        (555, "12."),
    ]
    scenes = parse_lines(lines, LAYOUT)
    assert [s.heading for s in scenes] == ["EXT. SHIRE - DAY", "INT. BAG END - NIGHT"]
    assert scenes[0].lines[0].speaker == "GANDALF"
    assert scenes[0].lines[0].text == "A wizard is never late, Frodo Baggins."
    assert scenes[0].lines[0].parenthetical == "smiling"
    assert scenes[1].lines[0].speaker == "FRODO"
    assert classify(468, "CUT TO:", LAYOUT) == "transition"


def test_canonical_speaker():
    assert canonical_speaker("GALADRIEL (V.O.) (CONT’D)") == "GALADRIEL"
    assert canonical_speaker("THÉODEN (cont’d)") == "THEODEN"


def test_book_cleaning_and_segmentation():
    pages = ["Chapter 1\nA LONG-EXPECTED\nPARTY\nWhen Mr. Bilbo announced a mag-\nnificent party.",
             "22 THE LORD OF THE RINGS\nThe party was grand.\n",
             "BOOK TWO\nChapter 1\nMANY MEETINGS\nFrodo woke."]
    text = clean_pages(pages)
    assert "magnificent" in text and "THE LORD OF THE RINGS" not in text
    chs = _segment_numbered(text.split("\n"), re.compile(r"^Chapter (\d+)$"),
                            re.compile(r"^BOOK (ONE|TWO)$"), None)
    assert [(c.part, c.number) for c in chs] == [(0, 1), (2, 1)]
    assert chs[0].title == "A LONG-EXPECTED PARTY"


def test_chunk_inherits_citation():
    u = Unit("x:c01", "lotr", "x", "book", "fotr", 0, "Cap. 1", "T",
             "\n\n".join(" ".join(["word"] * 100) for _ in range(5)))
    ch = chunk_unit(u, max_words=220, overlap_words=40)
    assert len(ch) >= 2 and all(c.locator == "Cap. 1" for c in ch)
    assert all(len(c.text.split()) <= 220 for c in ch)


def test_diacritic_cleanup_keeps_line_breaks():
    # regresión: la limpieza de diacríticos sueltos no debe unir el encabezado con el título
    text = clean_pages(["Chapter 1 ́\nTHE TAMING OF SME´AGOL\nBody."])
    assert text.split("\n")[0].strip() == "Chapter 1"
    assert "SMEAGOL" in text


def test_letterspaced_titles():
    from rag_mb.ingest.book_pdf import fix_letterspaced
    assert fix_letterspaced("A u n t M a r g e ’ sB i g M i s t a k e") == "Aunt Marge’s Big Mistake"
    assert fix_letterspaced("T h e D e p a r t m e n t o f M y s t e r i e s") == "The Department of Mysteries"
    assert fix_letterspaced("T h e I n v i t a t i o n") == "The Invitation"
    assert fix_letterspaced("The Burrow") == "The Burrow"


def test_book_quote_attribution_single_and_double_quotes():
    import pandas as pd
    from rag_mb.analysis.quotes import QUOTE_SINGLE, quote_regex
    t = "'They have left us fruit,' said Pippin. 'Don't you think so?' Frodo asked."
    assert quote_regex(t) is QUOTE_SINGLE
    qs = [m.group(1) for m in QUOTE_SINGLE.finditer(t)]
    assert qs == ["They have left us fruit,", "Don't you think so?"]


def test_chunking_never_emits_overlap_only_chunks():
    """Regresión: capítulo sin saltos de párrafo (PDF) → no deben quedar restos de palabras repetidas."""
    from rag_mb.ingest.chunking import chunk_unit
    from rag_mb.schema import Unit
    words = [f"w{i}" for i in range(1000)]
    u = Unit(unit_id="b:c1", saga="x", source_id="s", medium="book", work="w", order=0,
             locator="Cap. 1", title="", text=" ".join(words))
    ch = [c.text.split() for c in chunk_unit(u, max_words=220, overlap_words=40)]
    assert all(len(c) > 40 for c in ch)
    assert ch[0][-40:] == ch[1][:40]                                  # solapamiento correcto
    covered = [w for i, c in enumerate(ch) for w in (c if i == 0 else c[40:])]
    assert covered == words                                           # sin huecos ni duplicados
