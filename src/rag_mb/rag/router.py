"""Enrutador de consultas ("self-query"): convierte menciones de saga, medio y obra en FILTROS de metadatos
sobre un índice único (no hay un índice por saga).

"¿Cómo muere Boromir?"             → saga=lotr (por el personaje)
Sin pistas ("¿Quién cuida la posada del Poney Pisador?") → el índice vota con una búsqueda preliminar
"In the book, ..." / "En el libro" → medium=book
"In The Two Towers film, ..."      → saga=lotr, medium=screenplay, work=ttt
"... en el libro frente a la película" → recuperación balanceada (k/2 libro + k/2 película)
"Is X part of the film?"           → también balanceada: la ausencia se prueba con el otro medio

Las palabras de medio se eliminan de la consulta: "book"/"película" no aportan señal léxica
y desplazan el ranking de BM25 hacia pasajes que las contienen por accidente.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

BOOK = re.compile(r"\b(book|books|novel|tolkien|herbert|lewis|rowling|chapter|libro|libros|novela|"
                  r"cap[ií]tulo)\b", re.I)
FILM = re.compile(r"\b(film|films|movie|movies|screenplay|script|adaptation|jackson|villeneuve|"
                  r"pel[ií]cula|pel[ií]culas|guion|gui[oó]n|filme|cinta)\b", re.I)
META = re.compile(r"\b(in|according to|of)?\s*the\s+(book|books|novel|film|films|movie|movies|screenplay|script|"
                  r"film adaptation)\b|\b(compared (with|to)|versus|vs\.?|and how does .* differ)\b|"
                  r"\b(en|seg[uú]n|de)?\s*(el|la|los|las)\s+(libro|libros|novela|pel[ií]cula|pel[ií]culas|"
                  r"guion|gui[oó]n|filme|adaptaci[oó]n)\b|\b(comparad[oa]s? con|frente a)\b", re.I)

WORK_PATTERNS = {
    "lotr": {"fotr": r"fellowship of the ring|comunidad del anillo",
             "ttt": r"two towers|dos torres",
             "rotk": r"return of the king|retorno del rey"},
    "dune": {"dune_p1": r"part one|part 1|parte uno|parte 1|primera parte",
             "dune_p2": r"part two|part 2|parte dos|parte 2|segunda parte"},
    "hp": {"hp1": r"philosopher'?s stone|sorcerer'?s stone|piedra filosofal",
           "hp2": r"chamber of secrets|c[aá]mara secreta", "hp3": r"prisoner of azkaban|prisionero de azkaban",
           "hp4": r"goblet of fire|c[aá]liz de fuego", "hp5": r"order of the phoenix|orden del f[eé]nix",
           "hp6": r"half-blood prince|pr[ií]ncipe mestizo|misterio del pr[ií]ncipe",
           "hp7": r"(?:deathly hallows|reliquias de la muerte)(?!,?\s*(?:-|–)?\s*(?:part|parte))",
           "hp7a": r"(?:deathly hallows|reliquias de la muerte),?\s*(?:-|–)?\s*(?:part|parte) (?:1|one|uno)",
           "hp7b": r"(?:deathly hallows|reliquias de la muerte),?\s*(?:-|–)?\s*(?:part|parte) (?:2|two|dos)"},
    "narnia": {"lww": r"lion, the witch|wardrobe|le[oó]n, la bruja|ropero|armario",
               "caspian": r"prince caspian|pr[ií]ncipe caspian", "dawn_treader": r"dawn treader|viajero del alba"},
}
# Nombre de la saga que, una vez usado como filtro, solo mete ruido léxico porque es palabra común
# ("Dune" → "dune", duna: atrae pasajes del desierto). Sensible a mayúsculas: solo el título.
# No se incluye "Harry Potter" (es el protagonista) ni "Narnia" (es el lugar).
SAGA_TITLE_NOISE = {"dune": r"\bDune\b:?"}

# Preguntas de PRESENCIA/AUSENCIA ("Is X part of the film?", "¿Aparece X en el libro?"): la evidencia de que
# algo NO está en un medio vive en el OTRO medio, así que se busca en ambos aunque solo se nombre uno.
PRESENCE = re.compile(
    r"^\W*(?:is|are|was|were|does|do|did|has|have)\b.*\b(?:part of|appear(?:s|ed)?|included|include[sd]?|"
    r"feature[sd]?|shown|exist(?:s|ed)?)\b|"
    r"^\W*(?:aparece|aparecen|est[aá]|existe|hay|sale|salen|forma parte|incluye)\b", re.I)

# Señales de saga (peso alto). Los títulos de obra también cuentan; los personajes suman de a uno.
SAGA_PATTERNS = {
    "lotr": r"lord of the rings|se[nñ]or de los anillos|tolkien|middle[- ]earth|tierra media|hobbits?|mordor|"
            r"\bshire\b|\bcomarca\b|one ring|anillo [uú]nico|rivendell|gondor|rohan|helm'?s deep|abismo de helm|"
            r"isengard|moria|minas tirith|lothl[oó]rien|mount doom|monte del destino|amon hen|bag end|bols[oó]n cerrado|"
            r"army of the dead|paths of the dead|ej[eé]rcito de los muertos|dunharrow|grey havens|"
            r"puertos grises|scouring of the shire|saneamiento de la comarca|nazg[uû]l|ringwraiths?",
    "hp": r"harry potter|hogwarts|rowling|quidditch|horcrux|horrocrux|muggle|gryffindor|slytherin|hogsmeade|"
          r"azkaban|diagon|callej[oó]n diagon|death eaters?|mort[ií]fagos?|patronus",
    "dune": r"\bdune\b|arrakis|fremen|melange|\bspice\b|especia|atreides|harkonnen|villeneuve|bene gesserit|"
            r"caladan|giedi prime|sietch|sandworms?|gusanos? de arena|kwisatz haderach|shai-hulud",
    "narnia": r"narnia|c\.\s*s\.\s*lewis|pevensie|cair paravel|stone table|mesa de piedra|beruna|turkish delight",
}
# Nombres que en inglés son palabras comunes o títulos: no sirven para decidir la saga.
GENERIC_NAMES = {"duke", "baron", "emperor", "witch", "fox", "professor", "lion", "faun", "dwarf", "merry",
                 "reverend mother", "dark lord", "padishah", "liet", "fangorn"}


@dataclass
class Route:
    query: str
    medium: str | None
    work: str | None
    balanced: bool
    saga: str | None = None
    saga_scores: dict | None = None
    hyde: str | None = None          # pasaje hipotético usado para buscar (si se activó HyDE)


@lru_cache(maxsize=1)
def _character_patterns() -> dict[str, list[re.Pattern]]:
    from .. import characters
    out = {}
    for saga in characters.load_characters():
        pats = []
        for names in characters.mentions(saga).values():
            ok = [n for n in names if n.lower() not in GENERIC_NAMES and len(n) >= 3]
            if ok:
                alt = "|".join(re.escape(n) for n in sorted(ok, key=len, reverse=True))
                pats.append(re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])", re.I))
        out[saga] = pats
    return out


def saga_scores(question: str) -> dict[str, int]:
    """Puntaje por saga: palabra clave de la saga = 3, título de obra = 3, cada personaje distinto = 1.
    'Part One/Two' no decide la saga (es ambiguo); solo filtra la obra una vez conocida la saga."""
    sc = {}
    for saga in set(SAGA_PATTERNS) | set(WORK_PATTERNS):
        s = 3 * bool(re.search(SAGA_PATTERNS.get(saga, r"(?!)"), question, re.I))
        if saga != "dune":
            s += 3 * sum(bool(re.search(p, question, re.I)) for p in WORK_PATTERNS.get(saga, {}).values())
        s += sum(bool(p.search(question)) for p in _character_patterns().get(saga, []))
        sc[saga] = s
    return sc


def detect_saga(question: str) -> tuple[str | None, dict[str, int]]:
    """Saga con mayor puntaje, solo si gana sin empate. Si nadie puntúa, None (se busca en todo)."""
    sc = saga_scores(question)
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    if ranked and ranked[0][1] > 0 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
        return ranked[0][0], sc
    return None, sc


def route(question: str, saga: str | None) -> Route:
    """saga=None → se detecta desde la pregunta (índice global)."""
    scores = None
    if saga is None:
        saga, scores = detect_saga(question)
    b, f = bool(BOOK.search(question)), bool(FILM.search(question))
    works = [w for w, p in WORK_PATTERNS.get(saga, {}).items() if re.search(p, question, re.I)] if saga else []
    medium = "book" if b and not f else "screenplay" if f and not b else None
    balanced = b and f
    if (b or f) and PRESENCE.search(question):
        medium, balanced = None, True          # "¿aparece X en la película?" → libro Y película
    # La obra se devuelve aunque no haya medio explícito: el índice decide si solo existe como
    # película ("Dune: Part Two") y en ese caso filtra al lado película.
    work = works[0] if len(works) == 1 else None
    # El título de la obra ya actuó como filtro (saga / obra): en la consulta solo mete ruido léxico
    # ("lion, witch, wardrobe" atrae capítulos que nombran el ropero, no la escena preguntada).
    q = question
    for p in WORK_PATTERNS.get(saga, {}).values():
        q = re.sub(rf"(?:\bthe\s+)?(?:{p})(?:\s+the)?", " ", q, flags=re.I)
    if saga in SAGA_TITLE_NOISE:
        q = re.sub(rf"(?:\bthe\s+)?{SAGA_TITLE_NOISE[saga]}", " ", q)
    q = META.sub(" ", re.sub(r"\s+", " ", q))
    q = re.sub(r"\b(?:book|books|novel|film|films|movie|movies|screenplay|libro|libros|novela|"
               r"pel[ií]cula|pel[ií]culas|guion|gui[oó]n)\b", " ", q, flags=re.I)   # medio suelto: ya es filtro
    q = re.sub(r"\b(?:in|of)\s+(?=[,?]|$)", " ", q.strip(), flags=re.I)   # "in" colgando al final
    return Route(re.sub(r"\s+", " ", q).strip(" ,?¿") + "?", medium, work, balanced, saga, scores)
