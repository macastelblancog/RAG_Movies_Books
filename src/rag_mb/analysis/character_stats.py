"""Personajes en libro vs película: menciones, intervenciones y presencia.

Medidas por personaje y medio (libro = obras adaptadas; película = obras con libro disponible):
  menciones       veces que el texto lo nombra (libro: narración + citas; película: acotaciones +
                  diálogo, sin el nombre del hablante ni el encabezado de escena)
  intervenciones  libro: citas con atribución explícita ("…," said Frodo) — cobertura parcial;
                  película: líneas de diálogo con su nombre como hablante
  presencia       unidades (capítulos / escenas) donde es mencionado o habla
Las comparaciones usan PROPORCIONES dentro de cada medio (el libro siempre es más largo) y el z de
log-odds con prior de Dirichlet (Monroe et al., 2008) calculado SOLO sobre los conteos de personajes.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .. import characters
from ..config import saga_config
from ..schema import Unit
from .lexical import dirichlet_log_odds
from .pipeline import ALIASES, FILM_MEDIA

MEASURES = ("menciones", "intervenciones", "presencia")
MEDIO = {"book": "Libro", "film": "Película"}


def film_mention_text(u: Unit) -> str:
    """Texto de la escena sin encabezado ni nombres de hablante: acotaciones + parlamentos."""
    if u.medium == "subtitles":
        return u.text
    said = " ".join(ln.text for ln in u.lines)
    if u.medium != "screenplay" or not u.meta.get("action_words"):
        return said
    body = u.text.split("\n")
    action = body[1] if body and body[0].startswith("[") and len(body) > 1 else ""
    return action + " " + said


def paired_units(units: list[Unit], saga: str) -> tuple[list[Unit], list[Unit]]:
    """(libro, película) restringidos a lo comparable: libros adaptados y películas con libro."""
    bff = saga_config(saga).get("book_for_film", {})
    book_works = {u.work for u in units if u.medium == "book"}
    film = [u for u in units if u.medium in FILM_MEDIA and bff.get(u.work, u.work) in book_works]
    adapted = {bff.get(u.work, u.work) for u in film}
    book = [u for u in units if u.medium == "book" and u.work in adapted]
    return book, film


def unit_counts(units: list[Unit], saga: str) -> pd.DataFrame:
    """Una fila por (unidad, personaje) con menciones e intervenciones; base de todas las medidas."""
    pats = {c: characters.mention_regex(n) for c, n in characters.mentions(saga).items()}
    al = ALIASES.get(saga, {})
    bff = saga_config(saga).get("book_for_film", {})
    rows = []
    for pos, u in enumerate(units):
        film = u.medium != "book"
        text = film_mention_text(u) if film else u.text
        spoken = Counter(al.get(ln.speaker, ln.speaker) for ln in u.lines) if film else Counter()
        words = max(len(text.split()), 1)
        for c, p in pats.items():
            m = len(p.findall(text)) if p else 0
            if m or spoken.get(c):
                rows.append({"unit_id": u.unit_id, "pos": pos, "medium": "film" if film else "book",
                             "work": u.work, "pair": bff.get(u.work, u.work), "personaje": c,
                             "menciones": m, "intervenciones": spoken.get(c, 0), "words": words})
    return pd.DataFrame(rows, columns=["unit_id", "pos", "medium", "work", "pair", "personaje",
                                       "menciones", "intervenciones", "words"])


def book_interventions(units: list[Unit], saga: str) -> pd.DataFrame:
    from .quotes import book_quotes   # import diferido: quotes depende de analysis.pipeline
    q = book_quotes(units, saga)
    return q.groupby(["work", "speaker"]).size().rename("intervenciones").reset_index() \
            .rename(columns={"speaker": "personaje"})


def character_table(units: list[Unit], saga: str, by_work: bool = False) -> pd.DataFrame:
    """Tabla personaje × medio (× obra) con conteos y proporciones dentro de cada medio (u obra)."""
    book, film = paired_units(units, saga)
    keys = ["medium", "work"] if by_work else ["medium"]
    med = lambda u: "book" if u.medium == "book" else "film"  # noqa: E731
    nu = (pd.DataFrame([{"medium": med(u), "work": u.work} for u in book + film])
          .groupby(keys).size().rename("n_unidades").reset_index())
    names = list(characters.mentions(saga))
    t = nu.merge(pd.DataFrame({"personaje": names}), how="cross")
    uc = unit_counts(book + film, saga).assign(presencia=1)
    agg = uc.groupby(keys + ["personaje"])[list(MEASURES)].sum().reset_index()
    t = t.merge(agg, on=keys + ["personaje"], how="left").fillna({m: 0 for m in MEASURES})
    # intervenciones del libro = citas con atribución explícita
    bi = book_interventions(book, saga).assign(medium="book")
    bi = bi.groupby(keys + ["personaje"]).intervenciones.sum().rename("_bi").reset_index()
    t = t.merge(bi, on=keys + ["personaje"], how="left")
    t["intervenciones"] = np.where(t.medium == "book", t._bi.fillna(0), t.intervenciones)
    t = t.drop(columns="_bi")
    for m in ("menciones", "intervenciones"):
        t[f"prop_{m}"] = t[m] / t.groupby(keys)[m].transform("sum").replace(0, np.nan)
    t["prop_presencia"] = t.presencia / t.n_unidades
    if by_work:
        bff = saga_config(saga).get("book_for_film", {})
        t["pair"] = t.work.map(lambda w: bff.get(w, w))
    t["medio"] = t.medium.map(MEDIO)
    t["nota"] = t.personaje.map(characters.notes(saga)).fillna("")
    for m in MEASURES:
        t[m] = t[m].astype(int)
    return t


def comparison(units: list[Unit], saga: str, measure: str = "menciones", alpha0: float = 100.0) -> pd.DataFrame:
    """Una fila por personaje: proporción en libro vs película + z de log-odds (película sobre libro)."""
    t = character_table(units, saga)
    w = t.pivot_table(index="personaje", columns="medium", values=[measure, f"prop_{measure}"], aggfunc="sum")
    d = pd.DataFrame({"personaje": w.index,
                      "count_book": w[(measure, "book")].values, "count_film": w[(measure, "film")].values,
                      "prop_book": w[(f"prop_{measure}", "book")].values,
                      "prop_film": w[(f"prop_{measure}", "film")].values})
    cb = Counter(dict(zip(d.personaje, d.count_book)))
    cf = Counter(dict(zip(d.personaje, d.count_film)))
    if measure == "presencia":   # presencia: proporción de unidades, no reparto de un total
        d["z"] = np.nan
    elif sum(cb.values()) and sum(cf.values()):
        z = dirichlet_log_odds(cf, cb, alpha0=alpha0, min_count=1).set_index("word")["z"]
        d["z"] = d.personaje.map(z)
    else:
        d["z"] = np.nan
    d["ratio"] = d.prop_film / d.prop_book.replace(0, np.nan)
    d["nota"] = d.personaje.map(characters.notes(saga)).fillna("")
    return d.sort_values("count_book", ascending=False).reset_index(drop=True)


def presence_curve(units: list[Unit], saga: str, bins: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menciones por mil palabras a lo largo de la narración (posición normalizada 0–1 por medio).

    Devuelve (curvas, límites de obra). Película: menciones + intervenciones como hablante, para que
    un personaje que habla mucho y es poco nombrado no desaparezca."""
    book, film = paired_units(units, saga)
    out, edges = [], []
    for medium, us in (("book", book), ("film", film)):
        if not us:
            continue
        n = len(us)
        w = np.array([max(len((film_mention_text(u) if medium == "film" else u.text).split()), 1) for u in us])
        cum = np.cumsum(w) / w.sum()               # posición por palabras, no por número de unidades
        start = np.concatenate([[0.0], cum[:-1]])
        b_of = np.minimum((start * bins).astype(int), bins - 1)
        words_bin = np.bincount(b_of, weights=w, minlength=bins)
        uc = unit_counts(us, saga)
        if uc.empty:
            continue
        pos_to_bin = dict(zip(range(n), b_of))
        uc["bin"] = uc.pos.map(pos_to_bin)
        uc["valor"] = uc.menciones + (uc.intervenciones if medium == "film" else 0)
        g = uc.groupby(["personaje", "bin"]).valor.sum().unstack(fill_value=0).reindex(columns=range(bins),
                                                                                       fill_value=0)
        for c, row in g.iterrows():
            out.append(pd.DataFrame({"personaje": c, "medio": MEDIO[medium], "posicion": (np.arange(bins) + .5) / bins,
                                     "por_mil": 1000 * row.to_numpy() / np.maximum(words_bin, 1)}))
        for k in range(1, n):
            if us[k].work != us[k - 1].work:
                edges.append({"medio": MEDIO[medium], "posicion": float(start[k]), "obra": us[k].work})
    curves = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["personaje", "medio", "posicion", "por_mil"])
    return curves, pd.DataFrame(edges, columns=["medio", "posicion", "obra"])
