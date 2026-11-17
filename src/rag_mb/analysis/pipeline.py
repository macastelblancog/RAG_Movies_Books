"""Análisis completo de una saga: libro vs guion por obra. Produce métricas JSON, tablas y figuras.

Uso:  rmb analyze lotr      (requiere `rmb build lotr`)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import characters  # noqa: E402
from ..config import paths, saga_config  # noqa: E402
from ..ingest.build import units_for  # noqa: E402
from ..schema import Unit  # noqa: E402
from ..text import tokenize  # noqa: E402
from . import graphs, lexical  # noqa: E402
from .dtw import dtw, path_to_mapping  # noqa: E402
from .semantic_shift import ppmi_vectors, semantic_shift  # noqa: E402
from .transport import sinkhorn, wasserstein_permutation_test  # noqa: E402
from .vectorize import cosine_cost, embed  # noqa: E402

log = logging.getLogger(__name__)

# Personajes: fuente única en configs/characters.yaml (ver rag_mb.characters).
ALIASES = {s: characters.aliases(s) for s in characters.load_characters()}        # variante → canónico
BOOK_MENTIONS = {s: characters.mentions(s) for s in characters.load_characters()}  # canónico → nombres
FILM_MEDIA = ("screenplay", "dialogue", "subtitles")   # lado "película" (dialogue = transcripción con hablante)


@dataclass
class Segment:
    chapter_idx: int
    unit: Unit
    text: str
    n_words: int


def segment_book(units: list[Unit], seg_words: int = 800) -> list[Segment]:
    segs = []
    for ci, u in enumerate(units):
        words = u.text.split()
        for s in range(0, len(words), seg_words):
            w = words[s:s + seg_words]
            if len(w) < seg_words // 4 and segs and segs[-1].chapter_idx == ci:
                segs[-1].text += " " + " ".join(w)
                segs[-1].n_words += len(w)
                continue
            segs.append(Segment(ci, u, " ".join(w), len(w)))
    return segs


def adaptation_map(book: list[Unit], scenes: list[Unit], *, method: str = "lsa", eps: float = 0.02,
                   tau: float = 0.1, seg_words: int = 800, n_perm: int = 0) -> dict:
    segs = segment_book(book, seg_words)
    emb = embed([s.text for s in segs] + [u.text for u in scenes], method=method)
    X, Y = emb.vectors[:len(segs)], emb.vectors[len(segs):]
    C = cosine_cost(X, Y)
    a = np.array([s.n_words for s in segs], float)
    b = np.array([max(u.n_words, 20) for u in scenes], float)
    bal = sinkhorn(a, b, C, eps=eps)
    unb = sinkhorn(a, b, C, eps=eps, tau=tau)
    dcost, path = dtw(C)
    dtw_pos = path_to_mapping(path, len(segs))

    seg_df = pd.DataFrame({
        "segment": range(len(segs)), "chapter_idx": [s.chapter_idx for s in segs],
        "chapter": [s.unit.locator for s in segs], "words": a,
        "retained": unb.retained, "ot_pos": bal.barycentric_pos, "dtw_pos": dtw_pos,
    })
    seg_df["retained_rel"] = seg_df.retained / np.average(seg_df.retained, weights=seg_df.words)
    ch = (seg_df.groupby(["chapter_idx", "chapter"])
          .apply(lambda g: pd.Series({"words": g.words.sum(),
                                      "retention": np.average(g.retained_rel, weights=g.words),
                                      "film_pos_ot": np.average(g.ot_pos, weights=g.words),
                                      "film_pos_dtw": np.nanmean(g.dtw_pos)}), include_groups=False)
          .reset_index())
    out = {
        "embedding": emb.method, "explained_variance": emb.explained_variance,
        "n_segments": len(segs), "n_scenes": len(scenes),
        "ot_cost_balanced": bal.cost, "ot_cost_unbalanced": unb.cost,
        "order_spearman_ot": bal.order_spearman, "dtw_cost_norm": dcost / len(path),
        "chapters": ch, "segments": seg_df, "plan": bal.plan, "plan_unb": unb.plan,
    }
    if n_perm:
        out["perm_test"] = wasserstein_permutation_test(X, Y, a, b, eps=eps, n_perm=n_perm)
    return out


def _plot_plan(res: dict, title: str, path, xbounds: list[tuple[float, str]] | None = None):
    P = res["plan"]
    seg = res["segments"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    ax.imshow(np.log10(P + 1e-12), aspect="auto", cmap="magma", vmin=np.log10(P.max()) - 3)
    bounds = np.where(np.diff(seg.chapter_idx.values) != 0)[0] + 0.5
    for y in bounds:
        ax.axhline(y, color="white", lw=0.3, alpha=0.5)
    for x, lab in (xbounds or []):
        ax.axvline(x, color="cyan", lw=1, ls="--")
        ax.text(x + 2, 8, lab, color="cyan", fontsize=8, va="top")
    ax.set_xlabel("Escena del guion (orden)")
    ax.set_ylabel("Segmento del libro (orden)")
    ax.set_title(f"Plan de transporte óptimo (log10) — {title}")
    ch = res["chapters"]
    ax2 = axes[1]
    colors = ["#c0392b" if r < 0.6 else "#2c7fb8" if r > 1.4 else "#7f8c8d" for r in ch.retention]
    ax2.barh(range(len(ch)), ch.retention, color=colors)
    ax2.axvline(1, color="k", lw=0.8, ls="--")
    ax2.set_yticks(range(len(ch)))
    ax2.set_yticklabels([c.split("—")[-1].strip()[:32] for c in ch.chapter], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Retención relativa ρ (OT no balanceado; 1 = promedio)")
    ax2.set_title("¿Qué sobrevive en la película?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_order(res: dict, title: str, path):
    ch = res["chapters"]
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(0, 1, len(ch))
    ax.plot(x, ch.film_pos_ot, "o-", label=f"OT baricéntrico (ρ_s={res['order_spearman_ot']:.2f})")
    ax.plot(x, ch.film_pos_dtw, "s--", alpha=0.7, label="DTW (monótono)")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("Posición del capítulo en el libro")
    ax.set_ylabel("Posición esperada en la película")
    ax.set_title(f"Fidelidad al orden narrativo — {title}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_zipf(cbook, cfilm, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c, lab in [(cbook, "Libro"), (cfilm, "Guion")]:
        f = np.array(sorted(c.values(), reverse=True))
        ax.loglog(np.arange(1, len(f) + 1), f / f.sum(), label=lab)
    ax.set_xlabel("Rango r")
    ax.set_ylabel("Frecuencia relativa")
    ax.set_title("Ley de Zipf")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_saga(saga: str, *, method: str = "lsa", n_perm: int = 0, top_chars: int = 12) -> dict:
    P = paths()
    figdir = P.figures / saga
    figdir.mkdir(parents=True, exist_ok=True)
    units = units_for(saga)
    book = [u for u in units if u.medium == "book"]
    film = [u for u in units if u.medium in FILM_MEDIA]
    book_for_film = saga_config(saga).get("book_for_film", {})
    # Solo los libros que tienen película (Narnia: ómnibus de 7 libros, guion solo de LWW)
    adapted = {book_for_film.get(u.work, u.work) for u in film}
    book = [u for u in book if u.work in adapted] or book
    book_works = {u.work for u in book}
    film = [u for u in film if book_for_film.get(u.work, u.work) in book_works]   # solo películas con libro
    works = list(dict.fromkeys(u.work for u in film))
    summary: dict = {"saga": saga, "works": {}}

    # --- 1. Estadística léxica (saga completa)
    tb = [tokenize(u.text) for u in book]
    # Guion: SOLO diálogo (la acción es prosa técnica: "ANGLE ON", "CLOSE ON"); subtítulos: todo
    # (escenas sin diálogo aportan 0 tokens; solo los subtítulos, que no tienen hablante, usan el texto)
    tf = [tokenize(u.text if u.medium == "subtitles" else " ".join(ln.text for ln in u.lines)) for u in film]
    cb, cf = lexical.counts(tb), lexical.counts(tf)
    zb, zf = lexical.zipf_fit(cb), lexical.zipf_fit(cf)
    hb, hf = lexical.heaps_fit([t for x in tb for t in x]), lexical.heaps_fit([t for x in tf for t in x])
    summary["lexical"] = {
        "tokens_book": sum(cb.values()), "tokens_film_dialogue": sum(cf.values()),
        "vocab_book": len(cb), "vocab_film": len(cf),
        "zipf_alpha_book": zb.alpha, "zipf_alpha_film": zf.alpha,
        "zipf_alpha_mle_book": zb.alpha_mle, "zipf_alpha_mle_film": zf.alpha_mle,
        "heaps_beta_book": hb.alpha, "heaps_beta_film": hf.alpha,
        "js_divergence_bits": lexical.js_divergence(cb, cf),
    }
    lo = lexical.dirichlet_log_odds(cf, cb)
    lo.head(40).to_csv(figdir / "logodds_film_over_book.csv", index=False)
    lo.tail(40).iloc[::-1].to_csv(figdir / "logodds_book_over_film.csv", index=False)
    _plot_zipf(cb, cf, figdir / "zipf.png")

    # --- 2. Mapa de adaptación por obra (OT + DTW)
    for w in works:
        bw = [u for u in book if u.work == book_for_film.get(w, w)]   # config: película → libro
        sw = [u for u in film if u.work == w]
        if not bw or len(sw) < 3:
            log.info("%s: sin libro asociado en el corpus, se omite el mapa de adaptación", w)
            continue
        res = adaptation_map(bw, sw, method=method, n_perm=n_perm)
        res["chapters"].to_csv(figdir / f"adaptation_{w}.csv", index=False)
        _plot_plan(res, w, figdir / f"ot_plan_{w}.png")
        _plot_order(res, w, figdir / f"order_{w}.png")
        ch = res["chapters"].sort_values("retention")
        summary["works"][w] = {k: v for k, v in res.items()
                               if k not in ("chapters", "segments", "plan", "plan_unb")}
        summary["works"][w]["least_retained"] = ch.head(5)[["chapter", "retention"]].values.tolist()
        summary["works"][w]["most_retained"] = ch.tail(5)[["chapter", "retention"]].values.tolist()

    # --- 2b. Transporte a nivel de saga: todo el libro vs todas las películas en orden.
    #     Detecta material movido entre películas (p.ej. Shelob: libro 4 → película 3).
    if len(works) > 1:
        film_sorted = sorted(film, key=lambda u: (works.index(u.work), u.order))
        res = adaptation_map(book, film_sorted, method=method)
        segs = res["segments"]
        film_of_scene = np.array([works.index(u.work) for u in film_sorted])
        mass = np.stack([res["plan"][:, film_of_scene == k].sum(1) for k in range(len(works))], 1)
        segs[[f"mass_{w}" for w in works]] = mass / mass.sum(1, keepdims=True)
        by_ch = segs.groupby(["chapter_idx", "chapter"])[[f"mass_{w}" for w in works]].mean().reset_index()
        by_ch["book_work"] = [book[i].work for i in by_ch.chapter_idx]
        by_ch["film_argmax"] = [works[k] for k in by_ch[[f"mass_{w}" for w in works]].values.argmax(1)]
        by_ch["retention"] = res["chapters"]["retention"].values
        # Un capítulo poco retenido no está "movido": está omitido y su argmax es ruido.
        film_book = by_ch.film_argmax.map(lambda f: book_for_film.get(f, f))
        by_ch["status"] = np.where(by_ch.retention < 0.6, "omitido",
                                   np.where(by_ch.book_work != film_book, "movido", "en su lugar"))
        by_ch.to_csv(figdir / "saga_chapter_to_film.csv", index=False)
        cols = ["chapter", "book_work", "film_argmax", "retention"]
        summary["saga_ot"] = {
            "order_spearman_ot": res["order_spearman_ot"],
            "moved_chapters": by_ch[by_ch.status == "movido"][cols].values.tolist(),
            "omitted_chapters": by_ch[by_ch.status == "omitido"][cols].values.tolist(),
        }
        starts = [(float(np.argmax(film_of_scene == k)) - 0.5, w) for k, w in enumerate(works)]
        _plot_plan(res, "saga completa", figdir / "ot_plan_saga.png", xbounds=starts)

    # --- 3. Desplazamiento semántico (Procrustes)
    try:
        Wb, vb = ppmi_vectors(tb, dim=100, min_count=10)
        Wf, vf = ppmi_vectors(tf, dim=100, min_count=5)
        shift = semantic_shift(Wb, vb, Wf, vf, cb, cf, min_count=15)
        shift.head(40).to_csv(figdir / "semantic_shift.csv", index=False)
        summary["semantic_shift_top"] = shift.head(15)[["word", "shift"]].values.tolist()
        summary["semantic_shift_median"] = float(shift["shift"].median())
    except Exception as e:  # corpus demasiado pequeño
        log.warning("semantic shift omitido: %s", e)

    # --- 4. Redes de personajes
    mentions = BOOK_MENTIONS.get(saga) or {}
    scene_units = [u for u in film if u.lines]
    if mentions and scene_units:
        Gf = graphs.graph_from_scenes(scene_units, ALIASES.get(saga))
        Gb = graphs.graph_from_mentions(book, mentions)
        nodes = [n for n in graphs.top_nodes(Gf, 40) if n in Gb and Gb.degree(n) > 0][:top_chars]
        comp = graphs.compare_graphs(Gb, Gf, nodes)
        comp["table"].to_csv(figdir / "character_centrality.csv", index=False)
        summary["graphs"] = {k: v for k, v in comp.items() if k != "table"}
        summary["graphs"]["nodes"] = nodes
        summary["graphs"]["fiedler_partition_film"] = graphs.fiedler_partition(Gf, nodes)
        summary["graphs"]["fiedler_partition_book"] = graphs.fiedler_partition(Gb, nodes)

    with open(figdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    return summary
