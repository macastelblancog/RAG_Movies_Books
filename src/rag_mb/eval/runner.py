"""Benchmark: closed-book (LLM general) vs RAG (mismo LLM + contexto citado).

    rmb eval all --client local                          # Phi-4-mini: sin RAG vs RAG global (pipeline real)
    rmb eval all --client local --lang es                # mismas preguntas en español (question_es)
    rmb eval all --client manual                         # prompts para pegar en ChatGPT / Gemini web
    rmb eval all --client manual --import reports/prompts/all_prompts.csv
    rmb eval-score reports/runs/<run>/review.csv         # puntajes manuales (interpretativas)

Condiciones: closed_book (el LLM responde de memoria) · rag_global (índice único + enrutado + reranker +
ventana de contexto + respuesta por medio: lo mismo que `rmb ask`) · rag_* (índice por saga).
Categoría `interpretive`: sin palabras clave; se puntúa a mano con su `rubric` (review.csv).

Salidas en reports/runs/<saga>_<cliente>_<fecha>/: results.jsonl, summary.csv, report.md
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..analysis.stats import bootstrap_ci, mcnemar_exact
from ..config import paths
from ..ingest.build import units_for
from ..rag.index import RagIndex
from ..rag.llm import SYSTEM_CLOSED, SYSTEM_RAG, get_client, rag_prompt
from .gold import GOLD_FNS
from .metrics import has_citation, retrieval_metrics, score_answer

log = logging.getLogger(__name__)

CONDITIONS = {
    # nombre: (usa_rag, modo_recuperación, balanceado libro/guion)
    "closed_book": (False, None, False),
    "rag_bm25": (True, "bm25", False),
    "rag_dense": (True, "dense", False),
    "rag_hybrid": (True, "hybrid", False),
    "rag_hybrid_balanced": (True, "hybrid", True),
    "rag_bm25_routed": (True, "bm25", "routed"),
    "rag_hybrid_routed": (True, "hybrid", "routed"),
    "rag_global": (True, "global", "global"),          # pipeline real (rag.qa)
    "rag_graph": (True, "global", "global"),           # el mismo pipeline orquestado con LangGraph (rag.graph)
}
MANUAL_CATEGORIES = {"interpretive"}                     # se puntúan con rúbrica, no con `must`


def load_questions(saga: str) -> list[dict]:
    with open(paths().questions / f"{saga}.yaml", encoding="utf-8") as fh:
        qs = yaml.safe_load(fh)["questions"]
    units = None
    for q in qs:
        if q.get("gold_fn"):
            units = units or units_for(saga)
            g = GOLD_FNS[q["gold_fn"]](units, saga, **q.get("gold_args", {}))
            q["answer"], q["must"] = g["answer"], g["must"]
    return qs


def check_questions(saga: str) -> pd.DataFrame:
    """Verifica que cada evidencia tenga al menos un chunk relevante en el corpus."""
    from ..schema import load_chunks
    from .metrics import chunk_relevant
    chunks = load_chunks(paths().chunks_file(saga))
    rows = []
    for q in load_questions(saga):
        for e in q.get("evidence", []):
            n = sum(chunk_relevant(c, e) for c in chunks)
            rows.append({"id": q["id"], "evidence": json.dumps(e, ensure_ascii=False), "n_relevant_chunks": n})
    return pd.DataFrame(rows)


GLOBAL_MODES = ("hybrid", "hybrid_routed", "hybrid_routed_oracle", "hybrid_routed_fusion", "hybrid_routed_norerank")
REF_MODE = "hybrid_routed"     # referencia de las comparaciones pareadas


def retrieval_eval(saga: str, k: int = 8, modes=("bm25", "dense", "hybrid", "hybrid_balanced",
                                                 "bm25_routed", "dense_routed", "hybrid_routed"),
                   dense: str = "lsa", use_global: bool = False, hyde: bool = False) -> pd.DataFrame:
    """Evalúa SOLO recuperación (sin LLM): Recall@k, MRR, nDCG por modo y categoría.

    use_global=True: las preguntas de `saga` contra el ÍNDICE ÚNICO (todas las sagas). Modos:
      hybrid                  sin enrutar (busca en todo)
      hybrid_routed           saga/medio/obra deducidos de la pregunta + reranker (config: rag.rerank_fusion)
      hybrid_routed_fusion    igual, pero FUSIONANDO el orden del reranker con el híbrido (RRF)
      hybrid_routed_norerank  igual, sin reranker (aísla el aporte del cross-encoder)
      hybrid_routed_hyde      igual + HyDE (con --hyde; requiere el LLM local)
      hybrid_routed_oracle    igual que hybrid_routed pero con la saga CONOCIDA (cota de lo que se
                              pierde por detectarla; las preguntas del banco se escribieron por saga)
    y se reporta si el enrutador acertó la saga y qué fracción de pasajes es de la saga correcta."""
    idx = RagIndex.load_or_build("all" if use_global else saga, dense)
    if use_global:
        modes = GLOBAL_MODES + (("hybrid_routed_hyde",) if hyde else ())
    rows = []
    for q in load_questions(saga):
        if not q.get("evidence"):
            continue
        for m in modes:
            rerank = not m.endswith("_norerank")
            base = m.removesuffix("_norerank").removesuffix("_fusion").removesuffix("_oracle").removesuffix("_hyde")
            idx.last_route = None
            if base.endswith("_routed"):
                keep = (idx.reranker, getattr(idx, "rerank_fusion", "rrf"))
                if not rerank:
                    idx.reranker = None
                if m.endswith("_fusion"):
                    idx.rerank_fusion = "rrf"
                try:
                    hits = idx.search_routed(q["question"], k, base.removesuffix("_routed"),
                                             saga=saga if (use_global and m.endswith("_oracle")) else None,
                                             hyde=m.endswith("_hyde"))
                finally:
                    idx.reranker, idx.rerank_fusion = keep
            elif base == "hybrid_balanced":
                hits = idx.search_balanced(q["question"], k)
            else:
                hits = idx.search(q["question"], k, base, rerank=rerank)
            r = retrieval_metrics([h.chunk for h in hits], q["evidence"], k)
            row = {"id": q["id"], "category": q["category"], "mode": m, **r}
            if use_global:
                rt = idx.last_route
                row["saga_detectada"] = rt.saga if rt else None
                row["saga_ok"] = float((rt.saga if rt else None) == saga) if base.endswith("_routed") else np.nan
                row["frac_misma_saga"] = float(np.mean([h.chunk.saga == saga for h in hits])) if hits else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def retrieval_stats(df: pd.DataFrame, ref: str = REF_MODE, n_boot: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) por modo: media e IC 95 % bootstrap de hit@k y MRR; (2) cada modo vs `ref`, pareado por
    pregunta: McNemar exacto sobre hit@k y IC bootstrap de la diferencia de MRR.
    Con ~60 preguntas, una diferencia de 0,05 son 3 preguntas: sin esto no se puede afirmar mejora."""
    from ..analysis.stats import bootstrap_ci, mcnemar_exact, paired_diff_ci
    keycols = [c for c in ("saga", "id") if c in df.columns]
    rows, comp = [], []
    for m, g in df.groupby("mode", sort=False):
        h, hl, hh = bootstrap_ci(g["hit@k"].to_numpy(), n_boot)
        r, rl, rh = bootstrap_ci(g["mrr"].to_numpy(), n_boot)
        rows.append({"mode": m, "n": len(g), "hit@k": h, "hit_ic95": f"[{hl:.2f}, {hh:.2f}]",
                     "mrr": r, "mrr_ic95": f"[{rl:.2f}, {rh:.2f}]"})
    if ref in set(df["mode"]):
        piv = df.pivot_table(index=keycols, columns="mode", values=["hit@k", "mrr"])
        for m in df["mode"].unique():
            if m == ref:
                continue
            mc = mcnemar_exact(piv[("hit@k", ref)].to_numpy() > 0, piv[("hit@k", m)].to_numpy() > 0)
            d, dl, dh = paired_diff_ci(piv[("mrr", ref)].to_numpy(), piv[("mrr", m)].to_numpy(), n_boot)
            comp.append({"modo": m, "acierta_solo_ref": mc["a_only"], "acierta_solo_modo": mc["b_only"],
                         "p_mcnemar_hit": mc["p_value"], "Δmrr_ref−modo": d,
                         "Δmrr_ic95": f"[{dl:.2f}, {dh:.2f}]"})
    return pd.DataFrame(rows), pd.DataFrame(comp)


def _retrieve(idx: RagIndex, q: dict, mode: str, balanced, k: int):
    if balanced == "routed":
        return idx.search_routed(q["question"], k, mode)
    return idx.search_balanced(q["question"], k, mode) if balanced else idx.search(q["question"], k, mode)


def run(saga: str, client: str = "extractive", conditions=("closed_book", "rag_global"), k: int = 8,
        dense: str = "lsa", import_answers: str | None = None, limit: int | None = None,
        lang: str = "en", reuse: str | None = None) -> Path:
    """reuse: carpeta de una corrida anterior; se copian sus resultados de las condiciones que NO se
    corren ahora (p. ej. closed_book, que no cambia con mejoras del RAG) para que el reporte sea completo."""
    from ..rag.index import indexed_sagas
    from ..rag.llm import split_evidence
    from ..rag.qa import prompts as qa_prompts
    from ..rag.qa import retrieve as qa_retrieve
    from ..rag.qa import tool_answer
    sagas = indexed_sagas() if saga == "all" else [saga]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = paths().reports / "runs" / f"{saga}_{client.replace(':', '-')}_{lang}_{stamp}"

    # --- construir prompts
    jobs = []
    for sg in sagas:
        qs = load_questions(sg)[:limit]
        idx = None
        for q in qs:
            question = q.get("question_es") if lang == "es" and q.get("question_es") else q["question"]
            for cond in conditions:
                use_rag, mode, bal = CONDITIONS[cond]
                hits, meta, preset = [], {}, None
                if cond == "rag_graph":            # recuperación + generación ocurren dentro del grafo
                    if client == "manual" or import_answers:
                        raise ValueError("rag_graph necesita un cliente LLM (no funciona con --client manual)")
                    jobs.append({"saga": sg, "q": q, "question": question, "cond": cond, "hits": [],
                                 "system": "", "user": "", "meta": {}, "preset": None, "graph": True})
                    continue
                if cond == "rag_global":
                    a = tool_answer(question)
                    if a is not None:              # cuantitativa → cálculo sobre el corpus, sin LLM
                        system, user, preset = "", "", a.text
                        meta = {"saga_detectada": a.saga, "medio": None, "balanceado": False, "tool": a.tool}
                    else:
                        a = qa_retrieve(question)
                        system, user = qa_prompts(a)
                        hits = a.passages
                        meta = {"saga_detectada": a.saga, "medio": a.medium, "balanceado": a.balanced}
                elif use_rag:
                    idx = idx or RagIndex.load_or_build(sg, dense)
                    hits = _retrieve(idx, {**q, "question": question}, mode, bal, k)
                    system, user = SYSTEM_RAG, rag_prompt(question, hits)
                else:
                    system, user = SYSTEM_CLOSED, f"Question: {question}\nAnswer:"
                jobs.append({"saga": sg, "q": q, "question": question, "cond": cond, "hits": hits,
                             "system": system, "user": user, "meta": meta, "preset": preset})

    # --- modo manual: exportar y salir / importar respuestas
    if client == "manual" and not import_answers:
        pdir = paths().reports / "prompts"
        pdir.mkdir(parents=True, exist_ok=True)
        out = pdir / f"{saga}_{lang}_prompts.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["job_id", "saga", "condition", "question_id", "prompt", "answer"])
            for i, j in enumerate(jobs):
                w.writerow([i, j["saga"], j["cond"], j["q"]["id"],
                            "(herramienta: no requiere LLM)" if j["preset"] else j["system"] + "\n\n" + j["user"],
                            j["preset"] or ""])
        log.info("Prompts exportados a %s. Llena la columna 'answer' y usa --import.", out)
        return out
    answers = None
    if import_answers:
        answers = pd.read_csv(import_answers).set_index("job_id")["answer"].fillna("").to_dict()

    llm = None if answers is not None else get_client(client)
    results = []
    for i, j in enumerate(jobs):
        q = j["q"]
        if j.get("graph"):
            from ..rag.graph import run as graph_run
            try:
                a = graph_run(j["question"], llm)
                j["hits"] = a.passages
                j["meta"] = {"saga_detectada": a.saga, "medio": a.medium, "balanceado": a.balanced,
                             "tool": a.tool, "rama": "divergencia" if a.raw and a.raw.startswith("[book]") else None}
                if a.tool:
                    j["preset"] = a.text
                ans = a.text if a.tool else (a.raw or "")
            except Exception as e:
                log.warning("Falla en %s/%s: %s", q["id"], j["cond"], e)
                ans = ""
        elif j["preset"] is not None:
            ans = j["preset"]
        elif answers is not None:
            ans = answers.get(i, "")
        else:
            try:
                ans = llm.complete(j["system"], j["user"])
            except Exception as e:          # una falla puntual no tumba toda la corrida
                log.warning("Falla en %s/%s: %s", q["id"], j["cond"], e)
                ans = ""
        manual = q.get("category") in MANUAL_CATEGORIES
        if j.get("graph") and not j["preset"] and ans:
            evidence, final = a.evidence, a.text      # el grafo ya separó (y unió) evidencia y respuesta
        else:
            evidence, final = split_evidence(ans) if j["cond"] != "closed_book" and not j["preset"] else ("", ans)
        rec = {"saga": j["saga"], "id": q["id"], "category": q["category"], "condition": j["cond"],
               "question": j["question"], "reference": q.get("answer"), "rubric": q.get("rubric"),
               "answer": final, "evidence": evidence, "raw": ans,
               "correct": None if manual else score_answer(final, q.get("must"), q.get("expect")),
               "cited": has_citation(final) or has_citation(evidence) or bool(j["preset"]),
               "citations": [h.citation for h in j["hits"]], **j["meta"]}
        if j["hits"] and q.get("evidence"):
            rec.update(retrieval_metrics([h.chunk for h in j["hits"]], q["evidence"], max(k, len(j["hits"]))))
        results.append(rec)
        log.info("[%d/%d] %s %s → %s", i + 1, len(jobs), q["id"], j["cond"],
                 "manual" if rec["correct"] is None else ("OK" if rec["correct"] else "x"))

    if reuse:
        prev = pd.read_json(Path(reuse) / "results.jsonl", lines=True)
        ids = {(r["saga"], r["id"]) for r in results}
        prev = prev[~prev.condition.isin(conditions) & prev.apply(lambda r: (r.saga, r.id) in ids, axis=1)]
        results += prev.to_dict("records")
        log.info("Reutilizadas %d respuestas de %s (%s)", len(prev), reuse, sorted(prev.condition.unique()))
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    df = pd.DataFrame(results)
    rev = df[df.correct.isna()][["saga", "id", "condition", "question", "reference", "rubric", "answer"]]
    if len(rev):
        rev.assign(score="").to_csv(outdir / "review.csv", index=False, encoding="utf-8")
        log.info("%d respuestas interpretativas para puntuar a mano: %s", len(rev), outdir / "review.csv")
    summarize(df, outdir, model=getattr(llm, "name", client))
    return outdir


def rescore(run_dir: str) -> Path:
    """Vuelve a puntuar una corrida con la métrica ACTUAL, sin llamar al LLM (las respuestas están guardadas).
    Las preguntas de rúbrica manual conservan su puntaje. Sirve para comparar corridas con la misma métrica."""
    outdir = Path(run_dir)
    df = pd.read_json(outdir / "results.jsonl", lines=True)
    bank = {}
    for sg in df.saga.unique():
        bank.update({(sg, q["id"]): q for q in load_questions(sg)})
    new = []
    for r in df.itertuples():
        q = bank.get((r.saga, r.id))
        if q is None or q.get("category") in MANUAL_CATEGORIES:
            new.append(r.correct)
        else:
            new.append(score_answer(r.answer, q.get("must"), q.get("expect")))
    changed = int(sum(bool(a) != bool(b) for a, b in zip(df.correct, new) if a is not None and b is not None))
    df["correct"] = new
    df.to_json(outdir / "results.jsonl", orient="records", lines=True, force_ascii=False)
    summarize(df, outdir, model=outdir.name)
    log.info("Re-puntuadas %d filas; %d cambiaron de veredicto", len(df), changed)
    return outdir


def apply_review(review_csv: str) -> Path:
    """Incorpora los puntajes manuales (columna score: 1 / 0 / 0.5) y regenera el resumen de la corrida."""
    rp = Path(review_csv)
    outdir = rp.parent
    df = pd.read_json(outdir / "results.jsonl", lines=True)
    sc = pd.read_csv(rp).dropna(subset=["score"])
    key = {(r.saga, r.id, r.condition): float(r.score) for r in sc.itertuples()}
    df["correct"] = [key.get((r.saga, r.id, r.condition), r.correct) for r in df.itertuples()]
    df.to_json(outdir / "results.jsonl", orient="records", lines=True, force_ascii=False)
    summarize(df, outdir, model=outdir.name)
    return outdir


def to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Tabla markdown sin dependencias (evita requerir `tabulate`)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def summarize(df: pd.DataFrame, outdir: Path, model: str = "") -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["correct"] = pd.to_numeric(df.correct, errors="coerce")       # None (sin puntuar) → NaN
    for cond, g in df.groupby("condition"):
        gs = g.dropna(subset=["correct"])
        m, lo, hi = bootstrap_ci(gs.correct.values) if len(gs) else (np.nan, np.nan, np.nan)
        rows.append({"condition": cond, "category": "ALL", "n": len(gs), "pendientes": int(g.correct.isna().sum()),
                     "accuracy": m, "ci_low": lo, "ci_high": hi, "cited": g.cited.mean()})
        for cat, gc in g.groupby("category"):
            gcs = gc.dropna(subset=["correct"])
            rows.append({"condition": cond, "category": cat, "n": len(gcs), "pendientes": int(gc.correct.isna().sum()),
                         "accuracy": gcs.correct.mean() if len(gcs) else np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "cited": gc.cited.mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(outdir / "summary.csv", index=False)

    conds = list(df.condition.unique())
    tests = []
    keys = [c for c in ("saga", "id") if c in df.columns]
    piv = df.pivot_table(index=keys, columns="condition", values="correct", aggfunc="first")
    for a_i in range(len(conds)):
        for b_i in range(a_i + 1, len(conds)):
            a, b = conds[a_i], conds[b_i]
            pr = piv[[a, b]].dropna()
            t = mcnemar_exact(pr[a].values >= 0.5, pr[b].values >= 0.5)
            tests.append({"A": a, "B": b, "n": len(pr), **t})
    tests = pd.DataFrame(tests)

    md = [f"# Benchmark — {model}", "", "## Exactitud por condición (IC 95% bootstrap)", "",
          to_md(summ[summ.category == "ALL"].round(3)), "",
          "## Por categoría", "",
          to_md(summ[summ.category != "ALL"].pivot(index="category", columns="condition", values="accuracy").round(2), index=True), "",
          "## McNemar exacto (pareado)", "", to_md(tests.round(4)) if len(tests) else "-"]
    (outdir / "report.md").write_text("\n".join(md), encoding="utf-8")
    return summ
