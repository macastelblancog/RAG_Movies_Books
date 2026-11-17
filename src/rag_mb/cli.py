"""CLI: rmb <comando>.

  rmb build lotr                    ingesta → data/processed/lotr_{units,chunks}.jsonl
  rmb inspect lotr                  resumen de unidades por fuente (verificar segmentación)
  rmb analyze lotr [--neural]       análisis matemático → reports/figures/lotr/
  rmb ask lotr "question" [--mode hybrid|bm25|dense] [--balanced] [--client openai]
  rmb check-questions lotr          cada evidencia del YAML tiene chunks relevantes
  rmb retrieval lotr                métricas de recuperación (sin LLM, sin red)
  rmb eval lotr --client openai|gemini|manual|extractive [--conditions ...] [--import CSV]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rmb", description="RAG_Movies_Books")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build"); p.add_argument("saga")
    p = sub.add_parser("inspect"); p.add_argument("saga")
    p = sub.add_parser("analyze"); p.add_argument("saga"); p.add_argument("--neural", action="store_true")
    p.add_argument("--perm", type=int, default=0, help="nº de permutaciones para el test de Wasserstein")
    p = sub.add_parser("rechunk", help="re-trocea desde las unidades ya procesadas (sin re-leer PDFs)")
    p.add_argument("sagas", nargs="*", help="por defecto: todas las que tengan corpus")
    p = sub.add_parser("index", help="construye el índice ÚNICO (todas las sagas)")
    p.add_argument("--dense", default=None, help="neural | lsa (por defecto: configs/corpora.yaml → rag.dense)")
    p.add_argument("--rebuild", action="store_true")
    p = sub.add_parser("ask", help="pregunta sobre cualquier saga (se detecta sola)"); p.add_argument("question")
    p.add_argument("--saga", default=None, help="forzar una saga (opcional)")
    p.add_argument("--k", type=int, default=None); p.add_argument("--client", default=None,
                                                                   help="local | gemini | openai | extractive")
    p.add_argument("--dense", default=None)
    p.add_argument("--hyde", action="store_true", default=None, help="buscar también con un pasaje hipotético")
    p = sub.add_parser("check-questions"); p.add_argument("saga")
    p = sub.add_parser("retrieval"); p.add_argument("saga", help="lotr | hp | dune | narnia | all")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--dense", default="lsa")
    p.add_argument("--global", dest="use_global", action="store_true",
                   help="evaluar contra el índice único (con enrutamiento de saga y reranker)")
    p.add_argument("--hyde", action="store_true", help="agrega el modo hybrid_routed_hyde (requiere llama-server)")
    p = sub.add_parser("eval", help="benchmark de respuestas (closed-book vs RAG)")
    p.add_argument("saga", help="lotr | hp | dune | narnia | all"); p.add_argument("--client", default="local")
    p.add_argument("--conditions", nargs="+", default=["closed_book", "rag_global"])
    p.add_argument("--k", type=int, default=8); p.add_argument("--dense", default="lsa")
    p.add_argument("--import", dest="import_answers", default=None); p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lang", choices=["en", "es"], default="en", help="es: usa question_es del banco")
    p.add_argument("--reuse", default=None, help="carpeta de una corrida previa: copia las condiciones no corridas")
    p = sub.add_parser("serve", help="levanta llama-server para un modelo de rag.llm.models")
    p.add_argument("model", nargs="?", default=None, help="phi4mini | gemma3_4b | ... (por defecto: rag.llm.default)")
    p.add_argument("--ctx", type=int, default=8192); p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--dry", action="store_true", help="solo imprime el comando")
    sub.add_parser("graph", help="imprime el grafo del RAG (Mermaid; requiere langgraph)")
    p = sub.add_parser("eval-rescore", help="re-puntúa una corrida con la métrica actual (sin LLM)")
    p.add_argument("run_dir")
    p = sub.add_parser("eval-score", help="aplica los puntajes manuales de review.csv a una corrida")
    p.add_argument("review_csv")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "urllib3", "filelock", "huggingface_hub", "sentence_transformers",
                  "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.ERROR)     # solo errores de librerías externas
    if a.cmd in ("index", "ask", "retrieval", "eval", "eval-score"):
        from .rag.index import rag_config
        from .rag.retrievers import prefer_offline_if_cached
        cfg = rag_config()
        prefer_offline_if_cached([cfg.get("embedding_model"), cfg.get("reranker_model")])

    if a.cmd == "build":
        from .ingest.build import build_saga
        print(json.dumps(build_saga(a.saga), indent=2))
    elif a.cmd == "inspect":
        import pandas as pd
        from .ingest.build import units_for
        df = pd.DataFrame([{"source": u.source_id, "work": u.work, "medium": u.medium, "words": u.n_words,
                            "lines": len(u.lines), "locator": u.locator} for u in units_for(a.saga)])
        print(df.groupby(["source", "work"]).agg(units=("locator", "size"), words=("words", "sum"),
                                                 lines=("lines", "sum")).to_string())
        print(df[df.medium == "book"][["work", "locator", "words"]].to_string(index=False))
    elif a.cmd == "analyze":
        from .analysis.pipeline import analyze_saga
        s = analyze_saga(a.saga, method="neural" if a.neural else "lsa", n_perm=a.perm)
        print(json.dumps({k: v for k, v in s.items() if k != "works"}, indent=2, default=float, ensure_ascii=False))
    elif a.cmd == "rechunk":
        from .ingest.build import rechunk_saga
        from .rag.index import indexed_sagas
        for s in a.sagas or indexed_sagas():
            print(rechunk_saga(s))
    elif a.cmd == "index":
        from .rag.index import RagIndex
        idx = RagIndex.load_or_build("all", a.dense, rebuild=a.rebuild)
        import collections
        print(f"Índice: {idx.path()}  ({len(idx.chunks)} fragmentos, denso={idx.dense_kind})")
        print(dict(collections.Counter(c.saga for c in idx.chunks)))
    elif a.cmd == "ask":
        from .rag.qa import ask
        r = ask(a.question, a.k, a.client, a.saga, a.dense, a.hyde)
        if r.tool:
            print(f"Herramienta cuantitativa ({r.tool}):\n{r.text}")
            return 0
        print(f"Saga: {r.saga or '(todas)'} · medio: {r.medium or '(ambos)'} · obra: {r.work or '-'}"
              f"   consulta: {r.query}\n")
        if r.hyde:
            print(f"Pasaje hipotético (HyDE, solo guía la búsqueda):\n  {r.hyde}\n")
        from .rag.qa import passage_groups
        for title, hits in passage_groups(r):
            if title:
                print(f"── {title} (citas [n] de la respuesta '{title}') " + "─" * 30)
            for h in hits:
                print(f"[{h.rank}] {h.score:.3f}  {h.citation}\n     {h.chunk.text[:300]}…\n")
        if a.client:
            print("=" * 80)
            if r.evidence:
                print(f"Evidencia:\n{r.evidence}\n")
            print(r.text if r.text is not None else f"(sin respuesta del LLM: {r.error})")
    elif a.cmd == "check-questions":
        from .eval.runner import check_questions
        df = check_questions(a.saga)
        print(df.to_string(index=False))
        bad = df[df.n_relevant_chunks == 0]
        if len(bad):
            print(f"\n{len(bad)} evidencias SIN chunks relevantes: revisar términos.")
            return 1
    elif a.cmd == "retrieval":
        from .config import paths
        from .eval.runner import retrieval_eval
        import pandas as pd
        from .rag.index import indexed_sagas
        sagas = indexed_sagas() if a.saga == "all" else [a.saga]
        cols = ["recall@k", "hit@k", "mrr", "ndcg@k"] + (["saga_ok", "frac_misma_saga"] if a.use_global else [])
        suffix = f"{a.dense}{'_global' if a.use_global else ''}"
        frames = []
        for s in sagas:
            df = retrieval_eval(s, a.k, dense=a.dense, use_global=a.use_global, hyde=a.hyde)
            df.to_csv(paths().reports / f"retrieval_{s}_{suffix}.csv", index=False)
            frames.append(df.assign(saga=s))
            print(f"\n== {s}\n" + df.groupby("mode")[cols].mean().round(3).to_string())
        df = pd.concat(frames, ignore_index=True)
        if len(frames) > 1:
            df.to_csv(paths().reports / f"retrieval_all_{suffix}.csv", index=False)
            print("\n== TODAS LAS SAGAS (promedio por pregunta)\n" +
                  df.groupby("mode")[cols].mean().round(3).to_string())
        from .eval.runner import REF_MODE, retrieval_stats
        ref = REF_MODE if a.use_global else "hybrid_routed"
        st, comp = retrieval_stats(df, ref)
        print("\n== Intervalos de confianza 95 % (bootstrap por pregunta)\n" + st.round(3).to_string(index=False))
        if not comp.empty:
            print(f"\n== Comparación pareada vs {ref} (McNemar sobre hit@k; IC de la diferencia de MRR)\n"
                  + comp.round(3).to_string(index=False))
        print("\n== Por categoría (recall@k)\n" +
              df.pivot_table(index="category", columns="mode", values="recall@k").round(3).to_string())
    elif a.cmd == "eval":
        from .eval.runner import run
        out = run(a.saga, a.client, a.conditions, a.k, a.dense, a.import_answers, a.limit, a.lang, a.reuse)
        print(f"Salida: {out}")
        if out.is_dir():
            print((out / "report.md").read_text(encoding="utf-8"))
    elif a.cmd == "graph":
        from .rag.graph import mermaid
        print(mermaid())
    elif a.cmd == "serve":
        import subprocess
        from urllib.parse import urlparse

        from .config import project_root
        from .rag.llm import local_models_config
        cfg = local_models_config()
        key = a.model or cfg.get("default")
        m = (cfg.get("models") or {}).get(key)
        if not m:
            raise SystemExit(f"Modelo desconocido: {key}. Opciones: {sorted(cfg.get('models') or {})}")
        root = project_root()
        server, gguf = (root / cfg["server"]).resolve(), (root / m["gguf"]).resolve()
        port = str(urlparse(m["url"]).port or 8080)
        cmd = [str(server), "-m", str(gguf), "-c", str(a.ctx), "-ngl", str(a.ngl), "--port", port]
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        if not a.dry:
            if not gguf.exists():
                raise SystemExit(f"No existe el modelo: {gguf}")
            subprocess.run(cmd, check=False)
    elif a.cmd == "eval-rescore":
        from .eval.runner import rescore
        out = rescore(a.run_dir)
        print((out / "report.md").read_text(encoding="utf-8"))
    elif a.cmd == "eval-score":
        from .eval.runner import apply_review
        out = apply_review(a.review_csv)
        print((out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
