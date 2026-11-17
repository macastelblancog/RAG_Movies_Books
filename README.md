# RAG_Movies_Books

Análisis de las diferencias entre libros y sus adaptaciones al cine (*The Lord of the Rings*, *Harry Potter*,
*Dune*, *The Chronicles of Narnia*) y un sistema RAG local que responde preguntas sobre ambos medios con citas.

Este README cubre **solo la ejecución**.

---

## Requisitos

| Componente | Requisito |
|---|---|
| Python | 3.10 o superior |
| GPU (opcional) | NVIDIA con CUDA. El proyecto se ejecutó en una RTX 3050 de 6 GB |
| LLM local (opcional) | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) y modelos GGUF; ver [LLM local](#llm-local) |
| Datos | Copias propias de los libros y guiones; ver [Datos](#datos) |

Sin GPU todo funciona en CPU, más lento. Sin LLM local, el RAG puede devolver solo los pasajes
(`--client extractive`) o usar una API (`--client openai | gemini`).

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -e ".[dev]"             # núcleo: numpy, scipy, pandas, scikit-learn, pdfplumber, pypdf, networkx
pip install -e ".[neural]"          # embeddings y reranker (sentence-transformers)
pip install -e ".[graph]"           # orquestación del RAG con LangGraph
pip install -e ".[llm]"             # clientes de API (openai, google-genai), opcional
```

Para usar la GPU, instalar antes la versión de PyTorch con CUDA que corresponda al equipo
(ver las instrucciones de pytorch.org). `pypdf` y `pdfplumber` son Python puro y no requieren poppler.

Los modelos de Hugging Face (`intfloat/multilingual-e5-base` y
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) se descargan solos la primera vez; después el
proyecto trabaja sin conexión si ya están en la caché.

## Datos

Los libros y guiones tienen derechos de autor y **no se versionan**. Cada persona debe conseguir sus propias
copias y ubicarlas según `data/raw/README.md`, que también documenta la procedencia y la licencia de cada
fuente. Las rutas y reglas de lectura de cada fuente están en `configs/corpora.yaml`.

```
data/raw/
├── lotr/    book/, screenplay/, kaggle/
├── hp/      book/, maven/, kaggle/
├── dune/    book/, screenplay/
└── narnia/  book/, screenplay/   (el guion de LWW usa una caché de OCR: lww_2005.pdf.layout.json)
```

## Configuración

| Archivo | Contenido |
|---|---|
| `configs/corpora.yaml` | Fuente única de corpus, rutas, segmentación de capítulos y escenas, parámetros del análisis y del RAG (`rag:`) |
| `configs/characters.yaml` | Fuente única de personajes: nombres, alias de hablante y notas |
| `questions/<saga>.yaml` | Banco de preguntas del benchmark, con respuesta de referencia y evidencia esperada |

Variables de entorno:

| Variable | Uso |
|---|---|
| `RMB_ROOT` | Raíz del proyecto, si se ejecuta desde otra carpeta |
| `RMB_LOCAL_URL`, `RMB_LOCAL_MODEL` | Sobrescriben la URL y el nombre del LLM local |
| `OPENAI_API_KEY`, `RMB_OPENAI_MODEL` | Cliente OpenAI |
| `GEMINI_API_KEY`, `RMB_GEMINI_MODEL` | Cliente Gemini |
| `HF_HOME`, `HF_HUB_CACHE` | Ubicación de la caché de Hugging Face |

## Flujo de ejecución

```bash
# 1. Ingesta: PDFs y CSV → data/processed/<saga>_{units,chunks}.jsonl
rmb build lotr                  # ~1 min
rmb build hp                    # ~4 min
rmb build dune                  # ~1 min
rmb build narnia                # ~1 min
rmb inspect lotr                # verificar capítulos y escenas detectados

# 2. Análisis libro vs película → reports/figures/<saga>/
rmb analyze lotr                # LSA
rmb analyze lotr --neural       # el mismo análisis con embeddings neuronales
rmb analyze lotr --perm 200     # agrega la prueba de permutación del transporte óptimo

# 3. Índice único del RAG (todas las sagas) → reports/indexes/
rmb index                       # usa rag.dense del config (neural por defecto)
rmb index --rebuild

# 4. Verificación del banco y métricas de recuperación (sin LLM)
rmb check-questions lotr                       # cada evidencia esperada existe en el corpus
rmb retrieval lotr                             # índice por saga (LSA)
rmb retrieval all --global --dense neural      # índice único, enrutador y reranker
```

`rmb rechunk` vuelve a cortar los fragmentos desde las unidades ya procesadas, sin releer los PDF.

## LLM local

Cada modelo se declara en `configs/corpora.yaml → rag.llm.models`, con su archivo GGUF y su puerto.
Las rutas de `llama-server` y de los GGUF son relativas a la raíz del repositorio; por defecto apuntan a
`../AGENTIC_LLM/`. Ajustarlas en el config si están en otro lugar.

| Nombre | Modelo | Puerto |
|---|---|---|
| `phi4mini` | `Phi-4-mini-instruct-Q4_K_M.gguf` | 8080 |
| `gemma3_4b` | `gemma-3-4b-it-Q4_K_M.gguf` | 8081 |

Con 6 GB de VRAM, levantar **un modelo a la vez**:

```bash
rmb serve phi4mini              # opciones: --ctx 8192 (contexto)  --ngl 99 (capas en GPU)  --dry
```

En otra terminal:

```bash
rmb ask "¿Cómo muere Boromir en el libro?" --client local:phi4mini   # la saga se detecta sola
rmb ask "..." --client extractive                                     # solo pasajes, sin LLM
rmb graph                                                             # diagrama Mermaid del grafo del RAG
```

## Benchmark

```bash
# LLM local: sin RAG frente a RAG
rmb eval all --client local:phi4mini --conditions closed_book rag_graph
rmb eval all --client local:gemma3_4b --conditions closed_book rag_graph
rmb eval all --client local:phi4mini --lang es        # preguntas en español (question_es)

# Reutilizar las condiciones de una corrida anterior y correr solo las nuevas
rmb eval all --client local:phi4mini --conditions rag_graph --reuse reports/runs/<corrida>

# APIs
rmb eval lotr --client openai --conditions closed_book rag_global
rmb eval lotr --client gemini --conditions closed_book rag_global

# Sin API: exportar los prompts, responder en la interfaz web y volver a importar
rmb eval lotr --client manual --conditions closed_book rag_global
rmb eval lotr --client manual --conditions closed_book rag_global --import reports/prompts/lotr_prompts.csv

# Puntuación
rmb eval-rescore reports/runs/<corrida>               # re-puntúa con la métrica actual, sin LLM
rmb eval-score reports/runs/<corrida>/review.csv      # aplica los puntajes manuales de review.csv
```

Condiciones disponibles: `closed_book`, `rag_global` (pipeline lineal), `rag_graph` (pipeline orquestado
con LangGraph) y las de ablación `rag_bm25`, `rag_dense`, `rag_hybrid`, `rag_hybrid_balanced`,
`rag_bm25_routed`, `rag_hybrid_routed`.

Cada corrida queda en `reports/runs/<saga>_<cliente>_<idioma>_<fecha>/` con `results.jsonl`,
`summary.csv`, `review.csv` y `report.md`. Estos archivos contienen citas de los textos y no se versionan.

## Pruebas

```bash
pytest -q
```

Las pruebas usan datos sintéticos y soluciones conocidas (transporte óptimo, DTW, Procrustes, BM25,
parser de guiones, enrutador); no necesitan los textos. `tests/test_graph.py` se omite si LangGraph no
está instalado.

## Paquete privado (datos y resultados no versionados)

`scripts/private_bundle.py` empaqueta lo que no va a git (los textos, los fragmentos procesados, el índice y los
resultados) en un `.tar.gz` **fuera del repositorio**, con la huella SHA-256 de cada archivo. Contiene textos con
derechos de autor: guardarlo solo en un lugar privado.

```bash
# En el equipo de origen
python scripts/private_bundle.py pack --out ../RAG_Movies_Books_privado.tar.gz
#   --no-processed / --no-index / --no-results / --all-indexes / --runs <corrida> ...

# En el equipo de destino, desde la raíz del repositorio clonado
python scripts/private_bundle.py verify ../RAG_Movies_Books_privado.tar.gz
python scripts/private_bundle.py unpack ../RAG_Movies_Books_privado.tar.gz
```

`unpack` restaura las rutas, verifica las huellas y marca los índices como más recientes que los fragmentos, para
que el RAG los reutilice en lugar de reconstruirlos.

### Equipo sin GPU

- Con el índice restaurado no hace falta `rmb build` ni `rmb index`: reconstruir el índice neural en CPU es lento.
  La búsqueda, el reranker y `rmb retrieval` funcionan en CPU (e5 y el reranker se descargan la primera vez).
- `rmb ask "..." --client extractive` devuelve los pasajes sin LLM.
- `rmb eval-rescore reports/runs/<corrida>` vuelve a puntuar una corrida restaurada sin LLM.
- llama.cpp también corre en CPU (`rmb serve phi4mini --ngl 0`), pero bastante más lento.

## Reproducir en otro equipo

1. Clonar, crear el entorno e instalar los extras (ver [Instalación](#instalación)).
2. Copiar `data/raw/` según `data/raw/README.md`.
3. `pytest -q`.
4. `rmb build` para cada saga y `rmb index`.
5. `rmb check-questions <saga>` para cada saga: no debe reportar evidencias sin fragmentos.
6. `rmb retrieval all --global --dense neural`: Hit@8 ≈ 0.84 y MRR ≈ 0.68 en la configuración
   `hybrid_routed`.
7. `rmb serve phi4mini` y `rmb eval all --client local:phi4mini --conditions rag_graph`: exactitud ≈ 0.62.
   Otra GPU u otra versión de llama.cpp pueden cambiar una o dos respuestas.

## Estructura

```
RAG_Movies_Books/
├── configs/                    # corpora.yaml y characters.yaml (fuentes únicas)
├── data/raw/README.md          # procedencia y licencias (los datos no se versionan)
├── questions/<saga>.yaml       # banco de preguntas del benchmark
├── scripts/private_bundle.py   # paquete privado de datos y resultados
├── src/rag_mb/
│   ├── schema.py               # Unit (capítulo/escena) y Chunk citables
│   ├── ingest/                 # book_pdf, screenplay_pdf (geometría), tabular (CSV/relacional/SRT), chunking
│   ├── analysis/               # lexical, vectorize, transport, dtw, semantic_shift, graphs, quotes,
│   │                           #   character_stats, stats, pipeline
│   ├── rag/                    # retrievers, router, index, tools, llm, qa, graph, hyde
│   ├── eval/                   # metrics, gold (preguntas cuantitativas), runner
│   └── cli.py                  # comando `rmb`
├── notebooks/informe.ipynb
├── reports/                    # salidas (no versionadas)
└── tests/
```

## Datos de terceros

El código no incluye textos de los libros ni de los guiones. Los diálogos de *Harry Potter* de Maven
Analytics se usan bajo CC BY-SA 3.0; la atribución y el resto de las fuentes están en `data/raw/README.md`.
