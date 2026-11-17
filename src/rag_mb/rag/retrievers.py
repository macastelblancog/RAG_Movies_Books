r"""Recuperadores: BM25 (léxico), denso (LSA o neuronal) e híbrido por Reciprocal Rank Fusion.

BM25 (Robertson & Zaragoza 2009), implementado de forma vectorizada sobre una matriz dispersa:
    idf(t)   = ln(1 + (N − df_t + ½)/(df_t + ½))
    s(q, d)  = Σ_{t∈q} idf(t) · tf_{t,d}(k₁+1) / (tf_{t,d} + k₁(1 − b + b·|d|/avgdl))

Denso: coseno entre el embedding de la consulta y el de cada chunk.
    LSA: q ↦ Vₖᵀ tfidf(q)  (fold-in en el subespacio latente)

Neuronal: bi-encoder multilingüe (E5); reranker: cross-encoder sobre los mejores candidatos.

RRF (Cormack et al. 2009):  s(d) = Σ_r 1/(k + rank_r(d)),  k = 60.
    No requiere calibrar escalas entre recuperadores (solo usa rangos).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from ..analysis.vectorize import make_tfidf
from ..text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]) -> "BM25":
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsc().astype(float)
        N = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf = tf.tocoo()
        data = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.W = sparse.csr_matrix((data, (tf.row, tf.col)), shape=tf.shape)   # pesos BM25 por (d,t)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        q.data[:] = 1.0                                   # términos únicos de la consulta
        return np.asarray(self.W @ (q.multiply(self.idf)).T.toarray()).ravel()


class DenseLSA:
    def __init__(self, k: int = 256, seed: int = 1998):
        self.k, self.seed = k, seed

    def fit(self, texts: list[str]) -> "DenseLSA":
        self.tfidf = make_tfidf(min_df=2, max_df=0.5)
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.k, X.shape[1] - 1), random_state=self.seed)
        Z = self.svd.fit_transform(X)
        self.Z = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.svd.transform(self.tfidf.transform([query]))[0]
        q /= max(np.linalg.norm(q), 1e-12)
        return self.Z @ q


def prefer_offline_if_cached(models: list[str]) -> bool:
    """Si todos los modelos ya están en la caché de Hugging Face, trabajar sin consultar Internet
    (evita las verificaciones HEAD/404 en cada carga). Debe llamarse ANTES de importar huggingface_hub."""
    import os
    from pathlib import Path
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")        # huggingface_hub reconfigura su logger al importar
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    cached = all(any((hub / f"models--{m.replace('/', '--')}" / "snapshots").glob("*")) for m in models if m)
    if cached:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return cached


class DenseNeural:
    """Embeddings neuronales (sentence-transformers). Con E5 se antepone 'query: ' / 'passage: '."""

    def __init__(self, model: str = "intfloat/multilingual-e5-base", query_prefix: str = "",
                 passage_prefix: str = "", max_seq_length: int | None = None):
        self.model_name, self.qp, self.pp, self.max_len = model, query_prefix, passage_prefix, max_seq_length

    def _load(self):
        if getattr(self, "model", None) is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            if self.max_len:
                self.model.max_seq_length = self.max_len
        return self.model

    def fit(self, texts: list[str]) -> "DenseNeural":
        self.Z = self._load().encode([self.pp + t for t in texts], normalize_embeddings=True,
                                     show_progress_bar=True, batch_size=32).astype(np.float32)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self._load().encode([self.qp + query], normalize_embeddings=True)[0]
        return self.Z @ q

    def scores_passage(self, text: str) -> np.ndarray:
        """Similitud documento-documento (HyDE): el texto hipotético se codifica como PASAJE."""
        q = self._load().encode([self.pp + text], normalize_embeddings=True)[0]
        return self.Z @ q

    def __getstate__(self):          # no serializar el modelo; se recarga al usar
        d = dict(self.__dict__)
        d.pop("model", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.model = None


class CrossEncoderReranker:
    """Reordenamiento: el cross-encoder lee (pregunta, pasaje) JUNTOS, a diferencia del bi-encoder,
    que compara vectores calculados por separado. Más caro → solo sobre los mejores `pool` candidatos."""

    def __init__(self, model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.model_name, self.model = model, None

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=512)
        return np.asarray(self.model.predict([(query, p) for p in passages]), dtype=float)

    def __getstate__(self):
        return {"model_name": self.model_name}

    def __setstate__(self, d):
        self.model_name, self.model = d["model_name"], None


def rrf(rank_lists: list[np.ndarray], n: int, k: int = 60) -> np.ndarray:
    s = np.zeros(n)
    for ranks in rank_lists:
        s[ranks] += 1.0 / (k + np.arange(1, len(ranks) + 1))
    return s
