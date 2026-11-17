r"""Representaciones vectoriales de unidades (capítulos / escenas).

LSA: X ∈ ℝ^{n×V} (TF-IDF sublineal) ≈ U_k Σ_k V_kᵀ (SVD truncada, Eckart–Young: mejor
aproximación de rango k en norma de Frobenius). Embedding de documento = filas de U_k Σ_k,
normalizadas en L2 para que el producto interno sea el coseno.

Se ajusta sobre la UNIÓN de libro y guion para que ambos vivan en el mismo subespacio
(condición necesaria para comparar distribuciones con transporte óptimo).

`neural`: sentence-transformers (opcional). Ventaja: semántica contextual; costo: caja negra
y descarga de modelo. Se reportan ambos para mostrar robustez de las conclusiones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import tokenize


@dataclass
class Embedding:
    vectors: np.ndarray            # (n, k) normalizado L2
    explained_variance: float | None
    method: str


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def make_tfidf(min_df: int = 2, max_df: float = 0.9) -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                           sublinear_tf=True, min_df=min_df, max_df=max_df)


def lsa_embed(texts: list[str], k: int = 100, seed: int = 1998) -> tuple[Embedding, TfidfVectorizer, TruncatedSVD]:
    tfidf = make_tfidf()
    X = tfidf.fit_transform(texts)
    k = min(k, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Z = svd.fit_transform(X)
    return Embedding(_l2(Z), float(svd.explained_variance_ratio_.sum()), f"lsa{k}"), tfidf, svd


def neural_embed(texts: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32) -> Embedding:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install -e .[neural]") from e
    m = SentenceTransformer(model)
    Z = m.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return Embedding(np.asarray(Z), None, f"st:{model.split('/')[-1]}")


def embed(texts: list[str], method: str = "lsa", **kw) -> Embedding:
    if method == "lsa":
        return lsa_embed(texts, **kw)[0]
    if method == "neural":
        return neural_embed(texts, **kw)
    raise ValueError(method)


def cosine_cost(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """C_ij = 1 − cos(x_i, y_j) ∈ [0, 2] (vectores ya normalizados)."""
    return np.clip(1.0 - X @ Y.T, 0.0, 2.0)
