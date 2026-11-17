r"""Desplazamiento semántico libro → guion con vectores de palabra y Procrustes ortogonal.

1) Vectores de palabra por factorización de PPMI (Levy & Goldberg 2014 muestran que
   skip-gram con negative sampling factoriza implícitamente una PMI desplazada):
       PMI(w,c) = ln [ P(w,c) / (P(w) P_α(c)) ],  P_α(c) ∝ #(c)^α (α=0.75, suavizado de contexto)
       PPMI = max(PMI, 0);   PPMI ≈ U_k Σ_k V_kᵀ;   W = U_k Σ_k^{1/2}
2) Los espacios del libro (A) y del guion (B) son arbitrarios salvo rotación. Se alinean
   sobre el vocabulario común (anclas) resolviendo Procrustes ortogonal:
       R* = argmin_{RᵀR=I} ‖A R − B‖_F  =  U Vᵀ,   donde  Aᵀ B = U Σ Vᵀ   (SVD)
3) Desplazamiento de w:  shift(w) = 1 − cos(A_w R*, B_w). Las palabras con mayor shift
   son las que la adaptación usa en contextos distintos.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds


def ppmi_vectors(token_lists: list[list[str]], window: int = 5, min_count: int = 5,
                 dim: int = 100, cds: float = 0.75) -> tuple[np.ndarray, list[str]]:
    freq = Counter(t for toks in token_lists for t in toks)
    vocab = sorted(w for w, c in freq.items() if c >= min_count)
    idx = {w: i for i, w in enumerate(vocab)}
    rows, cols = [], []
    for toks in token_lists:
        ids = [idx[t] for t in toks if t in idx]
        for p, wi in enumerate(ids):
            for q in range(max(0, p - window), min(len(ids), p + window + 1)):
                if q != p:
                    rows.append(wi)
                    cols.append(ids[q])
    V = len(vocab)
    M = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V)).tocsr()
    M.sum_duplicates()
    total = M.sum()
    pw = np.asarray(M.sum(1)).ravel() / total
    pc = np.asarray(M.sum(0)).ravel() ** cds
    pc /= pc.sum()
    M = M.tocoo()
    pmi = np.log((M.data / total) / (pw[M.row] * pc[M.col]))
    keep = pmi > 0
    P = sparse.csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=(V, V))
    k = min(dim, V - 2)
    U, S, _ = svds(P, k=k)
    W = U * np.sqrt(S)
    W /= np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-12, None)
    return W, vocab


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def semantic_shift(Wa: np.ndarray, va: list[str], Wb: np.ndarray, vb: list[str],
                   counts_a: Counter, counts_b: Counter, min_count: int = 10) -> pd.DataFrame:
    ia, ib = {w: i for i, w in enumerate(va)}, {w: i for i, w in enumerate(vb)}
    common = [w for w in va if w in ib]
    A, B = Wa[[ia[w] for w in common]], Wb[[ib[w] for w in common]]
    R = orthogonal_procrustes(A, B)
    AR = A @ R
    cos = np.sum(AR * B, axis=1) / (np.linalg.norm(AR, axis=1) * np.linalg.norm(B, axis=1))
    df = pd.DataFrame({"word": common, "shift": 1 - cos,
                       "count_book": [counts_a.get(w, 0) for w in common],
                       "count_film": [counts_b.get(w, 0) for w in common]})
    df = df[(df.count_book >= min_count) & (df.count_film >= min_count)]
    return df.sort_values("shift", ascending=False).reset_index(drop=True)


def neighbors(W: np.ndarray, vocab: list[str], word: str, k: int = 8) -> list[str]:
    i = vocab.index(word)
    sims = W @ W[i]
    return [vocab[j] for j in np.argsort(-sims)[1:k + 1]]
