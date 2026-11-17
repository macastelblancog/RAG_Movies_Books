r"""Estadística léxica: leyes de Zipf/Heaps, entropía, Jensen-Shannon y log-odds de Dirichlet.

Zipf:   f(r) ∝ r^{-α}.  Se estima por MCO en escala log-log sobre los rangos 1..r_max (descriptivo)
        y por máxima verosimilitud (aproximación continua de Clauset-Shalizi-Newman 2009) sobre
        las frecuencias ≥ x_min:   α̂ = 1 + n [ Σ_i ln(x_i / (x_min − ½)) ]^{-1}.
Heaps:  V(n) = K n^{β}, 0<β<1.  MCO en log-log sobre la curva de crecimiento del vocabulario.
JS:     JS(P‖Q) = ½ KL(P‖M) + ½ KL(Q‖M), M = ½(P+Q); en bits, acotada en [0, 1].
Log-odds con prior de Dirichlet informativo (Monroe, Colaresi & Quinn 2008):
        δ_w = ln[(y_w^a+α_w)/(n^a+α_0−y_w^a−α_w)] − ln[(y_w^b+α_w)/(n^b+α_0−y_w^b−α_w)]
        σ²(δ_w) ≈ 1/(y_w^a+α_w) + 1/(y_w^b+α_w),   z_w = δ_w / σ(δ_w)
        con α_w = α_0 · p_w(fondo). Corrige el sesgo de las frecuencias crudas hacia palabras
        raras (ratio inestable) y hacia palabras comunes (diferencias absolutas grandes).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def counts(token_lists: Iterable[list[str]]) -> Counter:
    c: Counter = Counter()
    for toks in token_lists:
        c.update(toks)
    return c


@dataclass
class PowerLawFit:
    alpha: float
    intercept: float
    r2: float
    alpha_mle: float | None = None
    x_min: int | None = None


def zipf_fit(c: Counter, r_max: int = 1000, x_min: int = 5) -> PowerLawFit:
    f = np.array(sorted(c.values(), reverse=True), dtype=float)
    r = np.arange(1, len(f) + 1, dtype=float)
    k = min(r_max, len(f))
    x, y = np.log(r[:k]), np.log(f[:k])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    tail = f[f >= x_min]
    # MLE del exponente de la distribución de frecuencias p(x) ∝ x^{-a}; relación a = 1 + 1/α_Zipf
    a_mle = 1 + len(tail) / np.sum(np.log(tail / (x_min - 0.5))) if len(tail) else np.nan
    alpha_zipf_from_mle = 1 / (a_mle - 1) if a_mle > 1 else np.nan
    return PowerLawFit(alpha=-slope, intercept=intercept, r2=r2, alpha_mle=alpha_zipf_from_mle, x_min=x_min)


def heaps_curve(tokens: list[str], n_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    ns = np.unique(np.geomspace(100, max(len(tokens), 101), n_points).astype(int))
    seen, V, j = set(), [], 0
    for n in ns:
        seen.update(tokens[j:n])
        j = n
        V.append(len(seen))
    return ns, np.array(V)


def heaps_fit(tokens: list[str]) -> PowerLawFit:
    n, V = heaps_curve(tokens)
    beta, logK = np.polyfit(np.log(n), np.log(V), 1)
    yhat = beta * np.log(n) + logK
    y = np.log(V)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return PowerLawFit(alpha=beta, intercept=float(np.exp(logK)), r2=r2)


def distribution(c: Counter, vocab: list[str]) -> np.ndarray:
    v = np.array([c.get(w, 0) for w in vocab], dtype=float)
    return v / v.sum() if v.sum() else v


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def js_divergence(ca: Counter, cb: Counter, vocab: list[str] | None = None) -> float:
    vocab = vocab or sorted(set(ca) | set(cb))
    p, q = distribution(ca, vocab), distribution(cb, vocab)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dirichlet_log_odds(ca: Counter, cb: Counter, background: Counter | None = None,
                       alpha0: float = 1000.0, min_count: int = 5) -> pd.DataFrame:
    """z-scores de log-odds (a vs b). z>1.96 ⇒ sobrerrepresentada en a (95%)."""
    background = background or (ca + cb)
    vocab = [w for w in set(ca) | set(cb) if ca.get(w, 0) + cb.get(w, 0) >= min_count]
    na, nb = sum(ca.values()), sum(cb.values())
    nbg = sum(background.values())
    rows = []
    for w in vocab:
        aw = alpha0 * background.get(w, 0) / nbg
        if aw <= 0:
            continue
        ya, yb = ca.get(w, 0), cb.get(w, 0)
        la = np.log((ya + aw) / (na + alpha0 - ya - aw))
        lb = np.log((yb + aw) / (nb + alpha0 - yb - aw))
        var = 1 / (ya + aw) + 1 / (yb + aw)
        rows.append((w, ya, yb, la - lb, (la - lb) / np.sqrt(var)))
    df = pd.DataFrame(rows, columns=["word", "count_a", "count_b", "delta", "z"])
    return df.sort_values("z", ascending=False).reset_index(drop=True)
