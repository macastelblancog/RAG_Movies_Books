r"""Transporte óptimo entre el libro y la película: el plan de transporte ES el mapa de adaptación.

Sean a ∈ Δ_n (masa de cada capítulo, ∝ nº de palabras) y b ∈ Δ_m (masa de cada escena),
C_ij = 1 − cos(capítulo_i, escena_j).

1) OT entrópico (Cuturi 2013):
       P* = argmin_{P ∈ U(a,b)} ⟨P, C⟩ − ε H(P),   U(a,b) = {P ≥ 0 : P1 = a, Pᵀ1 = b}
   Solución P* = diag(u) K diag(v), K = exp(−C/ε); iteraciones de Sinkhorn
       u ← a ⊘ (K v),   v ← b ⊘ (Kᵀ u)
   implementadas en dominio logarítmico (estables para ε pequeño).

2) OT NO balanceado (Chizat, Peyré, Schmitzer & Vialard 2018): las restricciones de
   marginal se relajan con penalización KL de peso τ:
       min_P ⟨P, C⟩ − ε H(P) + τ KL(P1‖a) + τ KL(Pᵀ1‖b)
   Iteraciones: u ← (a ⊘ K v)^λ, v ← (b ⊘ Kᵀ u)^λ, λ = τ/(τ+ε).
   Interpretación: el guion NO tiene que "cubrir" todo el libro. La masa retenida por
   capítulo  ρ_i = (P1)_i / a_i  mide qué tanto de ese capítulo sobrevive en la película
   (ρ_i ≈ 0 ⇒ capítulo omitido, p.ej. Tom Bombadil).

3) Proyección baricéntrica: posición esperada en la película del capítulo i
       t̂_i = Σ_j P_ij t_j / Σ_j P_ij ,   t_j ∈ [0,1] posición normalizada de la escena.
   La correlación de Spearman entre i y t̂_i mide la fidelidad al ORDEN narrativo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr


@dataclass
class OTResult:
    plan: np.ndarray
    cost: float              # ⟨P, C⟩
    n_iter: int
    marginal_error: float
    retained: np.ndarray     # ρ_i = (P1)_i / a_i
    received: np.ndarray     # (Pᵀ1)_j / b_j
    barycentric_pos: np.ndarray
    order_spearman: float


def _finish(logP, C, a, b, it, err) -> OTResult:
    P = np.exp(logP)
    t = np.linspace(0, 1, C.shape[1])
    row = P.sum(1)
    bary = (P @ t) / np.clip(row, 1e-300, None)
    rho = spearmanr(np.arange(len(a)), bary).statistic if len(a) > 2 else np.nan
    return OTResult(P, float((P * C).sum()), it, err, row / a, P.sum(0) / b, bary, float(rho))


def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, eps: float = 0.05,
             n_iter: int = 2000, tol: float = 1e-9, tau: float | None = None) -> OTResult:
    """Sinkhorn log-estabilizado. tau=None → balanceado; tau>0 → no balanceado (KL)."""
    a = np.asarray(a, float) / np.sum(a)
    b = np.asarray(b, float) / np.sum(b)
    la, lb = np.log(a), np.log(b)
    lK = -C / eps
    lam = 1.0 if tau is None else tau / (tau + eps)
    lu, lv = np.zeros_like(a), np.zeros_like(b)
    err, it = np.inf, 0
    for it in range(1, n_iter + 1):
        lu_new = lam * (la - logsumexp(lK + lv[None, :], axis=1))
        lv_new = lam * (lb - logsumexp(lK.T + lu_new[None, :], axis=1))
        err = max(np.abs(lu_new - lu).max(), np.abs(lv_new - lv).max())
        lu, lv = lu_new, lv_new
        if err < tol:
            break
    logP = lu[:, None] + lK + lv[None, :]
    P = np.exp(logP)
    marg = float(np.abs(P.sum(1) - a).sum() + np.abs(P.sum(0) - b).sum())
    return _finish(logP, C, a, b, it, marg)


def wasserstein_permutation_test(X: np.ndarray, Y: np.ndarray, a: np.ndarray, b: np.ndarray,
                                 eps: float = 0.05, n_perm: int = 200, seed: int = 0) -> dict:
    """¿Es el costo OT libro↔película menor que el de un emparejamiento aleatorio de etiquetas?

    H0: capítulos y escenas son intercambiables (misma distribución). Se permutan las etiquetas
    sobre la unión y se recalcula el costo; p = P(costo_perm ≥ costo_obs).
    p pequeño ⇒ el costo observado es mayor que bajo intercambiabilidad ⇒ libro y guion ocupan
    regiones distintas del espacio semántico (diferencia de registro/contenido, no solo ruido).
    """
    from .vectorize import cosine_cost

    rng = np.random.default_rng(seed)
    obs = sinkhorn(a, b, cosine_cost(X, Y), eps=eps).cost
    Z, w = np.vstack([X, Y]), np.concatenate([a / a.sum(), b / b.sum()])
    n = len(X)
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(Z))
        i1, i2 = idx[:n], idx[n:]
        null.append(sinkhorn(w[i1], w[i2], cosine_cost(Z[i1], Z[i2]), eps=eps, n_iter=500).cost)
    null = np.array(null)
    return {"cost": obs, "null_mean": float(null.mean()), "null_std": float(null.std()),
            "p_value_greater": float((np.sum(null >= obs) + 1) / (n_perm + 1))}
