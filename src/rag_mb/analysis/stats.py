r"""Inferencia para comparar sistemas en el benchmark.

- IC bootstrap percentil para la exactitud.
- McNemar exacto (pareado): sobre las preguntas discordantes, b = A acierta y B falla,
  c = A falla y B acierta. Bajo H0, b ~ Binomial(b+c, ½). p = 2·min(F(b), 1−F(b−1)).
- Tamaño de efecto: diferencia de exactitudes con IC bootstrap pareado.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * min(binom.cdf(b, n, 0.5), binom.sf(b - 1, n, 0.5))))
    return {"a_only": b, "b_only": c, "p_value": p}


def paired_diff_ci(correct_a, correct_b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
