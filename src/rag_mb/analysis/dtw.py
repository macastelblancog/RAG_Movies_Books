r"""Dynamic Time Warping: alineación MONÓTONA capítulos → escenas por programación dinámica.

    D(i,j) = C(i,j) + min{ D(i−1,j), D(i,j−1), D(i−1,j−1) },   D(0,0)=0

Contraste con OT: DTW impone orden (no permite reordenamientos) y cobertura total; OT permite
ambos. La diferencia entre los dos alineamientos localiza los reordenamientos de la adaptación.
"""
from __future__ import annotations

import numpy as np


def dtw(C: np.ndarray, band: int | None = None) -> tuple[float, list[tuple[int, int]]]:
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = 1, m
        if band is not None:
            c = int(round(i * m / n))
            lo, hi = max(1, c - band), min(m, c + band)
        for j in range(lo, hi + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtracking
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return float(D[n, m]), path[::-1]


def path_to_mapping(path: list[tuple[int, int]], n: int) -> np.ndarray:
    """Para cada capítulo i, posición media (normalizada) de las escenas alineadas."""
    m = max(j for _, j in path) + 1
    acc = [[] for _ in range(n)]
    for i, j in path:
        acc[i].append(j / max(m - 1, 1))
    return np.array([np.mean(x) if x else np.nan for x in acc])
