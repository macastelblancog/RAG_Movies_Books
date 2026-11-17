"""Pruebas de las piezas matemáticas con casos de solución conocida (no requieren datos)."""
from collections import Counter

import numpy as np

from rag_mb.analysis.dtw import dtw
from rag_mb.analysis.lexical import dirichlet_log_odds, js_divergence, zipf_fit
from rag_mb.analysis.semantic_shift import orthogonal_procrustes
from rag_mb.analysis.stats import mcnemar_exact
from rag_mb.analysis.transport import sinkhorn
from rag_mb.rag.retrievers import BM25, rrf


def test_sinkhorn_marginals_and_identity():
    n = 6
    C = 1 - np.eye(n)                     # costo 0 en la diagonal
    a = b = np.ones(n) / n
    r = sinkhorn(a, b, C, eps=0.01)
    assert np.allclose(r.plan.sum(1), a, atol=1e-6)
    assert np.allclose(r.plan.sum(0), b, atol=1e-6)
    assert np.trace(r.plan) > 0.99       # transporta casi todo por la diagonal
    assert r.order_spearman > 0.99


def test_unbalanced_drops_unmatched_mass():
    # la fila 2 está lejos de todo: el OT no balanceado debe retener poca masa de ella
    C = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    r = sinkhorn(np.ones(3), np.ones(2), C, eps=0.05, tau=0.2)
    assert r.retained[2] < 0.2 * min(r.retained[0], r.retained[1])


def test_dtw_identity_path():
    C = 1 - np.eye(5)
    cost, path = dtw(C)
    assert cost == 0 and path == [(i, i) for i in range(5)]


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 5))
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    R = orthogonal_procrustes(A, A @ Q)
    assert np.allclose(R, Q, atol=1e-8)


def test_js_bounds():
    a, b = Counter({"x": 10}), Counter({"y": 10})
    assert abs(js_divergence(a, b) - 1.0) < 1e-12        # soportes disjuntos ⇒ 1 bit
    assert js_divergence(a, a) == 0


def test_zipf_on_ideal_distribution():
    c = Counter({f"w{r}": int(1e6 / r) for r in range(1, 2000)})
    assert abs(zipf_fit(c).alpha - 1.0) < 0.02


def test_log_odds_sign():
    a = Counter({"ring": 100, "the": 1000})
    b = Counter({"ring": 5, "the": 1000})
    df = dirichlet_log_odds(a, b, min_count=1)
    assert df.iloc[0].word == "ring" and df.iloc[0].z > 1.96


def test_mcnemar():
    a = np.array([1] * 10 + [0] * 0, bool)
    b = np.zeros(10, bool)
    assert mcnemar_exact(a, b)["p_value"] < 0.01


def test_bm25_and_rrf():
    docs = ["frodo carries the ring", "gandalf the grey wizard", "sam cooks rabbit stew"]
    s = BM25().fit(docs).scores("ring frodo")
    assert int(np.argmax(s)) == 0
    fused = rrf([np.array([0, 1, 2]), np.array([0, 2, 1])], 3)
    assert int(np.argmax(fused)) == 0
