r"""Redes de personajes y teoría espectral de grafos.

Grafo ponderado G = (V, E, w): w_uv = nº de unidades (escenas / párrafos) donde u y v co-ocurren.
Laplaciano normalizado:  𝓛 = I − D^{-1/2} W D^{-1/2},  espectro en [0, 2].
  - multiplicidad del 0  = nº de componentes conexas
  - λ₂ (conectividad algebraica, Fiedler): qué tan "cohesionada" está la red
  - vector de Fiedler: bipartición espectral (relajación del corte normalizado, Shi & Malik)
Comparación libro vs película sobre nodos comunes:
  - τ de Kendall entre rankings de centralidad de vector propio
  - distancia espectral ‖λ(𝓛_libro) − λ(𝓛_película)‖₂ (mismo nº de nodos)
"""
from __future__ import annotations

import itertools
import re
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ..schema import Unit


def graph_from_scenes(units: list[Unit], aliases: dict[str, str] | None = None,
                      min_lines: int = 1) -> nx.Graph:
    aliases = aliases or {}
    G = nx.Graph()
    for u in units:
        c = Counter(aliases.get(ln.speaker, ln.speaker) for ln in u.lines)
        chars = sorted(s for s, k in c.items() if k >= min_lines)
        for s in chars:
            G.add_node(s)
        for x, y in itertools.combinations(chars, 2):
            G.add_edge(x, y, weight=G.get_edge_data(x, y, {"weight": 0})["weight"] + 1)
    return G


def graph_from_mentions(units: list[Unit], characters: dict[str, list[str]]) -> nx.Graph:
    """Co-ocurrencia de MENCIONES por párrafo (libro). characters: canónico → [patrones]."""
    pats = {c: re.compile(r"\b(" + "|".join(map(re.escape, ps)) + r")\b") for c, ps in characters.items()}
    G = nx.Graph()
    G.add_nodes_from(characters)
    for u in units:
        for para in u.text.split("\n\n"):
            present = sorted(c for c, p in pats.items() if p.search(para))
            for x, y in itertools.combinations(present, 2):
                G.add_edge(x, y, weight=G.get_edge_data(x, y, {"weight": 0})["weight"] + 1)
    return G


def top_nodes(G: nx.Graph, k: int) -> list[str]:
    return [n for n, _ in sorted(G.degree(weight="weight"), key=lambda t: -t[1])[:k]]


def normalized_laplacian_spectrum(G: nx.Graph, nodes: list[str]) -> np.ndarray:
    H = G.subgraph(nodes)
    L = nx.normalized_laplacian_matrix(H, nodelist=nodes, weight="weight").toarray()
    return np.sort(np.linalg.eigvalsh(L))


def fiedler_partition(G: nx.Graph, nodes: list[str]) -> dict[str, int]:
    H = G.subgraph(nodes)
    L = nx.normalized_laplacian_matrix(H, nodelist=nodes, weight="weight").toarray()
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1]
    return {n: int(f[i] >= 0) for i, n in enumerate(nodes)}


def compare_graphs(Ga: nx.Graph, Gb: nx.Graph, nodes: list[str]) -> dict:
    ea = nx.eigenvector_centrality_numpy(Ga.subgraph(nodes), weight="weight")
    eb = nx.eigenvector_centrality_numpy(Gb.subgraph(nodes), weight="weight")
    xa, xb = [ea.get(n, 0) for n in nodes], [eb.get(n, 0) for n in nodes]
    tau = kendalltau(xa, xb)
    sa, sb = normalized_laplacian_spectrum(Ga, nodes), normalized_laplacian_spectrum(Gb, nodes)
    table = pd.DataFrame({"character": nodes, "eig_book": xa, "eig_film": xb})
    table["delta"] = table.eig_film - table.eig_book
    return {"kendall_tau": float(tau.statistic), "kendall_p": float(tau.pvalue),
            "spectral_distance": float(np.linalg.norm(sa - sb)),
            "fiedler_book": float(sa[1]), "fiedler_film": float(sb[1]),
            "table": table.sort_values("delta")}
