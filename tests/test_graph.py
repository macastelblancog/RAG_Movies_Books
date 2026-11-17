"""Rama de divergencia del grafo: decisión determinista (sin LLM ni índice)."""
import pytest

from rag_mb.rag.qa import Answer


def _ans(q, balanced, work=None):
    return Answer(q, [], [], "dune", None, work, q, balanced=balanced)


def test_is_divergence_only_when_both_media_named():
    pytest.importorskip("langgraph")
    from rag_mb.rag.graph import is_divergence
    assert is_divergence(_ans("Who kills the Baron in the book, and who kills him in the film?", True))
    assert is_divergence(_ans("Who kills the Baron in the book, and in Dune: Part Two?", True, "dune_p2"))
    assert not is_divergence(_ans("In the book, why does Yueh betray the Atreides?", False))
    assert not is_divergence(_ans("Does Winky appear in the Goblet of Fire film as in the book?", True))
    assert is_divergence(_ans("Are Paul or Jessica descendants of the Harkonnens?", True))   # sin medio
    assert not is_divergence(_ans("In the book, how does Boromir die?", True))   # un solo medio
