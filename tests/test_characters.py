from rag_mb import characters
from rag_mb.analysis.pipeline import ALIASES, BOOK_MENTIONS


def test_yaml_is_single_source_of_truth():
    assert ALIASES["lotr"]["STRIDER"] == "ARAGORN"
    assert "Mithrandir" in BOOK_MENTIONS["lotr"]["GANDALF"]
    assert set(BOOK_MENTIONS) >= {"lotr", "dune", "hp", "narnia"}


def test_mention_regex_counts_once_and_respects_case():
    p = characters.mention_regex(["Feyd-Rautha", "Feyd"])
    assert len(p.findall("Feyd-Rautha smiled. Feyd turned. FEYD-RAUTHA enters.")) == 3
    s = characters.mention_regex(["Sam"])
    assert len(s.findall("'Sam!' said Frodo. Sam's pack. same sesame SAM")) == 3   # no 'same' / 'sesame'


def test_said_the_attribution():
    from rag_mb.analysis.quotes import _attribution_patterns
    assert _attribution_patterns("narnia")["WHITE WITCH"].search("said the Witch")
