from rag_mb.rag.router import detect_saga, route


def test_saga_from_character_and_places_es_en():
    assert detect_saga("¿Cómo muere Boromir?")[0] == "lotr"
    assert detect_saga("In the film, what happens at Helm's Deep?")[0] == "lotr"
    assert detect_saga("¿Qué hacen los gusanos de arena?")[0] == "dune"
    assert detect_saga("¿Qué es la Mesa de Piedra?")[0] == "narnia"


def test_no_signal_means_no_saga():
    assert detect_saga("Who narrates the prologue?")[0] is None


def test_medium_work_and_balance_in_spanish():
    r = route("¿Cómo destruye Harry el último horrocrux en Reliquias de la muerte parte 2?", None)
    assert (r.saga, r.work) == ("hp", "hp7b")
    r = route("¿Quién le da el regalo a Lucy en la película?", None)
    assert (r.saga, r.medium) == ("narnia", "screenplay") and "película" not in r.query
    assert route("¿Qué cambia entre el libro y la película en la muerte de Saruman?", None).balanced


def test_presence_questions_search_both_media():
    r = route("Is the Scouring of the Shire part of the film adaptation?", None)
    assert r.balanced and r.medium is None
    assert route("¿Aparece Tom Bombadil en la película?", None).balanced
    r = route("Does Gandalf die in the film?", None)          # pregunta sobre la película, no de presencia
    assert r.medium == "screenplay" and not r.balanced


def test_dune_title_is_stripped_but_not_other_sagas():
    r = route("In the Dune novel, whose writings are quoted in the epigraphs?", None)
    assert r.saga == "dune" and r.medium == "book" and "Dune" not in r.query
    r = route("Who kills Baron Vladimir Harkonnen in the book, and who kills him in Dune: Part Two?", None)
    assert r.work == "dune_p2" and "Dune" not in r.query and "Harkonnen" in r.query
    assert "sandworm" in route("How does Paul ride a sandworm on Arrakis?", None).query
    assert "Harry Potter" in route("In Harry Potter and the Chamber of Secrets, who opens the chamber?", None).query


def test_prompt_label_marks_film_and_translates_locator():
    from rag_mb.rag.index import Hit
    from rag_mb.schema import Chunk
    mk = lambda medium, loc: Hit(1, 0.0, Chunk("c", "u", "hp", medium, "hp6", 0, loc, "t", []))
    assert mk("dialogue", "Capítulo 23 — Horcruxes").prompt_label.startswith("FILM — dialogue transcript")
    assert "Scene group 23" in mk("dialogue", "Capítulo 23 — Horcruxes").prompt_label
    assert mk("book", "Cap. 23 — Horcruxes").prompt_label.startswith("BOOK") and "Chapter 23" in \
        mk("book", "Cap. 23 — Horcruxes").prompt_label
    assert "Part 2, Chapter 11" in mk("book", "Libro 2, Cap. 11").prompt_label
    assert "Scene 113" in mk("screenplay", "Escena 113 — EXT. ROCKY SPIRE").prompt_label
    assert mk("dialogue", "Capítulo 23").citation.startswith("Harry Potter · Diálogo película")
