from rag_mb.rag.llm import split_evidence


def test_split_evidence_scores_only_the_answer():
    ev, ans = split_evidence('Answer: Barty Crouch\'s house-elf is Winky [2].\nEvidence: "Winky was crying." [2]')
    assert ev.startswith('"Winky') and ans.startswith("Barty") and "crying" not in ans
    ev, ans = split_evidence('Evidence: "Winky was crying." [2]\nAnswer: It is Winky [2].')
    assert ans == "It is Winky [2]."
    assert split_evidence("Answer: Los pasajes no lo dicen.\nEvidencia: none")[1] == "Los pasajes no lo dicen."
    assert split_evidence("Plain answer without labels.") == ("", "Plain answer without labels.")
    # salida truncada: solo evidencia copiada, sin respuesta → respuesta vacía (cuenta como fallo)
    assert split_evidence("Evidence: In a chair sat a woman, Goldberry...")[1] == ""


def test_quantitative_tool_matches_gold_on_question_bank():
    from rag_mb.eval.metrics import score_answer
    from rag_mb.eval.runner import load_questions
    from rag_mb.rag.qa import tool_answer
    for saga in ("lotr", "hp", "dune", "narnia"):
        for q in load_questions(saga):
            if q.get("gold_fn"):
                a = tool_answer(q["question"])
                assert a is not None and a.tool, q["id"]
                assert score_answer(a.text, q["must"], None), (q["id"], a.text, q["must"])
    assert tool_answer("¿Cuántos capítulos tiene el libro El león, la bruja y el ropero?").text.startswith("Según")
    assert tool_answer("How did Boromir die?") is None


def test_abstention_metric_is_not_fooled_by_incidental_negations():
    from rag_mb.eval.metrics import score_answer
    ok = ["The passages do not say.", "Tom Bombadil does not appear in the films.",
          "False premise: the Yule Ball is in Goblet of Fire.", "Los pasajes no lo dicen.",
          "Saruman never comes to the Shire in the film."]
    bad = ["He tells Feyd-Rautha that the slave is not drugged and that he tried to kill him.",
           "Edmund defeats her using his archery skills.", "Filch's cat has been murdered."]
    assert all(score_answer(a, None, "abstain") for a in ok)
    assert not any(score_answer(a, None, "abstain") for a in bad)
