from src.self_rag.graph import edges


def test_route_entry_goes_to_contextualize_with_history():
    assert edges.route_entry({"history": ["prior turn"]}) == "contextualize"


def test_route_entry_skips_contextualize_without_history():
    assert edges.route_entry({"history": []}) == "decide_retrieval"
    assert edges.route_entry({}) == "decide_retrieval"


def test_route_after_decide_retrieves_when_needed():
    assert edges.route_after_decide({"need_retrieval": True}) == "retrieve"


def test_route_after_decide_answers_directly_otherwise():
    assert edges.route_after_decide({"need_retrieval": False}) == "generate_direct"


def test_route_after_relevance_uses_relevant_docs():
    assert edges.route_after_relevance({"relevant_docs": ["doc"]}) == "generate_answer"
    assert edges.route_after_relevance({"relevant_docs": []}) == "no_relevant_docs"


def test_route_after_grounding_passes_when_supported():
    for grade in ("fully_supported", "partially_supported"):
        assert (
            edges.route_after_grounding({"grounding": grade, "revise_tries": 0})
            == "check_usefulness"
        )


def test_route_after_grounding_revises_on_no_support():
    assert (
        edges.route_after_grounding({"grounding": "no_support", "revise_tries": 0})
        == "revise_answer"
    )


def test_route_after_grounding_gives_up_after_max_revise_tries(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "max_revise_tries", 2)
    state = {"grounding": "no_support", "revise_tries": 2}
    assert edges.route_after_grounding(state) == "check_usefulness"


def test_route_after_usefulness_done_when_useful():
    assert edges.route_after_usefulness({"is_useful": True}) == "done"


def test_route_after_usefulness_rewrites_when_not_useful():
    assert (
        edges.route_after_usefulness({"is_useful": False, "rewrite_tries": 0})
        == "rewrite_query"
    )


def test_route_after_usefulness_gives_up_after_max_rewrite_tries(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "max_rewrite_tries", 2)
    state = {"is_useful": False, "rewrite_tries": 2}
    assert edges.route_after_usefulness(state) == "no_answer_found"
