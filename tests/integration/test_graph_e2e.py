"""Live end-to-end checks against the real Groq API and the real FAISS index.

These are deliberately excluded from the default CI run (no live API keys there) —
run them manually before a release with:

    pytest tests/integration/test_graph_e2e.py -m e2e
"""

from __future__ import annotations

import pytest

from config.settings import settings
from src.self_rag.graph.builder import build_graph
from src.self_rag.retrieval.vector_store import index_exists

pytestmark = pytest.mark.e2e

requires_live_creds = pytest.mark.skipif(
    not settings.groq_api_key or not index_exists(),
    reason="needs a real GROQ_API_KEY and a built FAISS index",
)


def _initial_state(question: str) -> dict:
    return {
        "question": question,
        "original_question": question,
        "history": [],
        "retrieval_query": "",
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "grounding": "",
        "evidence": [],
        "revise_tries": 0,
        "is_useful": False,
        "use_reason": "",
    }


@requires_live_creds
@pytest.mark.asyncio
async def test_answers_a_question_grounded_in_the_indexed_docs():
    graph = build_graph()
    result = await graph.ainvoke(
        _initial_state("What are the password requirements in the information security policy?"),
        config={"recursion_limit": settings.recursion_limit},
    )
    assert result.get("need_retrieval") is True
    assert result.get("answer")
    assert result.get("grounding") in ("fully_supported", "partially_supported")


@requires_live_creds
@pytest.mark.asyncio
async def test_skips_retrieval_for_smalltalk():
    graph = build_graph()
    result = await graph.ainvoke(
        _initial_state("Hey, how are you?"),
        config={"recursion_limit": settings.recursion_limit},
    )
    assert result.get("need_retrieval") is False
    assert result.get("answer")
