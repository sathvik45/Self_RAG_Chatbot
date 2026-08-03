"""Smoke tests for the Self-RAG graph (no live API calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.self_rag.graph.builder import build_graph
from src.self_rag.graph.schemas import (
    GroundingDecision,
    RelevanceDecision,
    RouteDecision,
    usefulnessDecision,
)
from src.self_rag.retrieval.vector_store import index_exists


@pytest.fixture
def graph():
    return build_graph()


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


def test_graph_has_all_nodes(graph):
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__",
        "__end__",
        "contextualize",
        "decide_retrieval",
        "generate_direct",
        "retrieve",
        "grade_relevance",
        "generate_answer",
        "no_relevant_docs",
        "verify_grounding",
        "revise_answer",
        "check_usefulness",
        "rewrite_query",
        "no_answer_found",
    }
    assert expected.issubset(nodes)


def test_index_built():
    assert index_exists(), (
        "Run index build first: "
        "python -c \"from src.self_rag.ingestion.build_index import build_index; build_index()\""
    )


@pytest.mark.asyncio
async def test_retrieval_path_mocked(graph):
    fake_doc = MagicMock()
    fake_doc.page_content = "Employees must disclose conflicts of interest promptly."
    fake_doc.metadata = {"source": "code-of-ethics.pdf", "page": 0}

    mock_llm_out = MagicMock()
    mock_llm_out.content = "Employees must disclose conflicts of interest."

    with (
        patch("src.self_rag.graph.nodes.get_retriever") as mock_retriever,
        patch("src.self_rag.graph.nodes.get_llm") as mock_get_llm,
        patch("src.self_rag.graph.nodes.get_grader") as mock_grader,
    ):
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[fake_doc])
        mock_retriever.return_value = retriever

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=mock_llm_out)
        mock_get_llm.return_value = llm

        def structured_output(schema, method=None):
            grader = MagicMock()

            async def invoke(_messages):
                if schema is RouteDecision:
                    return RouteDecision(should_retrieve=True)
                if schema is RelevanceDecision:
                    return RelevanceDecision(is_relevant=True)
                if schema is GroundingDecision:
                    return GroundingDecision(
                        grade="fully_supported",
                        evidence=["Employees must disclose conflicts of interest promptly."],
                    )
                if schema is usefulnessDecision:
                    return usefulnessDecision(is_useful=True, reason="Direct answer")
                raise ValueError(f"unexpected schema {schema}")

            grader.ainvoke = invoke
            grader.abatch = AsyncMock(
                side_effect=lambda inputs: [
                    RelevanceDecision(is_relevant=True) for _ in inputs
                ]
            )
            return grader

        mock_grader.return_value.with_structured_output.side_effect = structured_output

        result = await graph.ainvoke(
            _initial_state("What does the code of ethics say about conflicts of interest?"),
            config={"recursion_limit": 50},
        )

    assert result.get("need_retrieval") is True
    assert result.get("is_useful") is True
    assert result.get("answer")


@pytest.mark.asyncio
async def test_no_relevant_docs_path_mocked(graph):
    """When nothing retrieved is relevant, the graph should short-circuit to the
    not-found answer instead of calling the answer LLM."""
    fake_doc = MagicMock()
    fake_doc.page_content = "Unrelated content about the weather."
    fake_doc.metadata = {"source": "gullivers-travels.pdf", "page": 3}

    with (
        patch("src.self_rag.graph.nodes.get_retriever") as mock_retriever,
        patch("src.self_rag.graph.nodes.get_grader") as mock_grader,
    ):
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[fake_doc])
        mock_retriever.return_value = retriever

        def structured_output(schema, method=None):
            grader = MagicMock()

            async def invoke(_messages):
                if schema is RouteDecision:
                    return RouteDecision(should_retrieve=True)
                raise ValueError(f"unexpected schema {schema}")

            grader.ainvoke = invoke
            grader.abatch = AsyncMock(
                side_effect=lambda inputs: [
                    RelevanceDecision(is_relevant=False) for _ in inputs
                ]
            )
            return grader

        mock_grader.return_value.with_structured_output.side_effect = structured_output

        result = await graph.ainvoke(
            _initial_state("What is the capital of France?"),
            config={"recursion_limit": 50},
        )

    assert result.get("relevant_docs") == []
    assert result.get("answer") == (
        "I could not find anything relevant in the docs I have access to."
    )
