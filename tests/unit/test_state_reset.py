from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.self_rag.graph.nodes import rewrite_query
from src.self_rag.graph.schemas import RewrittenQuery


@pytest.mark.asyncio
async def test_rewrite_query_resets_retrieval_state():
    """rewrite_query kicks off a fresh retrieval pass, so it must clear out
    everything from the previous attempt (stale docs/grounding/evidence) while
    bumping rewrite_tries and carrying the new query forward."""
    state = {
        "question": "What is the refund policy?",
        "retrieval_query": "refund policy",
        "rewrite_tries": 1,
        "docs": [MagicMock()],
        "relevant_docs": [MagicMock()],
        "context": "stale context",
        "grounding": "no_support",
        "evidence": ["stale evidence"],
        "revise_tries": 2,
        "answer": "a previous wrong answer",
    }

    with patch("src.self_rag.graph.nodes.get_grader") as mock_grader:
        grader = MagicMock()
        grader.ainvoke = AsyncMock(
            return_value=RewrittenQuery(query="refund policy eligibility window days")
        )
        mock_grader.return_value.with_structured_output.return_value = grader

        result = await rewrite_query(state)

    assert result["retrieval_query"] == "refund policy eligibility window days"
    assert result["rewrite_tries"] == 2
    assert result["docs"] == []
    assert result["relevant_docs"] == []
    assert result["context"] == ""
    assert result["grounding"] == ""
    assert result["evidence"] == []
    assert result["revise_tries"] == 0
