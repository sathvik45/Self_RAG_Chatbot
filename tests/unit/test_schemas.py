import pytest
from pydantic import ValidationError

from src.self_rag.graph.schemas import (
    ChatRequest,
    ChatResponse,
    GroundingDecision,
    RelevanceDecision,
    RouteDecision,
    Source,
)


def test_chat_request_requires_nonempty_question():
    with pytest.raises(ValidationError):
        ChatRequest(question="")


def test_chat_request_rejects_overlong_question():
    with pytest.raises(ValidationError):
        ChatRequest(question="x" * 4001)


def test_chat_request_session_id_optional():
    req = ChatRequest(question="What is the refund policy?")
    assert req.session_id is None


def test_chat_response_defaults():
    resp = ChatResponse(answer="hello", session_id="abc123", retrieved=False)
    assert resp.sources == []
    assert resp.trace == []
    assert resp.grounding is None


def test_source_defaults():
    src = Source(filename="doc.pdf")
    assert src.page is None
    assert src.snippet == ""


def test_grounding_decision_rejects_invalid_grade():
    with pytest.raises(ValidationError):
        GroundingDecision(grade="mostly_true")


def test_grounding_decision_accepts_valid_grades():
    for grade in ("fully_supported", "partially_supported", "no_support"):
        assert GroundingDecision(grade=grade).grade == grade


def test_route_decision_requires_bool():
    with pytest.raises(ValidationError):
        RouteDecision(should_retrieve="maybe")


def test_relevance_decision_roundtrip():
    assert RelevanceDecision(is_relevant=True).is_relevant is True
