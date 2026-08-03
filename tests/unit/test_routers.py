from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as main_module


@pytest.fixture
def client():
    with patch.object(main_module, "ensure_index_on_startup"):
        with TestClient(main_module.app) as c:
            yield c


def test_health_reports_index_state(client):
    with (
        patch.object(main_module, "index_exists", return_value=True),
        patch.object(main_module, "documents_indexed", return_value=42),
    ):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OK"
    assert body["index_ready"] is True
    assert body["documents_indexed"] == 42


def test_chat_returns_answer_from_dependencies(client):
    fake_result = {
        "answer": "The refund window is 30 days.",
        "sources": [],
        "session_id": "sess-1",
        "retrieved": True,
        "grounding": "fully_supported",
        "trace": ["decide_retrieval", "retrieve"],
    }
    with patch("api.dependencies.answer_once", new=AsyncMock(return_value=fake_result)):
        resp = client.post("/chat", json={"question": "What is the refund policy?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "The refund window is 30 days."


def test_chat_returns_500_on_failure(client):
    with patch(
        "api.dependencies.answer_once",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        resp = client.post("/chat", json={"question": "What is the refund policy?"})
    assert resp.status_code == 500


def test_reset_clears_the_given_session(client):
    """Regression test: `/reset` used to call SessionStore.reset(), a method
    that didn't exist, so this endpoint 500'd on every real request."""
    with patch.object(main_module.store, "reset") as mock_reset:
        resp = client.post("/reset", json={"question": "x", "session_id": "abc"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "cleared"}
    mock_reset.assert_called_once_with("abc")


def test_reset_without_session_id_is_a_noop(client):
    with patch.object(main_module.store, "reset") as mock_reset:
        resp = client.post("/reset", json={"question": "x"})
    assert resp.status_code == 200
    mock_reset.assert_not_called()


def test_upload_rejects_non_pdf(client):
    resp = client.post(
        "/upload",
        files={"files": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_indexes_saved_pdfs(client, tmp_path):
    with (
        patch.object(main_module.settings, "data_dir", tmp_path),
        patch.object(main_module, "add_to_index", return_value=123) as mock_add,
    ):
        resp = client.post(
            "/upload",
            files={"files": ("policy.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] == ["policy.pdf"]
    assert body["chunks_indexed"] == 123
    assert (tmp_path / "policy.pdf").exists()
    mock_add.assert_called_once()
