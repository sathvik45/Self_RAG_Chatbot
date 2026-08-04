---
title: Self-RAG PDF Chatbot
emoji: 📄
colorFrom: teal
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Self-RAG PDF Chatbot

A self-reflective RAG chatbot that answers questions from a set of company PDF documents, built with [LangGraph](https://github.com/langchain-ai/langgraph) and served via FastAPI.

Unlike a naive "retrieve then stuff into a prompt" RAG pipeline, the graph reasons about its own answer before returning it:

- **Query rewriting** — a follow-up question is contextualized against the conversation history into a standalone query.
- **Retrieval routing** — skips retrieval entirely for greetings/small talk, so the model doesn't waste a search on "hey, how are you?".
- **Per-chunk relevance grading** — each retrieved chunk is graded for relevance before being used, rather than trusting the top-k blindly.
- **Groundedness verification** — the generated answer is checked against the retrieved context and graded `fully_supported` / `partially_supported` / `no_support`.
- **Revise/retry loop** — an ungrounded answer triggers a rewrite; a query that isn't producing a useful answer triggers a query rewrite and a fresh retrieval pass, up to a configurable number of attempts.

The frontend streams every one of these steps live over SSE, so you can watch the reasoning trace as it happens rather than just waiting for a final answer.

## Running locally

**Docker** (matches how this is actually deployed):

```bash
docker compose up --build
```

**Direct Python** (faster iteration while developing):

```bash
pip install -r requirements-dev.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Either way, open `http://localhost:8000`. Both need a `.env` file (see `.env.example`) with at least `GROQ_API_KEY` set.

## Testing

```bash
pytest tests/                 # unit + mocked integration tests, no live API calls
python evals/run_eval.py      # live golden-question eval against the real graph
```

## Stack

FastAPI · LangGraph · LangChain · Groq (LLM) · Hugging Face `sentence-transformers` (embeddings) · FAISS (vector store)

## Known limitation

The FAISS index is a single file on disk with no cross-process locking, so this app is **not safe to run as multiple replicas** against the same index directory — deploy it as a single instance.
