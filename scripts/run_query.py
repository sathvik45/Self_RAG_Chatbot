"""Run a single Self-RAG query end-to-end (build index if needed)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "self_rag"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import settings  # noqa: E402
from graph.builder import get_graph  # noqa: E402
from retrieval.vector_store import ensure_index_on_startup, index_exists  # noqa: E402


def _initial_state(question: str) -> dict:
    return {
        "question": question,
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


async def _invoke(question: str) -> dict:
    ensure_index_on_startup()
    if not index_exists():
        raise RuntimeError(
            f"No FAISS index at {settings.index_dir} and no PDFs under {settings.data_dir}. "
            "Add PDFs and retry."
        )

    graph = get_graph()
    return await graph.ainvoke(
        _initial_state(question),
        config={"recursion_limit": settings.recursion_limit},
    )


def _print_result(question: str, result: dict) -> None:
    print("\n===== RAG EXECUTION RESULT =====\n")
    print("Question:", question)
    print("Need Retrieval:", result.get("need_retrieval"))
    print("Rewrite tries:", result.get("rewrite_tries", 0))
    print("Revise tries:", result.get("revise_tries", 0))
    print("\nRetrieval:")
    print("  Total docs:", len(result.get("docs", []) or []))
    print("  Relevant docs:", len(result.get("relevant_docs", []) or []))
    print("\nGrounding:", result.get("grounding"))
    print("Useful:", result.get("is_useful"), "-", result.get("use_reason", ""))
    print("\nFinal Answer:\n", result.get("answer"))
    print("\n===============================\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Self-RAG query")
    parser.add_argument(
        "question",
        nargs="?",
        default="What is the company's code of ethics about?",
        help="Question to ask",
    )
    args = parser.parse_args()

    if not settings.groq_api_key and not __import__("os").getenv("GROQ_API_KEY"):
        print(
            "Missing GROQ_API_KEY. Create a .env file in the project root:\n"
            "  GROQ_API_KEY=your_key_here",
            file=sys.stderr,
        )
        return 1

    try:
        result = asyncio.run(_invoke(args.question))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_result(args.question, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
