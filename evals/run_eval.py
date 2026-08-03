"""Golden-question eval harness for the Self-RAG graph.

Runs every question in golden_questions.jsonl through the real graph (live LLM
and embedding calls) and checks:
  - retrieval routing matches expectation (when specified)
  - the answer contains the expected keyword(s)
  - the graph's own grounding/usefulness self-assessment looks sane

This is a live check against real APIs, not a unit test - run it manually
before a release or a demo, not on every commit:

    python evals/run_eval.py
    python evals/run_eval.py --questions evals/golden_questions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# LLM output can contain non-ASCII characters (curly quotes, non-breaking
# hyphens, etc.) that crash `print` on Windows' default console codepage.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from config.settings import settings  # noqa: E402
from src.self_rag.graph.builder import build_graph  # noqa: E402

DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parent / "golden_questions.jsonl"

# LLMs routinely use "smart" punctuation (non-breaking hyphen, en/em dash, curly
# quotes) that looks identical to a human but defeats a plain substring match
# against an ASCII keyword like "multi-factor". Fold to ASCII before comparing.
_PUNCT_FOLD = str.maketrans(
    {
        "‐": "-", "‑": "-", "‒": "-", "–": "-",
        "—": "-", "―": "-",
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "…": "...",
    }
)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_PUNCT_FOLD)


@dataclass
class CaseResult:
    question: str
    category: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    answer: str = ""
    grounding: str | None = None
    is_useful: bool | None = None
    need_retrieval: bool | None = None


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


def _load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


async def _run_case(graph, case: dict) -> CaseResult:
    question = case["question"]
    category = case.get("category", "uncategorized")
    reasons: list[str] = []

    result = await graph.ainvoke(
        _initial_state(question),
        config={"recursion_limit": settings.recursion_limit},
    )
    answer = (result.get("answer") or "").strip()
    grounding = result.get("grounding") or None
    is_useful = result.get("is_useful")
    need_retrieval = result.get("need_retrieval")

    if "expect_retrieval" in case and need_retrieval != case["expect_retrieval"]:
        reasons.append(
            f"expected need_retrieval={case['expect_retrieval']}, got {need_retrieval}"
        )

    if not answer:
        reasons.append("answer was empty")

    if case.get("expect_answer_declines"):
        decline_markers = ("could not find", "don't have", "do not have", "not covered", "not available")
        if not any(m in answer.lower() for m in decline_markers):
            reasons.append("expected the answer to decline (not covered by docs)")

    expected_keywords = case.get("expected_keywords") or []
    if expected_keywords:
        mode = case.get("keyword_mode", "all")
        folded_answer = _fold(answer).lower()
        hits = [kw for kw in expected_keywords if _fold(kw).lower() in folded_answer]
        if mode == "any" and not hits:
            reasons.append(f"expected any of {expected_keywords} in the answer, found none")
        elif mode == "all" and len(hits) < len(expected_keywords):
            missing = [kw for kw in expected_keywords if kw not in hits]
            reasons.append(f"missing expected keyword(s): {missing}")

    if need_retrieval and expected_keywords:
        if grounding not in ("fully_supported", "partially_supported"):
            reasons.append(f"grounding was '{grounding}', expected a supported grade")
        if is_useful is not True:
            reasons.append(f"is_useful was {is_useful}, expected True")

    return CaseResult(
        question=question,
        category=category,
        passed=not reasons,
        reasons=reasons,
        answer=answer,
        grounding=grounding,
        is_useful=is_useful,
        need_retrieval=need_retrieval,
    )


async def _run_case_resilient(graph, case: dict) -> CaseResult:
    """A live LLM call (rate limit, transient network blip) failing shouldn't
    take down the whole eval batch - retry once, then report it as a failure
    with the real cause instead of crashing the harness."""
    for attempt in (1, 2):
        try:
            return await _run_case(graph, case)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            if attempt == 2:
                return CaseResult(
                    question=case["question"],
                    category=case.get("category", "uncategorized"),
                    passed=False,
                    reasons=[f"graph invocation raised: {exc!r}"],
                )
            print(f"        (retrying after error: {exc!r})")
            await asyncio.sleep(5)


async def run(questions_path: Path) -> list[CaseResult]:
    graph = build_graph()
    cases = _load_cases(questions_path)
    results = []
    for case in cases:
        result = await _run_case_resilient(graph, case)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] ({result.category}) {result.question}")
        if not result.passed:
            for reason in result.reasons:
                print(f"        - {reason}")
            print(f"        answer: {result.answer[:200]!r}")
        await asyncio.sleep(1)  # spread out calls against the Groq TPM budget
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to a golden_questions.jsonl file",
    )
    args = parser.parse_args()

    results = asyncio.run(run(args.questions))

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{total} passed ({passed / total:.0%})" if total else "no cases found")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
