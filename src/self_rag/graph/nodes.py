from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from config.logging_config import get_logger
from config.settings import settings
from src.self_rag.graph import prompts
from src.self_rag.graph.schemas import (
    GroundingDecision,
    RelevanceDecision,
    RewrittenQuery,
    RouteDecision,
    StandlaoneQuestion,
    usefulnessDecision,
)
from src.self_rag.graph.state import State
from src.self_rag.llm.client import get_grader, get_llm
from src.self_rag.retrieval.retriever import get_retriever

log = get_logger(__name__)

NOT_FOUND = "I could not find anything relevant in the docs I have access to."
GIVE_UP = (
    "I could not find a well supported answer after a few attempts. "
    "It may not be covered in the docs or the question might need to be more specific."
)


@lru_cache
def _contextualizer():
    return get_grader().with_structured_output(StandlaoneQuestion, method="json_schema")


@lru_cache
def _router():
    return get_grader().with_structured_output(RouteDecision, method="json_schema")


@lru_cache
def _relevance():
    return get_grader().with_structured_output(RelevanceDecision, method="json_schema")


@lru_cache
def _grounding():
    return get_grader().with_structured_output(GroundingDecision, method="json_schema")


@lru_cache
def _usefulness():
    return get_grader().with_structured_output(usefulnessDecision, method="json_schema")


@lru_cache
def _rewrite():
    return get_grader().with_structured_output(RewrittenQuery, method="json_schema")


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "document")

        page = meta.get("page")
        tag = f"[{src}" + (f" p.{page + 1})]" if isinstance(page, int) else "]")
        blocks.append(f"{tag}\n{d.page_content.strip()}")
    return "\n\n----\n\n".join(blocks).strip()


async def contextualize(state: State) -> dict:
    history = state.get("history") or []
    if not history:
        return {}
    out: StandlaoneQuestion = await _contextualizer().ainvoke(
        prompts.CONTEXTUALIZE_PROMPT.format_messages(
            history=history, question=state["question"]
        )
    )
    log.info("Contextualised -> %s", out.question)
    return {"question": out.question}


async def decide_retrieval(state: State) -> dict:
    out: RouteDecision = await _router().ainvoke(
        prompts.ROUTE_PROMPT.format_messages(question=state["question"])
    )
    log.info("need_retrieval = %s", out.should_retrieve)
    return {"need_retrieval": out.should_retrieve}


async def generate_direct(state: State) -> dict:
    chain = prompts.DIRECT_ANSWER_PROMPT | get_llm()
    out = await chain.ainvoke(
        {"question": state["question"], "history": state.get("history") or []}
    )
    return {"answer": out.content}


async def retrieve(state: State) -> dict:
    retriever = get_retriever()
    if retriever is None:
        log.warning("no retriever is available.")
        return {"docs": []}
    query = state.get("retrieval_query") or state["question"]
    docs = await retriever.ainvoke(query)
    log.info("retrieved %d chunks for: %s", len(docs), query)
    return {"docs": docs}


async def grade_relevance(state: State) -> dict:
    docs = state.get("docs") or []
    if not docs:
        return {"relevant_docs": []}
    inputs = [
        prompts.RELEVANCE_PROMPT.format_messages(
            question=state["question"], document=d.page_content
        )
        for d in docs
    ]
    try:
        results = await _relevance().abatch(inputs)
        relevant = [d for d, r in zip(docs, results) if r.is_relevant]
    except Exception:
        log.exception("Relevance grading failed, keeping all retrieved chunks.")
        relevant = docs
    log.info("Relevant: %d/%d", len(relevant), len(docs))
    return {"relevant_docs": relevant}


async def generate_answer(state: State) -> dict:
    context = _format_context(state.get("relevant_docs") or [])
    if not context:
        return {"answer": NOT_FOUND, "context": ""}
    chain = prompts.RAG_ANSWER_PROMPT | get_llm()
    out = await chain.ainvoke(
        {
            "question": state["question"],
            "context": context,
            "history": state.get("history") or [],
        }
    )
    return {"answer": out.content, "context": context}


async def no_relevant_docs(state: State) -> dict:
    return {"answer": NOT_FOUND, "context": ""}


async def verify_grounding(state: State) -> dict:
    context = state.get("context") or ""
    if not context:
        return {"grounding": "fully_supported", "evidence": []}
    out: GroundingDecision = await _grounding().ainvoke(
        prompts.GROUNDING_PROMPT.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=context,
        )
    )
    log.info("grounding=%s", out.grade)
    return {"grounding": out.grade, "evidence": out.evidence}


async def revise_answer(state: State) -> dict:
    chain = prompts.REVISE_PROMPT | get_llm()
    out = await chain.ainvoke(
        {
            "question": state["question"],
            "answer": state.get("answer", ""),
            "context": state.get("context", ""),
        }
    )
    return {"answer": out.content, "revise_tries": state.get("revise_tries", 0) + 1}


async def check_usefulness(state: State) -> dict:
    out: usefulnessDecision = await _usefulness().ainvoke(
        prompts.USEFULNESS_PROMPT.format_messages(
            question=state["question"], answer=state.get("answer", "")
        )
    )
    log.info("is_useful=%s (%s)", out.is_useful, out.reason)
    return {"is_useful": out.is_useful, "use_reason": out.reason}


async def rewrite_query(state: State) -> dict:
    out: RewrittenQuery = await _rewrite().ainvoke(
        prompts.REWRITE_QUERY_PROMPT.format_messages(
            question=state["question"],
            previous_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    log.info("rewritten query -> %s", out.query)
    return {
        "retrieval_query": out.query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "grounding": "",
        "evidence": [],
        "revise_tries": 0,
    }


async def no_answer_found(state: State) -> dict:
    return {"answer": GIVE_UP}
