from typing import Literal

from config.settings import settings
from src.self_rag.graph.state import State


def route_entry(state: State) -> Literal["contextualize", "decide_retrieval"]:
    return "contextualize" if state.get("history") else "decide_retrieval"


def route_after_decide(state: State) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state.get("need_retrieval") else "generate_direct"


def route_after_relevance(state: State) -> Literal["generate_answer", "no_relevant_docs"]:
    return "generate_answer" if state.get("relevant_docs") else "no_relevant_docs"


def route_after_grounding(state: State) -> Literal["check_usefulness", "revise_answer"]:
    grade = state.get("grounding")
    if grade in ("fully_supported", "partially_supported"):
        return "check_usefulness"
    if state.get("revise_tries", 0) >= settings.max_revise_tries:
        return "check_usefulness"
    return "revise_answer"


def route_after_usefulness(
    state: State,
) -> Literal["done", "rewrite_query", "no_answer_found"]:
    if state.get("is_useful"):
        return "done"
    if state.get("rewrite_tries", 0) >= settings.max_rewrite_tries:
        return "no_answer_found"
    return "rewrite_query"
