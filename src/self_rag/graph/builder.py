from functools import lru_cache

from langgraph.graph import StateGraph, START, END

from src.self_rag.graph import edges, nodes
from src.self_rag.graph.state import State


def build_graph():
    g = StateGraph(State)

    g.add_node("contextualize", nodes.contextualize)
    g.add_node("decide_retrieval", nodes.decide_retrieval)
    g.add_node("generate_direct", nodes.generate_direct)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_relevance", nodes.grade_relevance)
    g.add_node("generate_answer", nodes.generate_answer)
    g.add_node("no_relevant_docs", nodes.no_relevant_docs)
    g.add_node("verify_grounding", nodes.verify_grounding)
    g.add_node("revise_answer", nodes.revise_answer)
    g.add_node("check_usefulness", nodes.check_usefulness)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("no_answer_found", nodes.no_answer_found)

    g.add_conditional_edges(
        START,
        edges.route_entry,
        {"contextualize": "contextualize", "decide_retrieval": "decide_retrieval"},
    )
    g.add_edge("contextualize", "decide_retrieval")

    g.add_conditional_edges(
        "decide_retrieval",
        edges.route_after_decide,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "grade_relevance")
    g.add_conditional_edges(
        "grade_relevance",
        edges.route_after_relevance,
        {"generate_answer": "generate_answer", "no_relevant_docs": "no_relevant_docs"},
    )
    g.add_edge("generate_answer", "verify_grounding")
    g.add_edge("no_relevant_docs", END)

    g.add_conditional_edges(
        "verify_grounding",
        edges.route_after_grounding,
        {"check_usefulness": "check_usefulness", "revise_answer": "revise_answer"},
    )
    g.add_edge("revise_answer", "verify_grounding")

    g.add_conditional_edges(
        "check_usefulness",
        edges.route_after_usefulness,
        {
            "done": END,
            "rewrite_query": "rewrite_query",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("no_answer_found", END)

    return g.compile()


@lru_cache
def get_graph():
    return build_graph()
