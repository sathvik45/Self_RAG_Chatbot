# from config.settings import settings
# from src.self_rag.graph.builder import get_graph

# app = get_graph()

# initial_state = {
#     "question": "Describe NexaAI's company culture.",
#     "retrieval_query": "",
#     "rewrite_tries": 0,
#     "docs": [],
#     "relevant_docs": [],
#     "context": "",
#     "answer": "",
#     "grounding": "",
#     "evidence": [],
#     "revise_tries": 0,
#     "is_useful": False,
#     "use_reason": "",
# }

# result = app.invoke(
#     initial_state,
#     config={"recursion_limit": settings.recursion_limit},
# )

# print("\n===== RAG EXECUTION RESULT =====\n")

# print("Question:", initial_state.get("question"))
# print("Need Retrieval:", result.get("need_retrieval"))
# print("Rewrite tries (retrieval):", result.get("rewrite_tries", 0))
# print("Support revise tries:", result.get("revise_tries", 0))

# print("\nRetrieval:")
# print("  Total retrieved docs:", len(result.get("docs", []) or []))
# print("  Relevant docs:", len(result.get("relevant_docs", []) or []))

# print("\nVerification (grounding):")
# print("  grade:", result.get("grounding"))
# print("  evidence:", result.get("evidence", []))

# print("\nUsefulness:")
# print("  is_useful:", result.get("is_useful"))
# print("  reason:", result.get("use_reason", ""))

# print("\nFinal Answer:")
# print(result.get("answer"))

# print("\n===============================\n")

import asyncio

from config.settings import settings
from src.self_rag.graph.builder import get_graph

app = get_graph()

initial_state = {
    "question": "what is the company that is been taked here?.",
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


async def main():
    result = await app.ainvoke(
        initial_state,
        config={"recursion_limit": settings.recursion_limit},
    )

    print("\n===== FINAL ANSWER =====\n")
    print(result["answer"])
    print("\n===== RAG EXECUTION RESULT =====\n")

    print("Question:", initial_state.get("question"))
    print("Need Retrieval:", result.get("need_retrieval"))
    print("Rewrite tries (retrieval):", result.get("rewrite_tries", 0))
    print("Support revise tries:", result.get("revise_tries", 0))

    print("\nRetrieval:")
    print("  Total retrieved docs:", len(result.get("docs", []) or []))
    print("  Relevant docs:", len(result.get("relevant_docs", []) or []))

    print("\nVerification (grounding):")
    print("  grade:", result.get("grounding"))
    print("  evidence:", result.get("evidence", []))

    print("\nUsefulness:")
    print("  is_useful:", result.get("is_useful"))
    print("  reason:", result.get("use_reason", ""))

    print("\nFinal Answer:")
    print(result.get("answer"))

    print("\n===============================\n")


if __name__ == "__main__":
    asyncio.run(main())