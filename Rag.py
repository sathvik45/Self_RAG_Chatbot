# %%
# ============================================================================
# FIXES APPLIED (see inline "# FIX:" comments for exact locations):
#   1. State schema had "retires" instead of "retries" -> every write to
#      "retries" was silently dropped by LangGraph (not a channel in the
#      schema) -> the retry counter never advanced -> the is_sup/revise_answer
#      loop could never exit via the MAX_RETRIES cap -> GraphRecursionError.
#   2. route_after_isuse can return "no_answer_found", but that key wasn't in
#      the conditional-edge mapping -> KeyError: 'no_answer_found' the first
#      time an answer is still "not_useful" after MAX_REWRITE_TRIES rewrites.
#      Added a real no_answer_found node and registered it.
#   3. rewrite_question reset docs/relevant_docs/context but not "retries" ->
#      after the grounding loop maxed out once, it silently became a no-op on
#      every subsequent rewrite pass. Now resets retries/issup/evidence too.
#   4. The is_sup conditional edges mapped "accept_answer" straight to
#      "is_use", bypassing the actual accept_answer node entirely (confirmed
#      via trace: the node never executed). Fixed the mapping and pointed
#      accept_answer -> is_use instead of -> END, so it's a real step you can
#      hook logging/tracing into later.
#   Also: removed unused imports (time, re, TavilySearchResults), moved
#   load_dotenv() after all imports, and raised recursion_limit from 80 to
#   150 - once fix #3 is in place, a worst-case run (never fully_supported,
#   never useful) legitimately needs ~109 steps with MAX_RETRIES=10 and
#   MAX_REWRITE_TRIES=3, which no longer fits under 80.
# ============================================================================

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
docs = (PyPDFLoader("Company_Policies.pdf").load() + PyPDFLoader("Company_Profile.pdf").load() + PyPDFLoader("Product_and_Pricing.pdf").load())
print(len(docs))

# 2) Chunk
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150).split_documents(docs)

print(len(chunks))


embeddings = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-107m-multilingual")
model = ChatGroq(model= "openai/gpt-oss-120b", streaming=True)


vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.save_local("faiss_index_SRAG")

vector_store = FAISS.load_local(
    "faiss_index_SRAG",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

class State(TypedDict):
    question: str

    retrieval_query: str
    rewrite_tries: int

    need_retrieval: bool

    docs: List[Document]
    relevant_docs : List[Document]

    context : str

    issup : Literal['fully_supported', "partially_supported", "no_support"]
    evidence : List[str]

    retries : int  # FIX: was "retires" (typo) - every node in this file already
                    # reads/writes "retries", so the old spelling meant those
                    # writes were silently dropped by LangGraph at runtime.

    isuse: Literal["useful", "not_useful"]
    use_reason: str

    answer: str

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False."
    )

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': boolean}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.\n"
            "- should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.\n"
            "- If unsure, choose True."
        ),
        ("human", "Question: {question}"),
    ]
)
should_retrieve_llm = model.with_structured_output(RetrieveDecision)

def decide_retrieval(state: "State"):
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)


def generate_direct(state: State):
    out = model.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"]
        )
    )
    return {
        "answer": out.content
    }

def retrieve(state: State):
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document helps answer the question, else False."
    )

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "A document is relevant if it contains information useful for answering the question."
        ),
        (
            "human",
            "Question:\n{question}\n\nDocument:\n{document}"
        ),
    ]
)

relevance_llm = model.with_structured_output(RelevanceDecision)

def is_relevant(state: State):

    relevant_docs: List[Document] = []

    for doc in state["docs"]:
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            )
        )

        if decision.is_relevant:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business RAG assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the context does not contain enough information, say:\n"
            "'No relevant document found.'\n"
            "Do not use outside knowledge.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

def generate_from_context(state: State):
    # Stuff relevant docs into one block
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No relevant document found.", "context": ""}

    out = model.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )
    return {"answer": out.content, "context": context}


def no_relevant_docs(state: State):
    return {"answer": "No relevant document found.", "context": ""}


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct"


def route_after_relevance(state: State) -> Literal["generate_from_context", "no_relevant_docs"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_relevant_docs"


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)



issup_llm = model.with_structured_output(IsSUPDecision)

def is_sup(state: State):
    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


MAX_RETRIES = 10

def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    # accept if fully supported
    if state.get("issup") == "fully_supported":
        return "accept_answer"

    # stop if we've already tried enough
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"   # or return a "give_up" node if you want

    # otherwise revise again
    return "revise_answer"


def accept_answer(state: State):
    return {}  # keep answer as-is


revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        ),
    ]
)



def revise_answer(state: State):
    out = model.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }



class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")

isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line."
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
    ]
)

isuse_llm = model.with_structured_output(IsUSEDecision)

def is_use(state: State):
    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}

MAX_REWRITE_TRIES = 3  # tune (2-4 is usually fine)

def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"

    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"

    return "rewrite_question"


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal company PDFs."
    )

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6-16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2-5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}"
        ),
    ]
)



rewrite_llm = model.with_structured_output(RewriteDecision)

def rewrite_question(state: State):
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )

    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # reset so the next pass starts clean
        "docs": [],
        "relevant_docs": [],
        "context": "",
        # FIX: also reset the grounding-loop state. Without this, once the
        # is_sup/revise_answer loop maxes out retries on the FIRST retrieval
        # pass, "retries" stays maxed forever - so on every rewrite pass after
        # that, route_after_issup accepts the answer immediately without ever
        # attempting to revise it against the new context.
        "retries": 0,
        "issup": "",
        "evidence": [],
    }


# FIX: dedicated terminal node for "exhausted all rewrites, still not useful"
# - previously route_after_isuse could return "no_answer_found" but that
# value wasn't registered in the is_use conditional-edge map, so reaching
# this branch crashed with KeyError: 'no_answer_found'.
def no_answer_found(state: State):
    return {
        "answer": "I couldn't find a confident, well-supported answer to this question after "
                   "multiple attempts. It may not be covered in the source documents, or the "
                   "question may need to be more specific.",
    }


g =StateGraph(State)
g.add_node("decide_retrieval", decide_retrieval)
g.add_node("retrieve", retrieve)
g.add_node('generate_direct', generate_direct)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_relevant_docs", no_relevant_docs)
g.add_node("is_relevant", is_relevant)
g.add_node("is_sup", is_sup)
g.add_node("accept_answer", accept_answer)
g.add_node("revise_answer", revise_answer)

g.add_node("is_use", is_use)

g.add_node("rewrite_question", rewrite_question)
g.add_node("no_answer_found", no_answer_found)  # FIX: register the missing node


g.add_edge(START, 'decide_retrieval')
g.add_conditional_edges('decide_retrieval', route_after_decide, {'retrieve' : "retrieve", "generate_direct" : 'generate_direct'})
g.add_edge('generate_direct', END)
g.add_edge("retrieve", "is_relevant")

g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_relevant_docs": "no_relevant_docs",
    },
)
g.add_edge("generate_from_context", "is_sup")
g.add_edge("no_relevant_docs", END)

g.add_conditional_edges(
    "is_sup",
    route_after_issup,
    # FIX: "accept_answer" now actually goes to the accept_answer node
    # (previously mapped straight to "is_use", so the node below never ran).
    {"accept_answer": "accept_answer", "revise_answer": "revise_answer"},
)
g.add_edge("revise_answer", "is_sup")  # loop back to verify
g.add_edge("accept_answer", "is_use")  # FIX: was END - now flows into the usefulness check

g.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found",  # FIX: was "no_relevant_docs", which the
                                                 # function never actually returns
    },
)

g.add_edge("rewrite_question", "retrieve")
g.add_edge("no_answer_found", END)  # FIX: give the new node a way out

app = g.compile()
app


initial_state = {
    "question": "Describe NexaAI’s company culture.",
    "retrieval_query": "",
    "rewrite_tries": 0,
    "docs": [],
    "relevant_docs": [],
    "context": "",
    "answer": "",
    "issup": "",
    "evidence": [],
    "retries": 0,
    "isuse": "not_useful",
    "use_reason": "",
}


result = app.invoke(
    initial_state,
    # FIX: bumped from 80. With retries now correctly resetting per rewrite
    # pass (see rewrite_question), a genuine worst case - MAX_RETRIES=10 x a
    # full grounding loop on each of 1 initial + MAX_REWRITE_TRIES=3 retrieval
    # passes - takes ~109 steps, verified against a live LangGraph run.
    # 80 is no longer enough headroom; 150 leaves comfortable margin.
    config={"recursion_limit": 150},
)


print("\n===== RAG EXECUTION RESULT =====\n")

print("Question:", initial_state.get("question"))
print("Need Retrieval:", result.get("need_retrieval"))

# If you added these counters/fields in your State:
print("Rewrite tries (retrieval):", result.get("rewrite_tries", 0))
print("Support revise tries:", result.get("retries", 0))

print("\nRetrieval:")
print("  Total retrieved docs:", len(result.get("docs", []) or []))
print("  Relevant docs:", len(result.get("relevant_docs", []) or []))

# Optional: show sources/pages for relevant docs
relevant_docs = result.get("relevant_docs", []) or []
if relevant_docs:
    print("\nRelevant docs (source/page):")
    for i, d in enumerate(relevant_docs, 1):
        src = (d.metadata or {}).get("source", "unknown")
        page = (d.metadata or {}).get("page", None)
        title = (d.metadata or {}).get("title", "")
        extra = f", title={title}" if title else ""
        if page is not None:
            print(f"  {i}. source={src}, page={page}{extra}")
        else:
            print(f"  {i}. source={src}{extra}")

print("\nVerification (IsSUP):")
print("  issup:", result.get("issup"))
evidence = result.get("evidence", []) or []
if evidence:
    print("  evidence:")
    for e in evidence:
        print("   -", e)
else:
    print("  evidence: (none)")

print("\nUsefulness (IsUSE):")
print("  isuse:", result.get("isuse"))
print("  reason:", result.get("use_reason", ""))

print("\nFinal Answer:")
print(result.get("answer"))

print("\n===============================\n")

print(result.get("retries"))