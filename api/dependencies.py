
import json
import os
from typing import AsyncGenerator, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from config.settings import settings
from src.self_rag.graph import ANSWER_NODES,STEP_LABLES
from src.self_rag.graph.builder import get_graph
from config.logging_config import get_logger
from api.session import store
from src.self_rag.graph.schemas import Source

log = get_logger(__name__)

_NEW_DRAFT_NODES = {"rewrite_query", "revise_answer"}

def _initial_state(question : str, history : List[BaseMessage]) -> dict:
    return {
    "question": question,
    "original_question" : question,
    "history" : history,
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

def _sources_from(docs : Optional[List[Document]]) -> List[Source]:
    seen = set()
    out : List[Source] =[]
    for d in docs or []:
        meta = d.metadata or {}
        filename = os.path.basename(str(meta.get("source","documnent")))
        page = meta.get("page")
        page = page + 1 if isinstance(page, int) else None
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = "".join(d.page_content.split())[:160]
        out.append(Source(filename=filename,page=page,snippet=snippet))
    return out
def _sse(payload : dict) -> dict:
    return {"data":json.dumps(payload, ensure_ascii=False)}

async def answer_once(question : str, session_id : Optional[str]) -> dict:
    session_id = session_id or store.new_id()
    history = store.get(session_id)
    graph = get_graph()

    final = await graph.ainvoke(
        _initial_state(question,history),
        config={"recursion_limit" : settings.recursion_limit}
    )
    answer = final.get("answer", "") or ""
    store.append(session_id, question, answer)
    return {
        "answer" : answer,
        "sources" : _sources_from(final.get("relevant_docs")),
        "session_id" : session_id,
        "retrieved" : bool(final.get("need_retrieval")),
        "grounding" : final.get("grounding") or None,
        "trace" : []
    }

async def stream_events(

    question: str, session_id: Optional[str]

) -> AsyncGenerator[dict, None]:

    session_id = session_id or store.new_id()

    history = store.get(session_id)

    graph = get_graph()

 

    yield _sse({"type": "meta", "session_id": session_id})

 

    final_answer = ""

    relevant_docs: List[Document] = []

    grounding: Optional[str] = None

    need_retrieval: Optional[bool] = None

    trace: List[str] = []

 

    current_key = None       # (node, step, epoch) of the draft currently streaming

    epoch = 0                # bumps whenever a new draft is about to be produced

 

    try:

        async for mode, chunk in graph.astream(

            _initial_state(question, history),

            config={"recursion_limit": settings.recursion_limit},

            stream_mode=["updates", "messages"],

        ):

            if mode == "updates":

                for node, delta in chunk.items():

                    if node not in trace:

                        trace.append(node)

                    yield _sse(

                        {

                            "type": "status",

                            "node": node,

                            "label": STEP_LABLES.get(node, node),

                        }

                    )

                    if node in _NEW_DRAFT_NODES:

                        epoch += 1

                    if isinstance(delta, dict):

                        if delta.get("answer"):

                            final_answer = delta["answer"]

                        if delta.get("relevant_docs"):

                            relevant_docs = delta["relevant_docs"]

                        if "grounding" in delta and delta["grounding"]:

                            grounding = delta["grounding"]

                        if "need_retrieval" in delta:

                            need_retrieval = delta["need_retrieval"]

 

            elif mode == "messages":

                msg_chunk, meta = chunk

                if meta.get("langgraph_node") not in ANSWER_NODES:

                    continue

                text = getattr(msg_chunk, "content", "") or ""

                if not text:

                    continue

                key = (meta.get("langgraph_node"), meta.get("langgraph_step"), epoch)

                if current_key is not None and key != current_key:

                    yield _sse({"type": "restart"})

                current_key = key

                yield _sse({"type": "token", "content": text})

 

    except Exception:

        log.exception("Streaming run failed")

        yield _sse(

            {

                "type": "error",

                "message": "Something went wrong while answering. Please try again.",

            }

        )

        yield _sse({"type": "done"})

        return

 

    store.append(session_id, question, final_answer)

 

    yield _sse(

        {

            "type": "sources",

            "sources": [s.model_dump() for s in _sources_from(relevant_docs)],

        }

    )

    yield _sse(

        {

            "type": "final",

            "answer": final_answer,

            "grounding": grounding,

            "retrieved": bool(need_retrieval),

            "trace": trace,

        }

    )

    yield _sse({"type": "done"})