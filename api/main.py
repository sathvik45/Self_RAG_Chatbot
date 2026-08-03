
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from api import __version__,dependencies
from config.settings import settings
from config.logging_config import configure_logging, get_logger
from src.self_rag.graph.schemas import ChatRequest,ChatResponse, HealthResponse
from api.session import store
from src.self_rag.retrieval.vector_store import documents_indexed, ensure_index_on_startup, index_exists, add_to_index

log = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR/ "static"

@asynccontextmanager
async def lifespan(app : FastAPI):
    configure_logging()
    log.info("starting self-RAG PDF chatbot v%s", __version__)
    if not settings.groq_api_key:
        log.warning("LLM key is not set.")
    await run_in_threadpool(ensure_index_on_startup)
    yield
    log.info("shutting down...")

app = FastAPI(title="Self-RAG PDF chatbot",version=__version__,lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"]
)
if STATIC_DIR.exists():
    app.mount("/static",StaticFiles(directory=str(STATIC_DIR)), name= "static")

@app.get("/", include_in_schema=False)
async def index():
    html = STATIC_DIR/"index.html"
    if html.exists():
        return FileResponse(str(html))
    return JSONResponse({"message" : "Self-RG PDF Chatbot API. see /docs."})

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="OK",
        index_ready= index_exists(),
        documents_indexed = documents_indexed(),
        llm_model=settings.llm_model,
        embedding_model = settings.embedding_model,
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await dependencies.answer_once(req.question, req.session_id)
    except Exception:
        log.exception("chat failed")
        raise HTTPException(status_code=500, detail="Failed to genertate an answer.")
    return ChatResponse(**result)

@app.post("/chat/stream")
async def chat_stream(req : ChatRequest):
    generator = dependencies.stream_events(req.question,req.session_id)
    return EventSourceResponse(
        generator,
        sep="\n",
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering" : "no"
        }
    )

@app.post("/reset")
async def reset(req : ChatRequest):
    if req.session_id:
        store.reset(req.session_id)
    return {"status" : "cleared"}


@app.post("/upload")
async def upload(files : List[UploadFile] = File(...)):
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents =True,exist_ok=True)

    saved : List[str] = []
    saved_paths : List[Path] = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"not a PD {f.filename}")
        dest = data_dir / Path(f.filename).name
        dest.write_bytes(await f.read())
        saved.append(dest.name)
        saved_paths.append(dest)

    try:
        chunks = await run_in_threadpool(add_to_index, saved_paths)
    except Exception:
        log.exception("Reindex after upload failed")
        raise HTTPException(status_code=500,detail="Saved files but indexing failed")
    return {"saved" : saved, "chunks_indexed" : chunks}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )