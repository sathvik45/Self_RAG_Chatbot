from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter
import threading

from config.settings import settings
from config.logging_config import get_logger
from src.self_rag.llm.client import get_embeddings

from typing import List, Optional
from pathlib import Path

from src.self_rag.ingestion.loaders import pdf_paths

log = get_logger(__name__)

_lock = threading.Lock()
_vector_store : Optional[FAISS] = None

def set_vector_store(vs : FAISS):
    global _vector_store
    with _lock:
        _vector_store = vs

def index_exists() -> bool:
    return (Path(settings.index_dir)/"index.faiss").exists()

def _load_from_disk():
    if not index_exists():
        return None
    log.info(f"Loading FAISS index from {settings.index_dir}")
    return FAISS.load_local(
        str(settings.index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

def get_vector_store() -> Optional[FAISS]:
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                _vector_store = _load_from_disk()
    return _vector_store

def documents_indexed() -> int:
    vs = get_vector_store()
    return int(vs.index.ntotal) if vs is not None else 0

def add_to_index(paths: List[Path]) -> int:
    from src.self_rag.ingestion.build_index import add_documents
    return add_documents(paths)

def ensure_index_on_startup():
    if index_exists():
        get_vector_store()
        log.info(f"index ready {documents_indexed()} vectors")
        return
    paths = pdf_paths()
    if paths:
        log.info(f"no index found but {len(paths)} pdf(s) present- building now")
        from src.self_rag.ingestion.build_index import build_index
        try:
            build_index()
            log.info(f"Index built: {documents_indexed()} vectors")
        except Exception:
            log.exception("Failed to build index on startup")
    else:
        log.warning(
            f"No index and no PDFs in {settings.data_dir}. add PDFs and run 'python -m ingestion.ingest.py'"
        )