

from typing import List
from langchain_community.vectorstores import FAISS
from config.settings import settings
from src.self_rag.llm.client import get_embeddings
from src.self_rag.ingestion.loaders import load_pdfs,pdf_paths
from config.logging_config import get_logger
from src.self_rag.ingestion.splitters import split_documents
from src.self_rag.retrieval.vector_store import get_vector_store, set_vector_store
from pathlib import Path
log = get_logger(__name__)

def build_index() -> int:

    paths = pdf_paths()
    if not paths:
        raise FileNotFoundError(
            f"No PDFs found in {settings.data_dir}. Add docs and try again."
        )

    docs =load_pdfs(paths)
    chucks = split_documents(docs)
    log.info("loaded %d pages -> %d chunks", len(docs), len(chucks))

    embeddings = get_embeddings()
    log.info(f"Embedding {len(chucks)} chucks and building FAISS index...")
    vs = FAISS.from_documents(chucks, embeddings)

    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok= True)
    vs.save_local(str(index_dir))

    set_vector_store(vs)
    return vs.index.ntotal


def add_documents(paths: List[Path]) -> int:
    """Embed only the given PDFs and merge them into the existing index, instead of
    re-embedding the whole corpus. Falls back to a full build if no index exists yet."""
    if not paths:
        raise FileNotFoundError("No PDF paths given to index.")

    docs = load_pdfs(paths)
    chunks = split_documents(docs)
    log.info("loaded %d pages -> %d chunks", len(docs), len(chunks))

    vs = get_vector_store()
    if vs is None:
        log.info(f"No existing index - embedding {len(chunks)} chunks and building FAISS index...")
        vs = FAISS.from_documents(chunks, get_embeddings())
    else:
        log.info(f"Embedding {len(chunks)} new chunks and adding to the existing FAISS index...")
        vs.add_documents(chunks)

    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))

    set_vector_store(vs)
    return vs.index.ntotal
