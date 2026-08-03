
from typing import Optional
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from src.self_rag.retrieval.vector_store import get_vector_store

def get_retriever() -> Optional[VectorStoreRetriever]:
    vs =get_vector_store()
    if vs is None:
        return None
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
        
    )