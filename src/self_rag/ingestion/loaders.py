from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
from pathlib import Path

from config.settings import settings
from config.logging_config import get_logger

log = get_logger(__name__)

def pdf_paths() -> List[Path]:
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*.pdf"))

def load_pdfs(paths : List[Path]) -> List[Document]:
    docs : List[Document] = []
    for p in paths:
        log.info("Loading %s",p.name)
        docs.extend(PyPDFLoader(str(p)).load())
    return docs
