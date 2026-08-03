from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chuck_size,
        chunk_overlap=settings.chuck_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(docs: List[Document]) -> List[Document]:
    return get_splitter().split_documents(docs)
