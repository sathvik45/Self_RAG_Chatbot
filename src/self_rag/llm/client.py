from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

from config.settings import settings
from config.logging_config import get_logger
import os
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

log = get_logger(__name__)

def _ensure_groq_key() -> None:
    if settings.groq_api_key and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = settings.groq_api_key


def _ensure_hf_token() -> None:
    if settings.hf_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = settings.hf_token


@lru_cache
def get_llm() -> ChatGroq:
    _ensure_groq_key()
    log.info(f"Initialising answer LLM {settings.llm_model}")
    return ChatGroq(
        model = settings.llm_model,
        temperature= settings.llm_temperature,
        streaming=True,
    )

@lru_cache
def get_grader() -> ChatGroq:
    _ensure_groq_key()
    name = settings.grader_model_name
    log.info(f"Initalising grader LLM {name}")
    return ChatGroq(model = name, temperature = 0.0, disable_streaming="tool_calling")


@lru_cache
def get_embeddings() -> Embeddings:
    _ensure_hf_token()
    if settings.hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    log.info(f"Initalising embedding model {settings.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name = settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
