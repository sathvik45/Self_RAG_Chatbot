import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _clear_lru_caches():
    """The grader/LLM factories are @lru_cache'd for production reuse, which means a
    mocked get_grader()/get_llm() from one test would otherwise leak into the next.
    Clear them before every test so each test's patches take effect cleanly."""
    from src.self_rag.graph import nodes
    from src.self_rag.llm import client

    cached_fns = [
        nodes._contextualizer,
        nodes._router,
        nodes._relevance,
        nodes._grounding,
        nodes._usefulness,
        nodes._rewrite,
        client.get_llm,
        client.get_grader,
        client.get_embeddings,
    ]
    for fn in cached_fns:
        fn.cache_clear()
    yield
    for fn in cached_fns:
        fn.cache_clear()
