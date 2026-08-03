
from typing import List, Literal,TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class State(TypedDict, total= False):

    question : str
    original_question : str
    history : List[BaseMessage]

    need_retrieval : bool

    retrieval_query : str
    rewrite_tries : int
    docs : List[Document]
    relevant_docs: List[Document]
    context : str

    grounding : Literal["fully_supported","partially_supported","no_support"]
    evidence : List[str]
    revise_tries : int

    is_useful : bool
    use_reason : str
    
    answer : str