from typing import List, Literal, Optional

from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    should_retrieve : bool = Field(
        description="True if answering needs the company documnets;"
        "False only if general traning knwoledge is enough to answer corectly and completely."
    )

class RelevanceDecision(BaseModel):
    is_relevant :  bool = Field(
        description='True only if chuck contains inforamtion useful for the question.'
    )

class GroundingDecision(BaseModel):
    grade : Literal["fully_supported","partially_supported","no_support"]= Field(
        description=(
            "fully_supported: every claim us backed by context."
            "partially_supported: core claims are backed but some pharsing is no stated verbatim. "
            "no_support: anser asserts facts absent from or contradicted by the context."
        )
    )
    evidence : List[str] = Field(
        default_factory= list,
        description= "upto 3 short quotes from the context backing the anser"
    )

class usefulnessDecision(BaseModel):
    is_useful : bool = Field(
        description="True if the answer directly addresses the question."
    )
    reason : str = Field(default="", description="one short line explainaing the call")

class RewrittenQuery(BaseModel):
    query : str = Field(description="A retrieval-optimised restatement of the question")

class StandlaoneQuestion(BaseModel):
    question : str = Field(description= "The user's latest message rewrittten to stand on its own.")


class Source(BaseModel):
    filename : str 
    page : Optional[int] = None
    snippet : str = ""


class ChatRequest(BaseModel):
    question : str = Field(...,min_length=1,max_length=4000)
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque id turns into one conversation."
        "omit to start resh and the server returns one you can reuse"
    )

class ChatResponse(BaseModel):
    answer : str
    sources : List[Source] = Field(default_factory=list)
    session_id : str
    retrieved : bool
    grounding : Optional[str] = None
    trace : List[str] = Field(default_factory=list, description="names of the pipeline steps than ran")

class HealthResponse(BaseModel):
    status : str
    index_ready : bool
    documents_indexed : int
    llm_model : str
    embedding_model : str