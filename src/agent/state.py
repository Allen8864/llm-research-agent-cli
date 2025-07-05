from typing import List, TypedDict, Optional, Literal

class Document(TypedDict):
    url: str
    title: str
    content: str

class GraphError(TypedDict):
    step: Literal["generate", "search", "reflect", "synthesize"]
    error_type: Literal[
        "RateLimit", "Timeout", "EmptyResult", "LLMFailure", "HTTPError", "UnknownError"
    ]
    message: str

class GraphState(TypedDict):
    question: str
    queries: List[str]
    documents: List[Document]
    need_more: bool
    final_answer: str
    citations: List[dict]
    loop_count: int
    max_iter: int
    errors: List[GraphError]
       
