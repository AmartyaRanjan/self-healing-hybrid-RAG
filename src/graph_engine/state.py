from typing import List, TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    sub_queries: List[str]      # FIXED: Track decomposed search queries
    hyde_context: str           # FIXED: Store hypothetical answer text explicitly
    documents: List[Document]
    generation: str
    steps: List[str]
