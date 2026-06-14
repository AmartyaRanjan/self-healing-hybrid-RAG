from typing import List, Dict, Any, TypedDict

class RAGState(TypedDict):
    """centralized graph memory state schema for the sel-healing Hybrid RAG
    pipeline. Tracks context properties across intermediate evaluation
    and execution paths"""

    question: str
    documents: List[Dict[str, Any]]
    generation: str
    steps: List[str]
    