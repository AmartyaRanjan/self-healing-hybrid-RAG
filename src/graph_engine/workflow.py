import os
import sys
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine.state import RAGState
from src.graph_engine.nodes import QueryTransformEngine, PrecisionRetrievalEngine, DynamicWebSearchEngine, ContextualGenerationEngine
from src.graph_engine.edges import decide_to_generate, grade_generation_v_documents

query_transformer = QueryTransformEngine()
retrieval_engine = PrecisionRetrievalEngine()
web_search_engine = DynamicWebSearchEngine()
generation_engine = ContextualGenerationEngine()

def transform_query_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: QUERY TRANSFORMATION ---")
    raw_question = state["question"]
    steps = state.get("steps", [])
    steps.append("transform_query")
    
    transform_payload = query_transformer.transform(raw_question)
    clean_question = transform_payload.get("optimized_query", raw_question)
    
    return {"question": clean_question, "steps": steps}

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Node wrapper running soft threshold context extraction.
    Fuses upstream graph document contexts with incoming Chroma vector frames.
    """
    print("\n--- ENTERING NODE: RETRIEVAL & RE-RANKING ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("retrieve")
    
    existing_documents = state.get("documents", []) or []
    if existing_documents:
        print(f"[*] Preserving {len(existing_documents)} upstream Graph context frame(s).")
    
    # Fetch Chroma text contexts
    chroma_context_frames = retrieval_engine.retrieve_and_rerank(question)
    
    # Combine lists
    fused_documents = existing_documents + chroma_context_frames
    
    # Sort descending based on metadata score attributes, prioritizing high anchors
    fused_documents.sort(key=lambda x: x.metadata.get("score", -99.0), reverse=True)
    
    # Cap context density at top 4 highly targeted context frames
    fused_documents = fused_documents[:4]
    
    print(f"[+] Re-Ranking finalized. Retained top {len(fused_documents)} unified contexts (Best Score: {fused_documents[0].metadata.get('score'):.4f})")
    return {"documents": fused_documents, "steps": steps}

def generate_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: GENERATION ---")
    question = state["question"]
    documents = state["documents"]
    steps = state.get("steps", [])
    steps.append("generate")
    
    answer = generation_engine.generate_response(question, documents)
    return {"generation": answer, "steps": steps}

def fallback_search_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: FALLBACK WEB SEARCH (LIVE GENERATION HEALER) ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("web_search")
    
    live_web_frames = web_search_engine.search(query=question)
    return {"documents": live_web_frames, "steps": steps}


# Workflow compilation layout map
workflow = StateGraph(RAGState)

workflow.add_node("transform_query", transform_query_node)
workflow.add_node("retrieve_docs", retrieve_node)
workflow.add_node("generate_response", generate_node)
workflow.add_node("fallback_web_search", fallback_search_node)

workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "retrieve_docs")

workflow.add_conditional_edges(
    "retrieve_docs",
    decide_to_generate,
    {
        "generate": "generate_response",
        "web_search": "fallback_web_search"
    }
)

workflow.add_edge("fallback_web_search", "generate_response")

workflow.add_conditional_edges(
    "generate_response",
    grade_generation_v_documents,
    {
        "useful": END,
        "not useful": "retrieve_docs"
    }
)

app = workflow.compile()
print("[+] LangGraph Self-Healing State Engine compiled with absolute success.")
