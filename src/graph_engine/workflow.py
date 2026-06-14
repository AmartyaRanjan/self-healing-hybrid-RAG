import os
import sys
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine.state import RAGState
from src.graph_engine.nodes import QueryTransformEngine, PrecisionRetrievalEngine, ContextualGenerationEngine
from src.graph_engine.edges import decide_to_generate, grade_generation_v_documents

query_transformer = QueryTransformEngine()
retrieval_engine = PrecisionRetrievalEngine()
generation_engine = ContextualGenerationEngine()

def transform_query_node(state: RAGState) -> Dict[str, Any]:
    """
    Node wrapper utilizing structured JSON transformations to maximize audit clarity.
    """
    print("\n--- ENTERING NODE: QUERY TRANSFORMATION ---")
    raw_question = state["question"]
    steps = state.get("steps", [])
    steps.append("transform_query")
    
    # Run structured payload transform mapping
    transform_payload = query_transformer.transform(raw_question)
    clean_question = transform_payload.get("optimized_query", raw_question)
    
    return {"question": clean_question, "steps": steps}

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Node wrapper running soft threshold context extraction.
    """
    print("\n--- ENTERING NODE: RETRIEVAL & RE-RANKING ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("retrieve")
    
    top_context_frames = retrieval_engine.retrieve_and_rerank(question)
    return {"documents": top_context_frames, "steps": steps}

def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Node wrapper compiling output synthesis.
    """
    print("\n--- ENTERING NODE: GENERATION ---")
    question = state["question"]
    documents = state["documents"]
    steps = state.get("steps", [])
    steps.append("generate")
    
    answer = generation_engine.generate_response(question, documents)
    return {"generation": answer, "steps": steps}

def fallback_search_node(state: RAGState) -> Dict[str, Any]:
    """
    The activated fallback node when local metrics indicate a complete knowledge gap.
    """
    print("\n--- ENTERING NODE: FALLBACK WEB SEARCH (SELF-HEALING EXECUTOR) ---")
    steps = state.get("steps", [])
    steps.append("web_search")
    
    # Live production fallback vector container injection simulation
    # (Tavily Search API interface hook point)
    healing_context = [{
        "page_content": (
            "System Resolution Notice: Local vector index failed to yield contextual matches clearing relevance ceilings. "
            "Fallback system override successfully engaged to prevent generation silence."
        ),
        "metadata": {"title": "Web Search System Override Escape Hatch"}
    }]
    return {"documents": healing_context, "steps": steps}


# Workflow configuration map compilation
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
