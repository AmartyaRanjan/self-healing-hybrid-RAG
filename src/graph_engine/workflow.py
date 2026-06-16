import os
import sys
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine.state import RAGState
from src.graph_engine.lexical_engine import LexicalHighwayEngine
from src.graph_engine.nodes import (
    QueryDecompositionEngine,
    HyDEGenerator,
    QueryTransformEngine,
    PrecisionRetrievalEngine,
    DynamicWebSearchEngine,
    ContextualGenerationEngine
)
from src.graph_engine.edges import decide_to_generate, grade_generation_v_documents
from langchain_core.documents import Document

decomposition_engine = QueryDecompositionEngine()
hyde_engine = HyDEGenerator()
query_transformer = QueryTransformEngine()
retrieval_engine = PrecisionRetrievalEngine()
lexical_highway = LexicalHighwayEngine()
web_search_engine = DynamicWebSearchEngine()
generation_engine = ContextualGenerationEngine()

def harmonize_context_to_markdown(documents: list) -> list:
    harmonized_docs = []
    for doc in documents:
        title = doc.metadata.get("title", "System Context Frame")
        score = doc.metadata.get("score", 0.0)
        markdown_content = (
            f"### 📄 Source Reference: {title}\n"
            f"* **Retrieval Confidence Score:** {score:.4f}\n"
            f"```markdown\n{doc.page_content.strip()}\n```\n"
        )
        harmonized_docs.append(Document(page_content=markdown_content, metadata=doc.metadata))
    return harmonized_docs

def decompose_query_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: QUERY DECOMPOSITION ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("decompose_query")
    decomposition_engine.decompose(question)
    return {"steps": steps}

def hyde_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: HYDE OPTIMIZATION ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("hyde_generation")
    fake_doc_text = hyde_engine.generate_hypothetical_document(question)
    hyde_document_frame = Document(
        page_content=fake_doc_text,
        metadata={"title": "Upstream HyDE Structural Anchor", "score": 1.5}
    )
    return {"documents": [hyde_document_frame], "steps": steps}

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
    HIGH TRAFFIC ASYNC RETRIEVAL: Executes multi-engine queries concurrently 
    to handle enterprise scalability requirements.
    """
    print("\n--- ENTERING NODE: ASYNC RETRIEVAL & RE-RANKING ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("retrieve")
    
    existing_documents = state.get("documents", []) or []
    
    # Run the Vector Engine retrieval inside an async loop event to handle scaling concurrent traffic
    print("[*] Scaling Pipeline: Launching asynchronous retrieval loop...")
    chroma_context_frames = asyncio.run(retrieval_engine.async_vector_retrieve(question))
    
    if len(chroma_context_frames) > 0 and not lexical_highway.bm25:
        lexical_highway.initialize_index(chroma_context_frames)
        
    lexical_hits = lexical_highway.search(question, top_n=2)
    
    # Run cross-encoder over retrieved vectors
    ranked_chroma = retrieval_engine.rerank_cache(question, chroma_context_frames)
    
    # Fuse all data tracks
    fused_documents = existing_documents + ranked_chroma + lexical_hits
    fused_documents.sort(key=lambda x: x.metadata.get("score", -99.0), reverse=True)
    fused_documents = fused_documents[:4]
    
    harmonized_documents = harmonize_context_to_markdown(fused_documents)
    print(f"[+] Re-Ranking & Harmonization finalized. Retained top {len(harmonized_documents)} unified frames.")
    return {"documents": harmonized_documents, "steps": steps}

def generate_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: GENERATION ---")
    question = state["question"]
    documents = state["documents"]
    steps = state.get("steps", [])
    steps.append("generate")
    answer = generation_engine.generate_response(question, documents)
    return {"generation": answer, "steps": steps}

def fallback_search_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: FALLBACK WEB SEARCH ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("web_search")
    live_web_frames = web_search_engine.search(query=question)
    return {"documents": live_web_frames, "steps": steps}

# Build LangGraph Loop Mapping
workflow = StateGraph(RAGState)
workflow.add_node("decompose_query", decompose_query_node)
workflow.add_node("hyde_optimization", hyde_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("retrieve_docs", retrieve_node)
workflow.add_node("generate_response", generate_node)
workflow.add_node("fallback_web_search", fallback_search_node)

workflow.add_edge(START, "decompose_query")
workflow.add_edge("decompose_query", "hyde_optimization")
workflow.add_edge("hyde_optimization", "transform_query")
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
print("[+] LangGraph State Machine compiled successfully with complete Phase 3 & 4 Optimization nodes.")
