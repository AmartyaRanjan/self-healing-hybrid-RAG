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
            f"### [Doc] Source Reference: {title}\n"
            f"* **Retrieval Confidence Score:** {score:.4f}\n"
            f"```markdown\n{doc.page_content.strip()}\n```\n"
        )
        harmonized_docs.append(Document(page_content=markdown_content, metadata=doc.metadata))
    return harmonized_docs

# =====================================================================
# FIXED WORKING NODE ROUTING
# =====================================================================

def decompose_query_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: QUERY DECOMPOSITION ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("decompose_query")
    
    # FIXED: Capture sub-queries and save them directly to state tracking context
    sub_queries = decomposition_engine.decompose(question)
    return {"sub_queries": sub_queries, "steps": steps}

def hyde_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: HYDE OPTIMIZATION ---")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("hyde_generation")
    
    # FIXED: Save the hypothetical text explicitly to state dictionary context
    fake_doc_text = hyde_engine.generate_hypothetical_document(question)
    return {"hyde_context": fake_doc_text, "steps": steps}

def transform_query_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- ENTERING NODE: QUERY TRANSFORMATION ---")
    raw_question = state["question"]
    steps = state.get("steps", [])
    steps.append("transform_query")
    
    transform_payload = query_transformer.transform(raw_query=raw_question)
    clean_question = transform_payload.get("optimized_query", raw_question)
    return {"question": clean_question, "steps": steps}

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    FIXED HYBRID FUSION: Actively passes sub-queries and HyDE text vectors 
    downstream to ensure complete multi-database coverage.
    """
    print("\n--- ENTERING NODE: ASYNC RETRIEVAL & RE-RANKING ---")
    question = state["question"]
    sub_queries = state.get("sub_queries", [question])
    hyde_text = state.get("hyde_context", question)
    steps = state.get("steps", [])
    steps.append("retrieve")
    
    existing_documents = state.get("documents", []) or []
    
    # FIXED: Run retrieval using the dense semantic HyDE context text block to optimize matching target
    print("[*] Scaling Pipeline: Launching retrieval over HyDE context anchors...")
    chroma_context_frames = retrieval_engine.vector_retrieve(hyde_text)
    
    # FIXED: Run a separate fallback pass for individual sub-queries if they exist
    additional_frames = []
    if len(sub_queries) > 1:
        print(f"[*] Extracting sub-query targets for deeper structural recall context...")
        for sub_q in sub_queries:
            sub_hits = retrieval_engine.vector_retrieve(sub_q)
            additional_frames.extend(sub_hits)
            
    all_raw_vector_hits = chroma_context_frames + additional_frames
    
    # Reset the lexical highway so each retrieval pass (including self-healing
    # loop iterations) builds a fresh index from the current query's hits.
    lexical_highway.reset()
    if len(all_raw_vector_hits) > 0:
        lexical_highway.initialize_index(all_raw_vector_hits)
        
    lexical_hits = lexical_highway.search(question, top_n=2)
    
    # Run cross-encoder over the text blocks using the optimized user question frame
    ranked_chroma = retrieval_engine.rerank_cache(question, all_raw_vector_hits)
    
    # Deduplicate and fuse everything
    fused_documents = existing_documents + ranked_chroma + lexical_hits
    
    # Prevent index duplication collisions
    seen_contents = set()
    deduped_documents = []
    for doc in fused_documents:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            deduped_documents.append(doc)
            
    deduped_documents.sort(key=lambda x: x.metadata.get("score", -99.0), reverse=True)
    final_context_payload = deduped_documents[:4]
    
    harmonized_documents = harmonize_context_to_markdown(final_context_payload)
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
