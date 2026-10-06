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
from src.logger import get_logger
from langchain_core.documents import Document

logger = get_logger(__name__)

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

def decompose_query_node(state: RAGState) -> Dict[str, Any]:
    logger.info("Entering node: query decomposition")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("decompose_query")
    sub_queries = decomposition_engine.decompose(question)
    return {"sub_queries": sub_queries, "steps": steps}

def hyde_node(state: RAGState) -> Dict[str, Any]:
    logger.info("Entering node: HyDE optimization")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("hyde_generation")
    fake_doc_text = hyde_engine.generate_hypothetical_document(question)
    return {"hyde_context": fake_doc_text, "steps": steps}

def transform_query_node(state: RAGState) -> Dict[str, Any]:
    logger.info("Entering node: query transformation")
    raw_question = state["question"]
    steps = state.get("steps", [])
    steps.append("transform_query")
    transform_payload = query_transformer.transform(raw_query=raw_question)
    clean_question = transform_payload.get("optimized_query", raw_question)
    return {"question": clean_question, "steps": steps}

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    logger.info("Entering node: async retrieval & re-ranking")
    question = state["question"]
    sub_queries = state.get("sub_queries", [question])
    hyde_text = state.get("hyde_context", question)
    steps = state.get("steps", [])
    steps.append("retrieve")

    existing_documents = state.get("documents", []) or []

    logger.info("Launching retrieval over HyDE context anchors")
    chroma_context_frames = retrieval_engine.vector_retrieve(hyde_text)

    additional_frames = []
    if len(sub_queries) > 1:
        logger.info("Extracting sub-query targets for deeper structural recall")
        for sub_q in sub_queries:
            sub_hits = retrieval_engine.vector_retrieve(sub_q)
            additional_frames.extend(sub_hits)

    all_raw_vector_hits = chroma_context_frames + additional_frames

    lexical_highway.reset()
    if len(all_raw_vector_hits) > 0:
        lexical_highway.initialize_index(all_raw_vector_hits)

    lexical_hits = lexical_highway.search(question, top_n=2)

    ranked_chroma = retrieval_engine.rerank_cache(question, all_raw_vector_hits)

    fused_documents = existing_documents + ranked_chroma + lexical_hits

    seen_contents = set()
    deduped_documents = []
    for doc in fused_documents:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            deduped_documents.append(doc)

    deduped_documents.sort(key=lambda x: x.metadata.get("score", -99.0), reverse=True)
    final_context_payload = deduped_documents[:4]

    harmonized_documents = harmonize_context_to_markdown(final_context_payload)
    logger.info("Re-ranking finalized. Retained top %d unified frames.", len(harmonized_documents))
    return {"documents": harmonized_documents, "steps": steps}

def generate_node(state: RAGState) -> Dict[str, Any]:
    logger.info("Entering node: generation")
    question = state["question"]
    documents = state["documents"]
    steps = state.get("steps", [])
    steps.append("generate")
    answer = generation_engine.generate_response(question, documents)
    return {"generation": answer, "steps": steps}

def fallback_search_node(state: RAGState) -> Dict[str, Any]:
    logger.info("Entering node: fallback web search")
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("web_search")
    live_web_frames = web_search_engine.search(query=question)
    return {"documents": live_web_frames, "steps": steps}

def increment_healing_iteration(state: RAGState) -> Dict[str, Any]:
    iterations = state.get("healing_iterations", 0) + 1
    logger.info("Self-healing iteration %d", iterations)
    return {"healing_iterations": iterations}

workflow = StateGraph(RAGState)
workflow.add_node("decompose_query", decompose_query_node)
workflow.add_node("hyde_optimization", hyde_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("retrieve_docs", retrieve_node)
workflow.add_node("generate_response", generate_node)
workflow.add_node("fallback_web_search", fallback_search_node)
workflow.add_node("increment_healing_iteration", increment_healing_iteration)

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
        "not useful": "increment_healing_iteration"
    }
)
workflow.add_edge("increment_healing_iteration", "retrieve_docs")

app = workflow.compile()
logger.info("LangGraph state machine compiled successfully")