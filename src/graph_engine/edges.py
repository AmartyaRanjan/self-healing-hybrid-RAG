import os
import sys
import json
from typing import Literal, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine.state import RAGState

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GENERATION_MODEL", "gemini-2.5-flash"),
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def decide_to_generate(state: RAGState) -> Literal["generate", "web_search"]:
    """
    Evaluates context viability across disparate dictionary structures and LangChain Document 
    objects. Routes to 'web_search' if the highest context score indicates irrelevant noise.
    """
    print("[*] Edge Evaluator: Assessing document context payload density...")
    docs = state.get("documents", [])
    
    if not docs or len(docs) == 0:
        print("[➔] Decision: Document payload empty. Routing to Fallback Web Search.")
        return "web_search"
    
    # Extract the leading context item
    best_doc = docs[0]
    best_score = -99.0
    
    # FIXED: Support both dictionary schemas and LangChain Document objects seamlessly
    if hasattr(best_doc, "metadata"):  # LangChain Document Object
        best_score = best_doc.metadata.get("score", -99.0)
    elif isinstance(best_doc, dict):    # Standard Python Dictionary
        best_score = best_doc.get("score", -99.0)
        
    CRITICAL_SCORE_FLOOR = -3.0
    
    if best_score < CRITICAL_SCORE_FLOOR:
        print(f"[!] Edge Evaluator: Best chunk score ({best_score:.4f}) is below acceptable floor ({CRITICAL_SCORE_FLOOR}). Triggering Escape Hatch!")
        return "web_search"
        
    print(f"[➔] Decision: Valid context identified (Score: {best_score:.4f}). Routing to Generation Node.")
    return "generate"

def grade_generation_v_documents(state: RAGState) -> Literal["useful", "not useful"]:
    """
    Acts as an Anti-Hallucination Guardrail verifying output factual grounding.
    """
    print("[*] Edge Evaluator: Executing rigorous hallucination and grounding audit...")
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    
    if not generation:
        return "not useful"
    
    # Standardize string compilation for validation
    context_blocks = []
    for doc in documents:
        if hasattr(doc, "page_content"):
            context_blocks.append(doc.page_content)
        elif isinstance(doc, dict):
            context_blocks.append(doc.get("page_content", ""))
            
    context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."
    
    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an enterprise compliance judge inspecting an AI system's output.\n"
            "Your job is to determine if the generated answer is strictly supported by the "
            "provided context facts. Do not allow outside knowledge.\n\n"
            "Respond strictly in a valid JSON object format with a single key 'score' "
            "and a string value of either 'yes' (fully grounded) or 'no' (hallucinated or ungrounded).\n\n"
            "=== REFERENCE DOCUMENT FACTS ===\n{context}"
        )),
        ("human", "=== SYSTEM GENERATED RESPONSE ===\n{generation}\n\nIs this grounded? Provide the JSON output now.")
    ])
    
    try:
        grader_chain = grading_prompt | llm
        result = grader_chain.invoke({"context": context_text, "generation": generation})
        cleaned = result.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        
        if parsed.get("score", "no").lower() == "yes":
            print("[+] Audit Passed: Output is fully grounded. No hallucinations detected.")
            return "useful"
        else:
            print("[!] Audit Failed: Output contains ungrounded claims.")
            return "not useful"
    except Exception as e:
        print(f"[!] Grader parsing fault: {str(e)}. Defaulting to safety regeneration pass.")
        return "not useful"
