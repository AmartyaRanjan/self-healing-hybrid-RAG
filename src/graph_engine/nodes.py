import os
import sys
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import CrossEncoder
from src.database.chroma_client import ChromaVectorClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", "config/.env"))

class QueryTransformEngine:
    def __init__(self):
        print("[*] Initializing Query Transformation Optimization Layer...")
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GENERATION_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert search optimization engine.\n"
                "Your task is to analyze an incoming user query, fix typos, and optimize it for a vector database.\n"
                "You must respond strictly with a valid JSON object containing exactly two keys:\n"
                "1. 'optimized_query': The polished, semantically rich string for search retrieval.\n"
                "2. 'reasoning': A brief explanation of what structural changes or corrections you made and why.\n\n"
                "Do not include markdown code block wrappers (like ```json) in your response. Output raw JSON text."
            )),
            ("human", "{raw_query}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def transform(self, raw_query: str) -> Dict[str, str]:
        print(f"[*] Original Query: '{raw_query}'")
        raw_output = self.chain.invoke({"raw_query": raw_query}).strip()
        
        try:
            # Clean potential markdown wrappers if the LLM slips them in
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            print(f"[➔] Transformation Reasoning: {parsed.get('reasoning')}")
            print(f"[+] Transformed Search Query: '{parsed.get('optimized_query')}'")
            return parsed
        except Exception as e:
            print(f"[!] Warning: Query transform parsing failed ({str(e)}). Falling back to raw text input.")
            return {"optimized_query": raw_query, "reasoning": "Fallback due to parsing error."}


class PrecisionRetrievalEngine:
    def __init__(self):
        print("[*] Connecting to Chroma Vector Client for retrieval indexing...")
        self.chroma_client = ChromaVectorClient()
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"[*] Initializing localized Cross-Encoder pass: {model_name}...")
        
        hf_token = os.environ.pop("HF_TOKEN", None)
        hf_hub_token = os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
        try:
            self.re_ranker = CrossEncoder(model_name)
        finally:
            if hf_token is not None: os.environ["HF_TOKEN"] = hf_token
            if hf_hub_token is not None: os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_hub_token

    def retrieve_and_rerank(self, question: str) -> List[Dict[str, Any]]:
        """
        Executes a two-stage retrieval pass with an optimized soft threshold 
        to capture nuance while filtering severe structural noise.
        """
        print(f"\n[*] Stage 1: Over-retrieving context frames (k=15)...")
        retriever = self.chroma_client.get_retriever(search_kwargs={"k": 15})
        initial_docs = retriever.invoke(question)

        if not initial_docs:
            print("[!] Warn: Vector search returned 0 documents.")
            return []
        
        pairs = [[question, doc.page_content] for doc in initial_docs]
        print("[*] Stage 2: Running cross-encoder pass to score contextual relevance...")
        scores = self.re_ranker.predict(pairs)

        scored_docs = []
        for idx, score in enumerate(scores):
            scored_docs.append({
                "score": float(score),
                "page_content": initial_docs[idx].page_content,
                "metadata": initial_docs[idx].metadata
            })
        
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # FIX: Lower threshold to -1.0 to let slightly lower-scoring but complementary chunks pass
        MIN_SCORE_THRESHOLD = -1.0
        top_k_final = [doc for doc in scored_docs if doc["score"] >= MIN_SCORE_THRESHOLD][:4]
        
        if not top_k_final and scored_docs:
            print("[!] Warn: No chunks cleared threshold. Holding absolute best match.")
            top_k_final = [scored_docs[0]]

        print(f"[+] Re-Ranking finalized. Retained top {len(top_k_final)} contexts (Best Score: {scored_docs[0]['score']:.4f})")
        return top_k_final


class ContextualGenerationEngine:
    def __init__(self):
        model_name = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
        print(f"[*] Initializing Contextual Generation Engine via : {model_name}...")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an advanced, deterministic enterprise RAG synthesis engine.\n"
                "Your task is to answer the user's question using ONLY the provided, "
                "re-ranked document context. If the context does not contain the answer, "
                "explicitly state that you do not possess sufficient data. Do not hallucinate.\n\n"
                "=== RETRIEVED DATA CONTEXT ===\n{context}"
            )),
            ("human", "{question}")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_response(self, question: str, context_frames: List[Dict[str, Any]]) -> str:
        print("[*] Harmonizing context frames and invoking Gemini generation chain ...")
        formatted_context_blocks = []
        for idx, frame in enumerate(context_frames):
            formatted_context_blocks.append(
                f"--- Context Frame {idx + 1} (Source : {frame['metadata'].get('title', 'Document Asset')}) ---\n"
                f"{frame['page_content']}"
            )
        unified_context_string = "\n\n".join(formatted_context_blocks)
        return self.chain.invoke({"question": question, "context": unified_context_string})
