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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document

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

    def retrieve_and_rerank(self, question: str) -> List[Document]:
        print(f"\n[*] Stage 1: Over-retrieving context frames (k=15)...")
        retriever = self.chroma_client.get_retriever(search_kwargs={"k": 15})
        initial_docs = retriever.invoke(question)

        if not initial_docs:
            print("[!] Warn: Vector search returned 0 documents.")
            return []
        
        pairs = [[question, doc.page_content] for doc in initial_docs]
        print("[*] Stage 2: Running cross-encoder pass to score contextual relevance...")
        scores = self.re_ranker.predict(pairs)

        for idx, score in enumerate(scores):
            initial_docs[idx].metadata["score"] = float(score)
            
        return initial_docs


class DynamicWebSearchEngine:
    def __init__(self):
        print("[*] Initializing Live DuckDuckGo API Search Engine...")
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
        self.search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

    def search(self, query: str) -> List[Document]:
        print(f"[*] Dispatching live web scraper payload for query: '{query}'...")
        try:
            raw_web_results = self.search_tool.invoke(query)
            
            healed_context = [Document(
                page_content=str(raw_web_results),
                metadata={"title": "Live Web Search Result", "score": 10.0}  # Force anchor high priority
            )]
            print("[+] Live web lookup execution finalized successfully.")
            return healed_context
        except Exception as e:
            print(f"[!] Critical Error: Web fallback breakout failed ({str(e)}). Returning empty frame.")
            return []


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
                "You are an advanced, deterministic enterprise synthesis engine.\n"
                "Your task is to answer the user's question using the provided reference context facts.\n"
                "This context contains information retrieved from either internal system documentation, "
                "relational knowledge graphs, or real-time live web search results.\n\n"
                "Review the context facts carefully and synthesize a clear response answering the query. "
                "If the context does not contain enough information to answer, explicitly state that you possess "
                "insufficient data. Do not use outside assumptions.\n\n"
                "=== PROVIDED REFERENCE CONTEXT ===\n{context}"
            )),
            ("human", "{question}")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_response(self, question: str, context_frames: List[Document]) -> str:
        print("[*] Harmonizing context frames and invoking Gemini generation chain ...")
        formatted_context_blocks = []
        for idx, frame in enumerate(context_frames):
            title = frame.metadata.get('title', 'Data Stream')
            formatted_context_blocks.append(
                f"--- Reference Source {idx + 1}: {title} ---\n"
                f"{frame.page_content}"
            )
        unified_context_string = "\n\n".join(formatted_context_blocks)
        return self.chain.invoke({"question": question, "context": unified_context_string})
