import os
import sys
import json
import asyncio
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

default_dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/.env")
load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", os.path.normpath(default_dotenv)))

class QueryDecompositionEngine:
    def __init__(self):
        print("[*] Initializing Phase 3: Query Decomposition Layer...")
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GENERATION_MODEL", "gemini-2.5-flash"),
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an advanced query deconstruction model.\n"
                "Your job is to analyze an incoming user query. If it contains multiple distinct "
                "technical topics, questions, or cross-phase dependencies, break it down into a list of "
                "2 to 3 simplified, individual sub-queries optimized for a search index.\n"
                "Respond strictly with a raw JSON array of strings. Do not include markdown code block wrappers."
            )),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def decompose(self, query: str) -> List[str]:
        print(f"[*] Analyzing structural complexity of input query...")
        try:
            raw_output = self.chain.invoke({"query": query}).strip()
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            sub_queries = json.loads(cleaned)
            print(f"[+] Query split into sub-tasks: {sub_queries}")
            return sub_queries
        except Exception as e:
            print(f"[!] Target decomposition failed ({str(e)}). Proceeding with raw query vector.")
            return [query]


class HyDEGenerator:
    def __init__(self):
        print("[*] Initializing Phase 3: HyDE Optimization Layer...")
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GENERATION_MODEL", "gemini-2.5-flash"),
            temperature=0.6,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert enterprise systems architect and technical writer.\n"
                "Write a highly detailed, idealized technical passage that perfectly answers the user's question.\n"
                "Do not include introductory commentary—output ONLY the direct, "
                "hypothetical technical response payload text."
            )),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_hypothetical_document(self, question: str) -> str:
        print(f"[*] HyDE: Generating optimized target semantic answer text...")
        return self.chain.invoke({"question": question}).strip()


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
                "Respond strictly with a valid JSON object containing exactly 'optimized_query' and 'reasoning' keys."
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
            print(f"[->] Transformation Reasoning: {parsed.get('reasoning')}")
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

    async def async_vector_retrieve(self, question: str) -> List[Document]:
        """Runs the semantic vector over-retrieval loop inside an async thread pool."""
        loop = asyncio.get_event_loop()
        retriever = self.chroma_client.get_retriever(search_kwargs={"k": 15})
        # Offload synchronous LangChain network I/O to thread pool
        return await loop.run_in_executor(None, retriever.invoke, question)

    def vector_retrieve(self, question: str) -> List[Document]:
        """Runs the semantic vector over-retrieval loop synchronously."""
        retriever = self.chroma_client.get_retriever(search_kwargs={"k": 15})
        return retriever.invoke(question)

    def rerank_cache(self, question: str, initial_docs: List[Document]) -> List[Document]:
        """Applies Cross-Encoder scaling weights over fused hits."""
        if not initial_docs:
            return []
        pairs = [[question, doc.page_content] for doc in initial_docs]
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
            return [Document(
                page_content=str(raw_web_results),
                metadata={"title": "Live Web Search Result", "score": 10.0}
            )]
        except Exception as e:
            print(f"[!] Critical Error: Web fallback breakout failed ({str(e)}).")
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
                "Your task is to answer the user's question using the provided reference context facts.\n\n"
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
            formatted_context_blocks.append(f"--- Reference Source {idx + 1}: {title} ---\n{frame.page_content}")
        return self.chain.invoke({"question": question, "context": "\n\n".join(formatted_context_blocks)})
