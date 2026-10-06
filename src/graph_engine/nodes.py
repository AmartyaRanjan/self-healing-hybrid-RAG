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
from src.logger import get_logger

logger = get_logger(__name__)

default_dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/.env")
load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", os.path.normpath(default_dotenv)))

class QueryDecompositionEngine:
    def __init__(self):
        logger.info("Initializing query decomposition layer")
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
        logger.info("Analyzing structural complexity of input query")
        try:
            raw_output = self.chain.invoke({"query": query}).strip()
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            sub_queries = json.loads(cleaned)
            logger.info("Query split into sub-tasks: %s", sub_queries)
            return sub_queries
        except Exception as e:
            logger.warning("Decomposition failed (%s). Using raw query.", str(e))
            return [query]


class HyDEGenerator:
    def __init__(self):
        logger.info("Initializing HyDE optimization layer")
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
        logger.info("Generating HyDE hypothetical document")
        return self.chain.invoke({"question": question}).strip()


class QueryTransformEngine:
    def __init__(self):
        logger.info("Initializing query transformation layer")
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
        logger.info("Original query: %s", raw_query)
        raw_output = self.chain.invoke({"raw_query": raw_query}).strip()
        try:
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            logger.info("Transformation reasoning: %s", parsed.get('reasoning'))
            logger.info("Transformed query: %s", parsed.get('optimized_query'))
            return parsed
        except Exception as e:
            logger.warning("Query transform parsing failed (%s). Using raw query.", str(e))
            return {"optimized_query": raw_query, "reasoning": "Fallback due to parsing error."}


class PrecisionRetrievalEngine:
    def __init__(self):
        logger.info("Connecting to Chroma vector client")
        self.chroma_client = ChromaVectorClient()
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info("Initializing cross-encoder: %s", model_name)
        
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
        logger.info("Initializing DuckDuckGo web search engine")
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
        self.search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

    def search(self, query: str) -> List[Document]:
        logger.info("Dispatching web search for query: %s", query)
        try:
            raw_web_results = self.search_tool.invoke(query)
            return [Document(
                page_content=str(raw_web_results),
                metadata={"title": "Live Web Search Result", "score": 10.0}
            )]
        except Exception as e:
            logger.error("Web search failed: %s", str(e))
            return []


class ContextualGenerationEngine:
    def __init__(self):
        model_name = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
        logger.info("Initializing contextual generation engine: %s", model_name)
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
        logger.info("Harmonizing context frames and invoking generation")
        formatted_context_blocks = []
        for idx, frame in enumerate(context_frames):
            title = frame.metadata.get('title', 'Data Stream')
            formatted_context_blocks.append(f"--- Reference Source {idx + 1}: {title} ---\n{frame.page_content}")
        return self.chain.invoke({"question": question, "context": "\n\n".join(formatted_context_blocks)})
