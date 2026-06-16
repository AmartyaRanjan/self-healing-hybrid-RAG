import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

class ChromaVectorClient:
    def __init__(self, collection_name: str = "self_healing_rag_docs"):
        # Retrieve target engine configurations from environment with clean defaults
        model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))
        
        print(f"[*] Initializing Google Generative AI Embeddings: {model_name}...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # FIXED FOR CONCURRENCY: Connecting via HttpClient network socket instead of SQLite file locks
        print(f"[*] Connecting to Distributed Chroma Server at http://{chroma_host}:{chroma_port}...")
        self.chroma_http_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.collection_name = collection_name

        # Bind the LangChain wrapper client interface to the living server pool
        self.vector_store = Chroma(
            client=self.chroma_http_client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def add_documents(self, finalized_chunks: List[Dict[str, Any]]):
        """
        UNTOUCHED LOGIC: Embeds full context payloads and stores them inside 
        the server-backed Chroma vector indices database instance.
        """
        print(f"[*] Vectorizing and uploading {len(finalized_chunks)} structural chunks to Chroma DB ...")
        texts = [chunk["page_content"] for chunk in finalized_chunks]
        metadatas = [chunk["metadata"] for chunk in finalized_chunks]
        ids = [chunk["metadata"]["chunk_id"] for chunk in finalized_chunks]

        # Execute cluster batch upload over HTTP endpoint
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print("[*] Chroma DB indexing successfully accomplished.")

    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Returns a standardized LangChain Retriever interface."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
