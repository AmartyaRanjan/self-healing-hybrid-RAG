import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path = os.path.join(os.path.dirname(__file__), "../../config/.env"))

class ChromaVectorClient:
    def __init__ (self, collection_name: str = "self_healing_rag_docs"):
        #Retrieve target storage vriables from the environment
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_storage")
        #the embedding model that we are going to use
        model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
        print(f"[*] Initializing Google Generative AI Embeddings: {model_name}...")
        self.emebddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        print(f"[*] Connecting to ChromaDB at {db_path}...")
        #connect the LangChain Chroma wrapper to persistent storage
        self.client = Chroma(
            collection_name=collection_name,
            embedding_function=self.emebddings,
            persist_directory=db_path
        )
    
    def add_documents(self, finalized_chunks: List[Dict[str, Any]]):
        """Embeds full context payloads and stores them inside
        the local Chroma vector indices database instance"""
        print(f"[*]Vectorizing and uploading {len(finalized_chunks}")} structural chunks to Chroma DB ...")
        texts = [chunks["page_content"] for chunk in finlize_chunks]
        metadatas = [chunks["metadata"]["chunk_id"] for chunk in finalized_chunks]

        #execute batch upload and vector computation
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids= ids)
        print("[*] Chroma DB indexing successfully accomplished")
    
    def get_retriever(self, search_kwargs: Dict[str, Any]=None):
        """returns a standarzed LangChain Retreiver interface."""
        if search_kwargs is None:
            search_kwargs = {"k": 15} #default to retrieving top 5 most relevant chunks
        return self.vector_store.as_retriever(
            search_type= "similarity",
            search_kwargs=search_kwargs
            )
    