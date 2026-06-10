import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

# Ensure the root project directory is on the system path for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import CrossEncoder
from src.database.chroma_client import ChromaVectorClient

# Load environment configurations
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

class PrecisionRetrievalEngine:
    def __init__(self):
        print("[*] Connecting to Chroma Vector Client for retrieval indexing...")
        self.chroma_client = ChromaVectorClient()
        
        # Phase 1.3: Local Hugging Face Cross-Encoder initialization
        # Wraps cross-encoder/ms-marco-MiniLM-L-6-v2 for high-speed, local re-ranking
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"[*] Initializing localized Cross-Encoder Pass: {model_name}...")
        self.re_ranker = CrossEncoder(model_name)

    def retrieve_and_rerank(self, question: str) -> List[Dict[str, Any]]:
        """
        Executes a two-stage retrieval pass: Over-retrieves k=15 frames from 
        Chroma DB, then applies cross-encoder re-ranking down to the top 4 contexts.
        """
        print(f"\n[*] Stage 1: Over-retrieving context frames (k=15) for query: '{question}'...")
        # Step 1: Broad semantic recall retrieval pass (k=15)
        retriever = self.chroma_client.get_retriever(search_kwargs={"k": 15})
        initial_docs = retriever.invoke(question)
        
        if not initial_docs:
            print("[!] Warn: Initial vector search returned 0 documents.")
            return []

        # Step 2: Prepare inputs for Cross-Encoder scoring pairs: (Query, Document_Text)
        pairs = [[question, doc.page_content] for doc in initial_docs]
        
        print("[*] Stage 2: Running cross-encoder pass to score contextual relevance...")
        scores = self.re_ranker.predict(pairs)
        
        # Merge scores back with original document metadata elements
        scored_docs = []
        for idx, score in enumerate(scores):
            scored_docs.append({
                "score": float(score),
                "page_content": initial_docs[idx].page_content,
                "metadata": initial_docs[idx].metadata
            })
        
        # Sort documents in descending order based on their cross-encoder ranking scores
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Compress the payload down to the top 4 highly targeted context frames
        top_k_final = scored_docs[:4]
        print(f"[+] Re-ranking finalized. Retained top {len(top_k_final)} highly targeted context frames.")
        
        return top_k_final
