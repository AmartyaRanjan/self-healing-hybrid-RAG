import re
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.logger import get_logger

logger = get_logger(__name__)

class LexicalHighwayEngine:
    def __init__(self):
        logger.info("Initializing lexical highway (BM25)")
        self.bm25 = None
        self.raw_documents = []

    def reset(self):
        """Clears the BM25 index and raw documents so each retrieval pass starts fresh."""
        self.bm25 = None
        self.raw_documents = []

    def initialize_index(self, documents: List[Document]):
        """
        Tokenizes text frames and fits the BM25 statistical matrix locally in memory.
        """
        if not documents:
            return
        self.raw_documents = documents
        
        # Simple lowercase word tokenization
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Lexical highway indexed %d context frames", len(documents))

    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    def search(self, query: str, top_n: int = 2) -> List[Document]:
        """
        Performs ultra-low latency exact keyword tracking matches.
        """
        if not self.bm25 or not self.raw_documents:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Extract top scoring documents and assign lexical scoring metrics
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        matched_docs = []
        for idx in top_indices:
            if scores[idx] > 0.0: # Only retain actual keyword overlap hits
                doc = self.raw_documents[idx]
                # Scale the score metadata so it aligns nicely with the cross-encoder
                doc.metadata["score"] = float(scores[idx]) - 5.0  
                matched_docs.append(doc)
                
        if matched_docs:
            logger.info("Lexical highway matched %d frames via keyword overlap", len(matched_docs))
        return matched_docs
