import pytest
from src.graph_engine.lexical_engine import LexicalHighwayEngine


class TestLexicalHighwayEngine:
    def test_initially_empty(self):
        engine = LexicalHighwayEngine()
        assert engine.bm25 is None
        assert engine.raw_documents == []

    def test_initialize_index(self, sample_docs):
        engine = LexicalHighwayEngine()
        engine.initialize_index(sample_docs)
        assert engine.bm25 is not None
        assert len(engine.raw_documents) == 3

    def test_search_returns_results(self, sample_docs):
        engine = LexicalHighwayEngine()
        engine.initialize_index(sample_docs)
        results = engine.search("quick brown fox", top_n=2)
        assert len(results) > 0
        assert all(hasattr(r, "metadata") for r in results)

    def test_search_empty_index(self):
        engine = LexicalHighwayEngine()
        results = engine.search("anything")
        assert results == []

    def test_reset_clears_state(self, sample_docs):
        engine = LexicalHighwayEngine()
        engine.initialize_index(sample_docs)
        assert engine.bm25 is not None
        engine.reset()
        assert engine.bm25 is None
        assert engine.raw_documents == []

    def test_reset_allows_reindexing(self, sample_docs):
        engine = LexicalHighwayEngine()
        engine.initialize_index(sample_docs)
        engine.reset()
        engine.initialize_index(sample_docs[:1])
        assert engine.bm25 is not None
        assert len(engine.raw_documents) == 1

    def test_search_no_keyword_overlap(self, sample_docs):
        engine = LexicalHighwayEngine()
        engine.initialize_index(sample_docs)
        results = engine.search("xyzzy nonexistent words")
        assert results == []