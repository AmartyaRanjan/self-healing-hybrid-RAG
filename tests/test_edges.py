import pytest
from unittest.mock import MagicMock, patch
from src.graph_engine.edges import grade_generation_v_documents, MAX_HEALING_ITERATIONS
from src.graph_engine.state import RAGState


class TestGradeGeneration:
    def _make_state(self, generation="test answer", iterations=0, documents=None):
        return RAGState(
            question="test question",
            sub_queries=[],
            hyde_context="",
            documents=documents or [],
            generation=generation,
            steps=[],
            healing_iterations=iterations,
        )

    def test_empty_generation_returns_not_useful(self):
        state = self._make_state(generation="")
        result = grade_generation_v_documents(state)
        assert result == "not useful"

    def test_max_iterations_returns_useful(self):
        state = self._make_state(iterations=MAX_HEALING_ITERATIONS)
        result = grade_generation_v_documents(state)
        assert result == "useful"

    def test_over_max_iterations_returns_useful(self):
        state = self._make_state(iterations=MAX_HEALING_ITERATIONS + 5)
        result = grade_generation_v_documents(state)
        assert result == "useful"

    @patch("src.graph_engine.edges.llm")
    def test_grounded_generation_returns_useful(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = '{"score": "yes"}'
        mock_llm.invoke.return_value = mock_response
        mock_llm.__or__ = MagicMock(return_value=mock_llm)

        state = self._make_state(iterations=0)
        result = grade_generation_v_documents(state)
        assert result == "useful"

    @patch("src.graph_engine.edges.llm")
    def test_ungrounded_generation_returns_not_useful(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = '{"score": "no"}'
        mock_llm.invoke.return_value = mock_response
        mock_llm.__or__ = MagicMock(return_value=mock_llm)

        state = self._make_state(iterations=0)
        result = grade_generation_v_documents(state)
        assert result == "not useful"