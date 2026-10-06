import pytest
from src.graph_engine.state import RAGState


class TestRAGState:
    def test_state_has_healing_iterations_field(self):
        state = RAGState(
            question="test",
            sub_queries=[],
            hyde_context="",
            documents=[],
            generation="",
            steps=[],
            healing_iterations=0,
        )
        assert "healing_iterations" in state
        assert state["healing_iterations"] == 0

    def test_state_healing_iterations_default(self):
        state = RAGState(
            question="test",
            sub_queries=[],
            hyde_context="",
            documents=[],
            generation="",
            steps=[],
            healing_iterations=3,
        )
        assert state["healing_iterations"] == 3