import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="The quick brown fox jumps over the lazy dog", metadata={"title": "Test Doc 1"}),
        Document(page_content="A fast brown animal leaps over a sleepy canine", metadata={"title": "Test Doc 2"}),
        Document(page_content="The weather is sunny and warm today", metadata={"title": "Test Doc 3"}),
    ]


@pytest.fixture
def mock_env():
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
        yield