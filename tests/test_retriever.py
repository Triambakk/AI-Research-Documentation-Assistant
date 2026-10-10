import numpy as np
import pytest

from src.retriever import Retriever


class FakeEmbeddingModel:
    """Return predictable vectors for testing retrieval."""

    def encode(self, texts):
        vectors = {
            "cats": [1.0, 0.0],
            "dogs": [0.0, 1.0],
            "cat question": [1.0, 0.0],
        }
        return np.array(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )


def test_search_returns_closest_chunk():
    retriever = Retriever(FakeEmbeddingModel())

    retriever.build_index([
        {"chunk_id": 0, "page_number": 1, "text": "cats"},
        {"chunk_id": 1, "page_number": 2, "text": "dogs"},
    ])

    results = retriever.search("cat question", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == 0
    assert results[0]["page_number"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_rejects_empty_question():
    retriever = Retriever(FakeEmbeddingModel())
    retriever.build_index([
        {"chunk_id": 0, "page_number": 1, "text": "cats"},
    ])

    with pytest.raises(ValueError):
        retriever.search("  ")


def test_search_requires_index():
    retriever = Retriever(FakeEmbeddingModel())

    with pytest.raises(RuntimeError):
        retriever.search("cat question")


def test_build_index_rejects_empty_chunks():
    retriever = Retriever(FakeEmbeddingModel())

    with pytest.raises(ValueError):
        retriever.build_index([])
