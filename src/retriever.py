import faiss
import numpy as np

from src.embeddings import EmbeddingModel


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[dict]) -> None:
        """Embed chunks and build a searchable FAISS index."""
        if not chunks:
            raise ValueError("Cannot build an index without chunks")

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedding_model.encode(texts)
        vectors = np.asarray(vectors, dtype=np.float32)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the most similar chunks for a question."""
        if self.index is None:
            raise RuntimeError("Build the index before searching")

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        question_vector = self.embedding_model.encode([question])
        question_vector = np.asarray(
            question_vector, dtype=np.float32
        )

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(question_vector, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                result = self.chunks[int(index)].copy()
                result["score"] = float(score)
                results.append(result)

        return results
