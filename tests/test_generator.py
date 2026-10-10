import pytest

from src.generator import AnswerGenerator


def make_generator():
    """Create a generator without making a real API connection."""
    generator = AnswerGenerator.__new__(AnswerGenerator)
    return generator


def test_generate_rejects_empty_question():
    generator = make_generator()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate(
            "   ",
            [{"page_number": 1, "text": "Sample document text."}],
        )


def test_generate_rejects_empty_chunks():
    generator = make_generator()

    with pytest.raises(ValueError, match="At least one context chunk"):
        generator.generate("What is this about?", [])


def test_generate_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
        AnswerGenerator(model_name="test-model")
