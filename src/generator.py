
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class AnswerGenerator:
    """Generate answers using retrieved document passages and Gemini."""

    def __init__(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, question: str, chunks: list[dict]) -> str:
        """Answer a question using the supplied document passages."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not chunks:
            raise ValueError("At least one context chunk is required.")

        context_parts = []

        for chunk in chunks:
            page_number = chunk["page_number"]
            text = chunk["text"]

            context_parts.append(
                f"[Source page: {page_number}]\n{text}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the document excerpts below.

Rules:
1. Do not use outside knowledge to fill gaps.
2. If the excerpts do not contain enough information, say:
   "The provided document excerpts do not contain enough
   information to answer this question."
3. Do not invent facts, quotations, or page references.
4. Keep the answer clear and concise.
5. Do not claim that something is in the document unless
   the supplied excerpts support it.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

Provide your answer based only on the supplied excerpts.
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        answer = response.text

        if not answer or not answer.strip():
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return answer.strip()
