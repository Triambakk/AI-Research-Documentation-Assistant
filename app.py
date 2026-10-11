
import hashlib

import streamlit as st

from src.pdf_processor import extract_pages, chunk_pages
from src.embeddings import EmbeddingModel
from src.retriever import Retriever
from src.generator import AnswerGenerator


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("AI Research & Documentation Assistant")
st.write(
    "Upload a PDF and ask questions about its contents. "
    "Answers are generated using relevant passages from your document."
)


@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def load_answer_generator():
    return AnswerGenerator(model_name="gemini-3.5-flash-lite")


@st.cache_data
def process_pdf(file_bytes: bytes):
    pages = extract_pages(file_bytes)
    chunks = chunk_pages(pages)
    return pages, chunks


@st.cache_resource
def build_retriever(file_hash: str, chunks: list[dict]):
    embedding_model = load_embedding_model()
    retriever = Retriever(embedding_model)
    retriever.build_index(chunks)
    return retriever


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        with st.spinner("Processing document..."):
            pages, chunks = process_pdf(file_bytes)

        if not pages:
            st.error(
                "No extractable text was found in this PDF. "
                "Scanned PDFs may require OCR."
            )
            st.stop()

        if not chunks:
            st.error("Could not create text chunks from this PDF.")
            st.stop()

        with st.spinner("Preparing document for search..."):
            retriever = build_retriever(file_hash, chunks)

        st.success(
            f"Document ready: {len(pages)} text pages, "
            f"{len(chunks)} chunks."
        )

        question = st.text_input(
            "Ask a question about your document",
            placeholder="What are the main findings?",
            key=f"question_{file_hash}",
        )

        if st.button("Get answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Finding relevant passages..."):
                        retrieved_chunks = retriever.search(
                            question,
                            top_k=3,
                            min_score=0.35,
                        )

                    if not retrieved_chunks:
                        st.warning(
                            "I couldn't find sufficiently relevant "
                            "passages in this document to answer that "
                            "question. Try rephrasing it."
                        )
                    else:
                        with st.spinner("Generating answer with Gemini..."):
                            generator = load_answer_generator()
                            answer = generator.generate(
                                question,
                                retrieved_chunks,
                            )

                        st.subheader("Answer")
                        st.write(answer)

                        st.subheader("Retrieved source passages")

                        for chunk in retrieved_chunks:
                            with st.expander(
                                f"Page {chunk['page_number']} "
                                f"· Similarity {chunk['score']:.3f}"
                            ):
                                st.write(chunk["text"])

                except Exception as error:
                    st.error(
                        "Something went wrong while answering. "
                        "Check your Gemini configuration and terminal."
                    )
                    st.exception(error)

    except Exception as error:
        st.error("Could not process this PDF.")
        st.exception(error)
