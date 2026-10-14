
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

# Keep the models in memory between Streamlit reruns.
@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def load_answer_generator():
    return AnswerGenerator(model_name="gemini-3.5-flash-lite")


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    try:
        pages = extract_pages(uploaded_file.getvalue())

        if not pages:
            st.error("No extractable text was found in this PDF.")
            st.stop()

        chunks = chunk_pages(pages)

        if not chunks:
            st.error("Could not create text chunks from this PDF.")
            st.stop()

        with st.spinner("Preparing document for search..."):
            embedding_model = load_embedding_model()
            retriever = Retriever(embedding_model)
            retriever.build_index(chunks)

        st.success(
            f"Document ready: {len(pages)} text pages, "
            f"{len(chunks)} chunks."
        )

        question = st.text_input(
            "Ask a question about your document",
            placeholder="What are the main findings?",
        )

        if st.button("Get answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding relevant passages..."):
                    retrieved_chunks = retriever.search(
                        question,
                        top_k=3,
                    )

                if not retrieved_chunks:
                    st.warning("No relevant passages were retrieved.")
                else:
                    try:
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
                                f"· Relevance {chunk['score']:.3f}"
                            ):
                                st.write(chunk["text"])

                    except Exception as error:
                        st.error(
                            "Could not generate an answer. "
                            "Check your Gemini configuration and try again."
                        )
                        st.exception(error)

    except Exception as error:
        st.error("Could not process this PDF.")
        st.exception(error)
