import hashlib
import html
from pathlib import Path
import streamlit as st

from src.pdf_processor import extract_pages, chunk_pages
from src.embeddings import EmbeddingModel
from src.retriever import Retriever
from src.generator import AnswerGenerator


# Page configuration
st.set_page_config(
    page_title="AI Research & Documentation Assistant",
    page_icon="📚",
    layout="wide",
)

# Load custom CSS
css_path = Path(__file__).parent / "src" / "styles.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Keep the models in memory between Streamlit reruns.
@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def build_retriever(_file_hash: str, chunks: list[dict]):
    embedding_model = load_embedding_model()
    retriever = Retriever(embedding_model)
    retriever.build_index(chunks)
    return retriever


@st.cache_resource
def load_answer_generator():
    return AnswerGenerator(model_name="gemini-3.5-flash-lite")


def format_file_size(size_bytes: int) -> str:
    """Format bytes into readable size matching reference mockup format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        kb = round(size_bytes / 1024)
        return f"{kb} KB (approx.)"
    else:
        mb = size_bytes / (1024 * 1024)
        return f"{mb:.1f} MB (approx.)"


# Header
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-icon">
            <svg width="46" height="46" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M6 11C6 9.067 7.567 7.5 9.5 7.5H21C22.6569 7.5 24 8.84315 24 10.5V38.5C24 37.1193 22.8807 36 21.5 36H9.5C7.567 36 6 37.567 6 39.5V11Z" fill="#B85D3B"/>
                <path d="M42 11C42 9.067 40.433 7.5 38.5 7.5H27C25.3431 7.5 24 8.84315 24 10.5V38.5C24 37.1193 25.1193 36 26.5 36H38.5C40.433 36 42 37.567 42 39.5V11Z" fill="#B85D3B" fill-opacity="0.9"/>
                <path d="M24 10.5V38.5" stroke="#FAF6F0" stroke-width="1.5"/>
            </svg>
        </div>
        <div class="app-header-text">
            <h1 class="app-header-title">AI Research & Documentation Assistant</h1>
            <p class="app-header-subtitle">Upload a PDF and ask questions about its contents. Get grounded, source-backed answers.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for persistent results and clean resetting
if "current_file_id" not in st.session_state:
    st.session_state["current_file_id"] = None
if "answer" not in st.session_state:
    st.session_state["answer"] = None
if "retrieved_chunks" not in st.session_state:
    st.session_state["retrieved_chunks"] = None

# Responsive 2-column layout: Main Content (left) and Sidebar (right)
col_main, col_sidebar = st.columns([2.3, 1.0], gap="large")

with col_main:
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF file to process and search",
    )

    pages = []
    chunks = []
    retriever = None
    file_bytes = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        file_id = f"{uploaded_file.name}_{file_hash}"

        # Clear stale results if a new file is uploaded
        if st.session_state["current_file_id"] != file_id:
            st.session_state["current_file_id"] = file_id
            st.session_state["answer"] = None
            st.session_state["retrieved_chunks"] = None

        try:
            pages = extract_pages(file_bytes)

            if not pages:
                st.error("No extractable text was found in this PDF.")
            else:
                chunks = chunk_pages(pages)

                if not chunks:
                    st.error("Could not create text chunks from this PDF.")
                else:
                    with st.spinner("Preparing document for search..."):
                        retriever = build_retriever(file_hash, chunks)

                    # Display the Uploaded File Row card
                    page_label = f"{len(pages)} page" if len(pages) == 1 else f"{len(pages)} pages"
                    st.markdown(
                        f"""
                        <div class="uploaded-file-row">
                            <div class="uploaded-file-left">
                                <div class="pdf-badge-icon">
                                    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                                    </svg>
                                    <span>PDF</span>
                                </div>
                                <div class="uploaded-file-meta">
                                    <div class="uploaded-file-name">{html.escape(uploaded_file.name)}</div>
                                    <div class="uploaded-file-sub">{page_label} • Processed successfully</div>
                                </div>
                            </div>
                            <div class="uploaded-file-status">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#207D3E" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                                Ready
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as error:
            st.error("Could not process this PDF.")

    else:
        # Reset when file is removed
        if st.session_state["current_file_id"] is not None:
            st.session_state["current_file_id"] = None
            st.session_state["answer"] = None
            st.session_state["retrieved_chunks"] = None

    # Question Input & Get Answer Form
    st.markdown(
        '<div class="question-label">Ask a question about your document</div>',
        unsafe_allow_html=True,
    )

    with st.form(key="qa_form", clear_on_submit=False):
        col_q, col_btn = st.columns([4.2, 1.4], gap="small")
        with col_q:
            question = st.text_input(
                "Ask a question about your document",
                placeholder="e.g. What are the main responsibilities?",
                label_visibility="collapsed",
            )
        with col_btn:
            submit_clicked = st.form_submit_button("✈ Get answer", type="primary")

    if submit_clicked:
        if uploaded_file is None or retriever is None:
            st.warning("Please upload a PDF document before asking questions.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding relevant passages..."):
                retrieved_chunks = retriever.search(question, top_k=3)

            if not retrieved_chunks:
                st.warning("No relevant passages were retrieved.")
            else:
                try:
                    with st.spinner("Generating answer with Gemini..."):
                        generator = load_answer_generator()
                        answer = generator.generate(question, retrieved_chunks)

                    st.session_state["answer"] = answer
                    st.session_state["retrieved_chunks"] = retrieved_chunks
                except Exception as error:
                    st.error(
                        "Could not generate an answer. Please check your Gemini configuration and try again."
                    )

    # Display Answer Card when available
    saved_answer = st.session_state.get("answer")
    saved_chunks = st.session_state.get("retrieved_chunks")

    if saved_answer:
        st.markdown(
            f"""
            <div class="content-card answer-card">
                <div class="card-header-row">
                    <div class="card-icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="#B85D3B">
                            <path d="M12 2L14.2 8.3L20.5 10.5L14.2 12.7L12 19L9.8 12.7L3.5 10.5L9.8 8.3L12 2Z"/>
                        </svg>
                    </div>
                    <div>
                        <h2 class="card-title">Answer</h2>
                    </div>
                </div>
                <div class="answer-body">{html.escape(saved_answer)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display Sources Section when available
    if saved_chunks:
        st.markdown(
            """
            <div class="content-card sources-card" style="margin-bottom: 0.75rem;">
                <div class="card-header-row" style="margin-bottom: 0;">
                    <div class="card-icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#B85D3B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                            <line x1="10" y1="9" x2="8" y2="9"></line>
                        </svg>
                    </div>
                    <div>
                        <h2 class="card-title">Sources</h2>
                        <p class="card-subtitle">These passages were retrieved from your document to support the answer.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for idx, chunk in enumerate(saved_chunks, start=1):
            expander_label = f"Source {idx}  ·  Page {chunk['page_number']}  ·  Similarity: {chunk['score']:.3f}"
            with st.expander(expander_label):
                st.write(chunk["text"])

# Right Sidebar: Document Overview Card
with col_sidebar:
    if uploaded_file is not None and retriever is not None:
        file_size_display = format_file_size(len(file_bytes)) if file_bytes else "—"
        st.markdown(
            f"""
            <div class="overview-card">
                <div class="overview-header">
                    <div class="overview-header-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#B85D3B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                            <line x1="10" y1="9" x2="8" y2="9"></line>
                        </svg>
                    </div>
                    <h3 class="overview-title">Document Overview</h3>
                </div>
                <div class="overview-list">
                    <div class="overview-row">
                        <span class="overview-key">File name</span>
                        <span class="overview-value" title="{html.escape(uploaded_file.name)}">{html.escape(uploaded_file.name)}</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Total pages</span>
                        <span class="overview-value">{len(pages)}</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Text chunks</span>
                        <span class="overview-value">{len(chunks)}</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Status</span>
                        <span class="overview-status-ready">
                            <span class="overview-status-dot">●</span> Ready
                        </span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">File type</span>
                        <span class="overview-value">PDF</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Size</span>
                        <span class="overview-value">{file_size_display}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Minimal empty state before upload
        st.markdown(
            """
            <div class="overview-card">
                <div class="overview-header">
                    <div class="overview-header-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#B85D3B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                            <line x1="10" y1="9" x2="8" y2="9"></line>
                        </svg>
                    </div>
                    <h3 class="overview-title">Document Overview</h3>
                </div>
                <div class="overview-list">
                    <div class="overview-row">
                        <span class="overview-key">File name</span>
                        <span class="overview-value" style="color: var(--text-muted); font-weight: 400;">—</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Total pages</span>
                        <span class="overview-value" style="color: var(--text-muted); font-weight: 400;">—</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Text chunks</span>
                        <span class="overview-value" style="color: var(--text-muted); font-weight: 400;">—</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Status</span>
                        <span class="overview-value" style="color: var(--text-muted); font-weight: 400;">Waiting for upload</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">File type</span>
                        <span class="overview-value" style="color: var(--text-muted); font-weight: 400;">PDF</span>
                    </div>
                    <div class="overview-row">
                        <span class="overview-key">Size</span>
                        <span class="overview-value" style="color: var(--text-muted); font-weight: 400;">—</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
