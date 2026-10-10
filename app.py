
import streamlit as st

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("AI Research & Documentation Assistant")
st.write(
    "Upload a PDF and ask questions about its contents. "
    "Answers will be grounded in the document."
)

st.info("Project setup has been complete. RAG functionality is next up.")
