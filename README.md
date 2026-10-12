# AI Research & Documentation Assistant

A simple web app that lets you upload a PDF document and ask questions about its content. The app uses Retrieval-Augmented Generation (RAG) to locate relevant passages from the document and generates accurate, source-backed answers using Google Gemini.

## Features

- **PDF upload & extraction**: Extracts readable text from uploaded PDFs while preserving page numbers.
- **Chunking & semantic search**: Splits documents into overlapping text chunks and indexes them using Sentence Transformers and FAISS.
- **Gemini-generated answers**: Produces clear, concise answers grounded strictly in the retrieved excerpts.
- **Source passages & citations**: Shows the exact passages, page numbers, and similarity scores used to answer each question.
- **Document overview**: Displays real-time metadata including page count, chunk count, and file size.

## Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web interface and interactive layout
- **PyMuPDF**: PDF text extraction and page parsing
- **Sentence Transformers**: Text embedding model (`all-MiniLM-L6-v2`)
- **FAISS**: Vector indexing and fast similarity retrieval
- **Gemini API**: Generative answer model (`gemini-3.5-flash-lite` via `google-genai`)
- **pytest**: Automated testing suite

## How It Works

1. **Extract**: PyMuPDF extracts text page by page from the uploaded PDF.
2. **Chunk & Embed**: The text is split into chunks of 1,000 characters (with 150-character overlap) and converted into vector embeddings with Sentence Transformers.
3. **Index**: The embeddings are stored in an in-memory FAISS index.
4. **Retrieve**: When you submit a question, FAISS finds the top 3 most relevant chunks.
5. **Generate**: The question and the retrieved excerpts are sent to Gemini, which returns an answer based only on the provided context.

## Screenshot

![Application Screenshot](docs/screenshot.png)

## How to Run

### 1. Set up a virtual environment
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install streamlit pymupdf sentence-transformers faiss-cpu google-genai python-dotenv pytest numpy
```

### 3. Configure the Gemini API key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Start the application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 5. Run tests
```bash
pytest -q
```

## Limitations

- Works only with text-based PDFs. Scanned documents containing only images require an OCR engine, which is not currently included.
- Uses an in-memory FAISS index, meaning documents are indexed per session rather than saved in a permanent external database.
