# NotebookLM vs Custom RAG Comparison

## Phase 1: Environment & Project Setup ✅

This phase establishes a clean, reproducible project environment before implementing the RAG pipeline.

### Completed:
- Isolated Python virtual environment
- Jupyter kernel bound to project venv
- Secure API key management using `.env`
- Clean project structure and Git workflow
- Environment verification inside notebook

### Tech Stack (Planned):
- Python
- HuggingFace Sentence Transformers
- Pinecone
- Gemini Flash
- LangChain

### How to Run:
```bash
python -m venv rag-venv
rag-venv\Scripts\activate
pip install jupyter ipykernel python-dotenv
jupyter notebook
```

Then open `rag_notebook.ipynb` and select kernel `Python (rag-venv)`.

> [!NOTE]
> API keys are required in a `.env` file (not committed).

## Phase 2: Document Ingestion & Text Extraction ✅

- Loaded PDFs from `data/pdfs/`
- Extracted page-level text using LangChain
- Preserved metadata (source file and page number)
- Validated text quality and page counts
- Computed dataset statistics

## Phase 3: Text Normalization & Chunking ✅

- Normalized extracted text (whitespace & formatting)
- Removed noise from PDF parsing artifacts
- Split documents using recursive chunking strategy
- Preserved metadata for traceability
- Validated chunk size distribution

## Phase 4: Embedding Generation ✅

- Initialized HuggingFace Sentence Transformer (`all-MiniLM-L6-v2`)
- Converted all text chunks into dense vectors
- Normalized embeddings for cosine similarity
- Verified embedding dimensions and consistency
- Performed sanity checks on sample vectors
