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
