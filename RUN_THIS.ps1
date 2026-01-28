# Quick Start - Run Fixed RAG Pipeline
# =====================================

# 1. Activate your virtual environment
# .\rag-venv\Scripts\Activate.ps1

# 2. Navigate to project directory  
cd c:\Users\GCV\.gemini\antigravity\scratch\notebooklm-rag-comparison

# 3. Run the fixed pipeline
python rag_pipeline_fixed.py

# This will:
# - Load ALL your lecture PDFs with FULL content
# - Create embeddings
# - Upload to Pinecone (with complete chunks, not truncated)
# - Run 3 demo questions
# - Show you the answers with 100K context window

# Expected output:
# - "Context built: 50,000+ chars" (not 6,000!)
# - Real answers from your lecture content
# - Top sources with relevance scores
