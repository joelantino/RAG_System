"""
FULLY CORRECTED RAG PIPELINE - NO TRUNCATION
===============================================
This script fixes ALL issues:
- ✅ All imports included
- ✅ Full document content (NO 1000 char limit)
- ✅ 100K context window (not 6K)
- ✅ Better retrieval (10 chunks)

Run this script OR copy cells into your notebook.
"""

# =============================================================================
# Phase 1: Setup & Environment
# =============================================================================

import sys
import platform
import os
from dotenv import load_dotenv

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("OS:", platform.system())

load_dotenv()

print("\nEnvironment variables loaded:")
print("GOOGLE_API_KEY loaded:", bool(os.getenv("GOOGLE_API_KEY")))
print("PINECONE_API_KEY loaded:", bool(os.getenv("PINECONE_API_KEY")))

# =============================================================================
# Phase 2: Document Ingestion - WITH FULL PATH IMPORT
# =============================================================================

from pathlib import Path  # ✅ FIXED: Added missing import
from langchain_community.document_loaders import PyPDFLoader

PDF_DIR = Path("data/pdfs")
assert PDF_DIR.exists(), "data/pdfs folder does not exist"

pdf_files = sorted(PDF_DIR.glob("*.pdf"))

print(f"\n📁 Found {len(pdf_files)} PDF files:")
for f in pdf_files:
    print(f"  - {f.name}")

assert len(pdf_files) > 0, "No PDFs found. Add files to data/pdfs/"

# Load all PDFs
documents = []

for pdf_path in pdf_files:
    print(f"\nLoading: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"  Pages extracted: {len(pages)}")
    documents.extend(pages)

print(f"\n✅ Total pages loaded: {len(documents)}")

# Validation
total_chars = sum(len(d.page_content) for d in documents)
print(f"Total characters: {total_chars:,}")
print(f"Average chars per page: {total_chars // len(documents):,}")

# =============================================================================
# Phase 3: Text Normalization - WITH RE IMPORT
# =============================================================================

import re  # ✅ FIXED: Added missing import
from langchain_text_splitters import RecursiveCharacterTextSplitter

def normalize_text(text: str) -> str:
    """Normalize extracted text"""
    text = re.sub(r"\n+", "\n", text)  # Collapse newlines
    text = re.sub(r"\s+", " ", text)   # Collapse whitespace
    text = text.strip()
    return text

# Normalize all documents
for doc in documents:
    doc.page_content = normalize_text(doc.page_content)

print("✅ Text normalization complete")

# Create chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Created {len(chunks)} chunks from {len(documents)} pages")

# Chunk statistics
sizes = [len(c.page_content) for c in chunks]
print(f"Chunk sizes - Min: {min(sizes)}, Max: {max(sizes)}, Avg: {sum(sizes)//len(sizes)}")

# =============================================================================
# Phase 4: Embedding Generation
# =============================================================================

from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"\n📥 Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ Model loaded")

texts = [chunk.page_content for chunk in chunks]

print(f"\n🔄 Generating embeddings for {len(texts)} chunks...")
embeddings = embedder.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"✅ Embeddings generated: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# =============================================================================
# Phase 5: Pinecone Upload - FULL CONTENT (NO TRUNCATION)
# =============================================================================

from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY is not None, "Missing PINECONE_API_KEY"

pc = Pinecone(api_key=PINECONE_API_KEY)
print("✅ Pinecone client initialized")

INDEX_NAME = "notebooklm-rag-antigravity"

existing_indexes = [i["name"] for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating new serverless index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=embeddings.shape[1],
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Using existing index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# ✅ FIXED: Store FULL content (NOT [:1000])
vectors = []

for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
    vectors.append({
        "id": f"chunk-{i}",
        "values": vector.tolist(),
        "metadata": {
            "source": chunk.metadata.get("source", ""),
            "page": chunk.metadata.get("page", ""),
            "text": chunk.page_content  # ✅ FULL CONTENT - NO TRUNCATION!
        }
    })

print(f"✅ Prepared {len(vectors)} vectors with FULL content")

# Upload in batches
BATCH_SIZE = 100

for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(vectors=batch)
    print(f"  Uploaded {min(i + BATCH_SIZE, len(vectors))} / {len(vectors)} vectors")

print("✅ All vectors uploaded to Pinecone")

# Verify
stats = index.describe_index_stats()
print(f"\n📊 Index stats: {stats['total_vector_count']} vectors")

# =============================================================================
# Phase 6: Semantic Retrieval
# =============================================================================

def embed_query(query: str) -> np.ndarray:
    """Embed a query string"""
    vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vec[0]

def retrieve_top_k(query: str, k: int = 10):  # ✅ FIXED: Default k=10 (not 5)
    """Retrieve top-k relevant chunks"""
    query_vec = embed_query(query)
    
    result = index.query(
        vector=query_vec.tolist(),
        top_k=k,
        include_metadata=True
    )
    
    return result["matches"]

print("✅ Retrieval functions ready")

# =============================================================================
# Phase 7: RAG Answer Generation - FULL CONTEXT
# =============================================================================

import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY is not None, "Missing GOOGLE_API_KEY"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")  # Working model from list

print("✅ Gemini initialized")

# ✅ FIXED: 100K char context (NOT 6K)
def build_context(retrieved_matches, max_chars=100000):  
    """Build context from retrieved chunks - FULL CONTEXT"""
    context_blocks = []
    total_chars = 0

    for match in retrieved_matches:
        text = match["metadata"].get("text", "")  # Full text, not truncated
        source = match["metadata"].get("source", "")
        page = match["metadata"].get("page", "")

        block = f"[Source: {source}, Page: {page}]\n{text}\n"
        
        if total_chars + len(block) > max_chars:
            break

        context_blocks.append(block)
        total_chars += len(block)
    
    print(f"  📝 Context built: {total_chars:,} chars from {len(context_blocks)} chunks")
    return "\n---\n".join(context_blocks)


def build_rag_prompt(context: str, question: str) -> str:
    """Build grounded RAG prompt"""
    prompt = f"""
You are a factual assistant.

Answer the question strictly using ONLY the context below.
If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt.strip()


def rag_answer(query: str, k: int = 10):  # ✅ FIXED: k=10 default
    """Full RAG pipeline: retrieve + generate"""
    print(f"\n🔍 Processing: '{query}'")
    
    # Retrieve
    retrieved = retrieve_top_k(query, k=k)
    print(f"  📚 Retrieved {len(retrieved)} chunks")
    
    # Build context
    context = build_context(retrieved)
    
    # Build prompt
    prompt = build_rag_prompt(context, query)
    
    # Generate
    print(f"  🤖 Generating answer...")
    response = model.generate_content(prompt)
    
    return {
        "query": query,
        "answer": response.text,
        "context": context,
        "retrieved": retrieved
    }

print("✅ RAG pipeline ready!\n")

# =============================================================================
# Phase 8: Interactive Q&A Mode
# =============================================================================

print("="*70)
print("🤖 INTERACTIVE Q&A MODE")
print("="*70)
print("\nYour RAG system is ready! Ask questions about your lecture PDFs.")
print("Type 'exit' or 'quit' to end the session.\n")
print("💡 All fixes applied:")
print("   ✅ Full document content loaded (no 1000 char limit)")
print("   ✅ 100K context window (not 6K)")  
print("   ✅ 10 chunks retrieved (not 5)")
print("   ✅ All imports fixed")
print("="*70)

while True:
    # Get user input
    user_question = input("\n❓ Your question: ").strip()
    
    # Check for exit command
    if user_question.lower() in ['exit', 'quit', 'q']:
        print("\n👋 Thanks for using the RAG system! Goodbye!\n")
        break
    
    # Skip empty questions
    if not user_question:
        continue
    
    try:
        # Get answer
        result = rag_answer(user_question, k=10)
        
        # Display answer prominently
        print("\n" + "="*70)
        print("📝 ANSWER:")
        print("="*70)
        print(f"\n{result['answer']}\n")
        print("="*70)
        
        # Show sources below
        print(f"\n📚 Sources Used (Top 3):")
        print("-" * 70)
        for i, match in enumerate(result['retrieved'][:3], 1):
            score = match['score']
            source = match['metadata'].get('source', 'N/A')
            page = match['metadata'].get('page', 'N/A')
            print(f"  {i}. {source} (Page {page}) - Relevance: {score:.4f}")
        print("-" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please try another question.\n")
