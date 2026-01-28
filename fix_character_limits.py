# =============================================================================
# FIX: Increase Character Limits for Full PDF Analysis
# =============================================================================
#
# Replace the relevant cells in your notebook with these updated versions
#
# =============================================================================

# -----------------------------------------------------------------------------
# FIX 1: Phase 5 - Increase Metadata Storage
# -----------------------------------------------------------------------------
# Replace the "Prepare vectors" cell with this:

vectors = []

for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
    vectors.append({
        "id": f"chunk-{i}",
        "values": vector.tolist(),
        "metadata": {
            "source": chunk.metadata.get("source", ""),
            "page": chunk.metadata.get("page", ""),
            # UPDATED: Store full chunk text (Pinecone allows up to 40KB per metadata field)
            "text": chunk.page_content  # Full content instead of [:1000]
        }
    })

print("Prepared vectors:", len(vectors))

# -----------------------------------------------------------------------------
# FIX 2: Phase 7 - Increase Context Window for LLM
# -----------------------------------------------------------------------------
# Replace the "build_context" function cell with this:

def build_context(retrieved_matches, max_chars=100000):  # UPDATED: 100K chars
    """
    Build context from retrieved matches.
    
    Args:
        retrieved_matches: List of Pinecone matches
        max_chars: Maximum total characters (default: 100K for Gemini Flash)
        
    Note: Gemini Flash 1.5 supports up to 1M tokens (~4M chars)
          We use 100K chars as a safe default for multiple chunks
    """
    context_blocks = []
    total_chars = 0

    for match in retrieved_matches:
        text = match["metadata"].get("text", "")
        source = match["metadata"].get("source", "")
        page = match["metadata"].get("page", "")

        block = f"[Source: {source}, Page: {page}]\n{text}\n"
        
        if total_chars + len(block) > max_chars:
            break

        context_blocks.append(block)
        total_chars += len(block)

    return "\n---\n".join(context_blocks)


# -----------------------------------------------------------------------------
# FIX 3 (OPTIONAL): Retrieve More Chunks
# -----------------------------------------------------------------------------
# In your Q&A cells, increase k value for more comprehensive retrieval:

# Before:
# result = rag_answer(user_question, k=5)

# After:
result = rag_answer(user_question, k=10)  # Retrieve top 10 chunks instead of 5


# =============================================================================
# IMPORTANT NOTES:
# =============================================================================
# 1. **Re-upload to Pinecone**: After fixing the metadata truncation, you'll 
#    need to re-run Phase 5 to upload the full content to Pinecone
#
# 2. **Gemini Flash Limits**: 
#    - Context window: 1M tokens (~4M characters)
#    - We set max_chars=100K as a safe default
#    - You can increase this further if needed
#
# 3. **Chunk Size**: Your current chunk_size=1000. For longer documents,
#    consider increasing to 2000-3000 for better context per chunk
#
# 4. **Cost**: More context = more tokens = higher API costs
#    Monitor your Gemini usage after these changes
# =============================================================================
