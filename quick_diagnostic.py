# Quick Diagnostic - Add this cell to your notebook

print("🔍 QUICK DIAGNOSTICS\n")

# 1. Check Pinecone
stats = index.describe_index_stats()
print(f"Vectors in Pinecone: {stats['total_vector_count']}")

# 2. Test simple retrieval
test_result = retrieve_top_k("lecture", k=1)
print(f"Retrieval test: {len(test_result)} results")

if test_result:
    print(f"Sample score: {test_result[0]['score']:.4f}")
    print(f"Sample text length: {len(test_result[0]['metadata'].get('text', ''))}")
    print(f"\nFirst 200 chars: {test_result[0]['metadata'].get('text', '')[:200]}")
else:
    print("❌ NO RESULTS - Retrieval broken!")
