# =============================================================================
# SOLUTION: Replace the interactive mode cell with this demo questions approach
# =============================================================================
#
# Copy this code into your notebook cell to replace the input() version
#
# This approach:
# - Works reliably in Jupyter notebooks
# - Allows easy modification of test questions
# - Can run without user interaction
# - Provides clear results
#
# =============================================================================

print("\n🤖 Running Demo Q&A Mode")
print("="*60)
print("To test different questions, modify the demo_questions list below\n")

# 📝 CUSTOMIZE YOUR QUESTIONS HERE
# Change these to match your PDF content
demo_questions = [
    "What are the main topics covered in these documents?",
    "Summarize the key findings or conclusions",
    "What methodologies are discussed?",
    "exit"
]

# Run automated Q&A
for user_question in demo_questions:
    if user_question.lower() == "exit":
        break
    
    print(f"\n❓ QUESTION: {user_question}")
    print("-" * 60)
    
    result = rag_answer(user_question, k=5)
    
    print(f"✅ ANSWER:")
    print(result['answer'])
    print("-" * 60)
    
    # Optional: Show sources
    print("\n📚 SOURCES:")
    for i, match in enumerate(result['retrieved'][:3], 1):  # Top 3 sources
        source = match['metadata'].get('source', 'Unknown')
        page = match['metadata'].get('page', 'N/A')
        score = match['score']
        print(f"  {i}. {source} (Page {page}) - Relevance: {score:.4f}")
    
    print("\n" + "="*60)

print("\n✅ Q&A Demo Complete!")
print("\n💡 TIP: For true interactive mode, use the interactive_qa.py script")
print("   Run in a separate Python terminal for proper input() support")
