"""
Interactive Q&A Script for RAG System
This script provides a terminal-based interface for asking questions.
Run this after executing the notebook up to Phase 7.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

def interactive_qa_mode(rag_answer_func):
    """
    Terminal-based interactive Q&A mode
    
    Args:
        rag_answer_func: The rag_answer function from the notebook
    """
    print("\n" + "="*60)
    print("🤖 Interactive RAG Q&A Mode")
    print("="*60)
    print("\nType your questions below. Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_question = input("\n❓ Your question: ").strip()
        
        # Check for exit
        if user_question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        # Skip empty questions
        if not user_question:
            continue
        
        # Get answer
        print("\n⏳ Processing...")
        try:
            result = rag_answer_func(user_question, k=5)
            
            print(f"\n✅ ANSWER:")
            print("-" * 60)
            print(result['answer'])
            print("-" * 60)
            
            # Optional: Show sources
            show_sources = input("\nShow sources? (y/n): ").strip().lower()
            if show_sources == 'y':
                print("\n📚 SOURCES:")
                for i, match in enumerate(result['retrieved'], 1):
                    source = match['metadata'].get('source', 'Unknown')
                    page = match['metadata'].get('page', 'N/A')
                    score = match['score']
                    print(f"  {i}. {source} (Page {page}) - Score: {score:.4f}")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please ensure the RAG system is fully initialized.")


if __name__ == "__main__":
    print("\n⚠️  USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. Run the Jupyter notebook up to Phase 7")
    print("2. In a notebook cell, run:")
    print("   >>> from interactive_qa import interactive_qa_mode")
    print("   >>> interactive_qa_mode(rag_answer)")
    print("\nOR run this standalone after setting up variables in __main__")
    print("="*60)
