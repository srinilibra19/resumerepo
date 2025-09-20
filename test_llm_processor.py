"""
Test script for LLM processor functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_processor import (
    chunk_text,
    find_relevant_chunks_fallback,
    validate_query_input,
    process_query
)

def test_chunk_text():
    """Test text chunking functionality"""
    print("Testing text chunking...")
    
    # Test short text
    short_text = "This is a short text."
    chunks = chunk_text(short_text, chunk_size=50)
    print(f"Short text chunks: {len(chunks)}")
    assert len(chunks) == 1
    
    # Test long text
    long_text = "This is a longer text. " * 50  # 1150 characters
    chunks = chunk_text(long_text, chunk_size=200, overlap=20)
    print(f"Long text chunks: {len(chunks)}")
    assert len(chunks) > 1
    
    print("✅ Text chunking tests passed!")

def test_fallback_search():
    """Test fallback keyword search"""
    print("\nTesting fallback search...")
    
    document_chunks = [
        "John Smith is a software engineer with 5 years of Python experience.",
        "He has worked at Google and Microsoft on machine learning projects.",
        "His skills include JavaScript, React, and Node.js development.",
        "John graduated from MIT with a Computer Science degree."
    ]
    
    query = "Python experience"
    results = find_relevant_chunks_fallback(query, document_chunks)
    
    print(f"Found {len(results)} relevant chunks for query: '{query}'")
    for result in results:
        print(f"  - Similarity: {result['similarity']:.3f}")
        print(f"    Text: {result['text'][:50]}...")
    
    assert len(results) > 0
    assert results[0]['similarity'] > 0
    
    print("✅ Fallback search tests passed!")

def test_query_validation():
    """Test query input validation"""
    print("\nTesting query validation...")
    
    # Valid queries
    assert validate_query_input("What is the candidate's experience?") == True
    assert validate_query_input("Python skills") == True
    
    # Invalid queries
    assert validate_query_input("") == False
    assert validate_query_input("Hi") == False  # Too short
    assert validate_query_input("x" * 1001) == False  # Too long
    assert validate_query_input(None) == False
    
    print("✅ Query validation tests passed!")

def test_process_query_fallback():
    """Test query processing with fallback method"""
    print("\nTesting query processing (fallback mode)...")
    
    document_content = """
    John Smith
    Software Engineer
    
    Experience:
    - 5 years of Python development
    - 3 years of JavaScript and React
    - Machine learning projects at Google
    - Backend development with Node.js
    
    Education:
    - MIT Computer Science degree
    - Graduated in 2018
    
    Skills:
    - Python, JavaScript, React, Node.js
    - Machine Learning, Data Science
    - AWS, Docker, Kubernetes
    """
    
    query = "What programming languages does the candidate know?"
    
    try:
        result = process_query(query, document_content)
        
        print(f"Query: {result['query']}")
        print(f"Model used: {result['model_used']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Relevant chunks: {result['relevant_chunks_count']}")
        print(f"Answer preview: {result['answer'][:100]}...")
        
        assert result['query'] == query
        assert result['confidence'] >= 0
        assert result['relevant_chunks_count'] >= 0
        assert len(result['answer']) > 0
        
        print("✅ Query processing tests passed!")
        
    except Exception as e:
        print(f"❌ Query processing test failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("Running LLM processor tests...\n")
    
    try:
        test_chunk_text()
        test_fallback_search()
        test_query_validation()
        test_process_query_fallback()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)