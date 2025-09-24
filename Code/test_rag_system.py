"""
Test script for the iterative RAG system

This script provides simple tests to verify the components work correctly.
"""

import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI

from document_splitter import DocumentSplitter, load_and_split_documents
from vector_store import MultiDocumentVectorStore
from iterative_rag import IterativeRAGRetriever

async def test_document_splitting():
    """Test document splitting functionality"""
    print("🔍 Testing document splitting...")
    
    # Find a test document
    docs_folder = Path("../Documents/docs")
    if not docs_folder.exists():
        print(f"❌ Documents folder not found: {docs_folder}")
        return False
    
    pdf_files = list(docs_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {docs_folder}")
        return False
    
    # Test splitting strategies
    strategies = ['pages', 'paragraphs']
    
    for strategy in strategies:
        print(f"  Testing {strategy} strategy...")
        chunks = load_and_split_documents(docs_folder, strategy)
        
        if chunks:
            doc_name = list(chunks.keys())[0]
            doc_chunks = chunks[doc_name]
            print(f"    ✅ {strategy}: {len(doc_chunks)} chunks for {doc_name}")
            
            if doc_chunks:
                sample_chunk = doc_chunks[0]
                print(f"    📄 Sample chunk: {len(sample_chunk.content)} chars, page {sample_chunk.page_number}")
        else:
            print(f"    ❌ {strategy}: No chunks created")
    
    return True

async def test_vector_store():
    """Test vector store creation and search"""
    print("🔍 Testing vector store...")
    
    load_dotenv()
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        print("❌ Missing Azure OpenAI credentials")
        return False
    
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
    
    # Load test documents
    docs_folder = Path("../Documents/docs")
    if not docs_folder.exists():
        print(f"❌ Documents folder not found: {docs_folder}")
        return False
    
    # Get a small set of chunks for testing
    chunks = load_and_split_documents(docs_folder, 'pages')
    if not chunks:
        print("❌ No document chunks loaded")
        return False
    
    # Take first document and limit to 2 chunks for testing
    first_doc = list(chunks.keys())[0]
    test_chunks = chunks[first_doc][:2]
    
    print(f"  Testing with {len(test_chunks)} chunks from {first_doc}")
    
    # Create vector store
    vector_manager = MultiDocumentVectorStore()
    await vector_manager.create_document_store(first_doc, test_chunks, client)
    
    # Test search
    test_query = "What is this document about?"
    results = await vector_manager.search_document(first_doc, test_query, client, top_k=2)
    
    if results:
        print(f"    ✅ Search returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"    📊 Result {i+1}: score={result.similarity_score:.3f}, "
                  f"content={result.chunk.content[:50]}...")
    else:
        print("    ❌ Search returned no results")
        return False
    
    return True

async def test_iterative_rag():
    """Test iterative RAG retrieval"""
    print("🔍 Testing iterative RAG...")
    
    load_dotenv()
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        print("❌ Missing Azure OpenAI credentials")
        return False
    
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
    
    # Load test documents
    docs_folder = Path("../Documents/docs")
    chunks = load_and_split_documents(docs_folder, 'paragraphs')
    
    if not chunks:
        print("❌ No document chunks loaded")
        return False
    
    # Create vector stores
    vector_manager = MultiDocumentVectorStore()
    
    # Use only first document for testing
    first_doc = list(chunks.keys())[0]
    test_chunks = chunks[first_doc][:5]  # Limit for testing
    
    await vector_manager.create_document_store(first_doc, test_chunks, client)
    
    # Create RAG retriever
    rag_retriever = IterativeRAGRetriever(
        vector_store_manager=vector_manager,
        max_context_chars=2000,  # Small for testing
        max_iterations=2
    )
    
    # Test context building
    test_question = "What are the main topics covered in this document?"
    test_answer = "The document covers various business processes and workflows."
    
    context = await rag_retriever.get_context_for_qa(first_doc, test_question, test_answer, client)
    
    if context and "Error" not in context:
        print(f"    ✅ RAG context generated: {len(context)} characters")
        print(f"    📄 Context preview: {context[:200]}...")
        
        # Get context summary
        rag_context = await rag_retriever.build_context_iteratively(first_doc, test_question, test_answer, client)
        summary = rag_retriever.get_context_summary(rag_context)
        print(f"    📊 Context summary: {summary}")
        
        return True
    else:
        print(f"    ❌ RAG context generation failed: {context}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting iterative RAG system tests...\n")
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for testing
    
    tests = [
        ("Document Splitting", test_document_splitting),
        ("Vector Store", test_vector_store),
        ("Iterative RAG", test_iterative_rag)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The iterative RAG system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and dependencies.")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
