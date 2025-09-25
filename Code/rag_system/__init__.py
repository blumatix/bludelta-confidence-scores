"""
Simple RAG System Package

This package contains all components for the simplified RAG-based QA validation system.
Documents are split by pages only.

Main components:
- document_splitter: Split documents into pages
- vector_store: Create and manage vector stores for similarity search
- simple_rag: Core simplified RAG retrieval logic
- simple_processor: QA processing orchestrator

Usage:
    from rag_system.simple_processor import SimpleQAProcessor
    
    processor = SimpleQAProcessor(config)
    results = await processor.run_full_pipeline()
"""

# Make key classes easily importable
from .simple_processor import SimpleQAProcessor
from .simple_rag import SimpleRAGRetriever, SimpleRAGContext
from .document_splitter import DocumentSplitter, load_and_split_documents
from .vector_store import MultiDocumentVectorStore, DocumentVectorStore

__version__ = "1.0.0"
__author__ = "LLM Confidence Team"

__all__ = [
    "SimpleQAProcessor",
    "SimpleRAGRetriever",
    "SimpleRAGContext",
    "DocumentSplitter",
    "load_and_split_documents",
    "MultiDocumentVectorStore",
    "DocumentVectorStore"
]
