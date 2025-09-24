"""
RAG System Package

This package contains all components for the iterative RAG-based QA validation system.
Documents are split by pages only.

Main components:
- document_splitter: Split documents into pages
- vector_store: Create and manage vector stores for similarity search
- iterative_rag: Implement iterative context retrieval
- qa_validation_rag: QA validation using RAG context
- main_qa_processor: Main orchestration script

Usage:
    from rag_system.main_qa_processor import QAProcessor, QAProcessorConfig
    
    config = QAProcessorConfig()
    processor = QAProcessor(config)
    results = await processor.run_full_pipeline()
"""

# Make key classes easily importable
from .main_qa_processor import QAProcessor, QAProcessorConfig
from .document_splitter import DocumentSplitter, load_and_split_documents
from .vector_store import MultiDocumentVectorStore, DocumentVectorStore
from .iterative_rag import IterativeRAGRetriever
from .qa_validation_rag import QAValidatorRAG, BatchQAProcessor

__version__ = "1.0.0"
__author__ = "LLM Confidence Team"

__all__ = [
    "QAProcessor",
    "QAProcessorConfig", 
    "DocumentSplitter",
    "load_and_split_documents",
    "MultiDocumentVectorStore",
    "DocumentVectorStore",
    "IterativeRAGRetriever",
    "QAValidatorRAG",
    "BatchQAProcessor"
]
