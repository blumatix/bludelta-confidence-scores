"""
Main QA Processor with Iterative RAG

This is the main script that replaces the Jupyter notebook functionality.
It processes QA pairs using iterative RAG for context retrieval.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .document_splitter import load_and_split_documents
from .vector_store import MultiDocumentVectorStore
from .iterative_rag import IterativeRAGRetriever
from .qa_validation_rag import QAValidatorRAG, BatchQAProcessor, QAValidationResult

class QAProcessorConfig:
    """Configuration for the QA processor"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv(override=True)
        
        # Paths
        self.document_storage = Path(os.getenv("DOCUMENT_STORAGE_QA"))
        self.docs_folder = self.document_storage / "docs"
        self.qa_file_path = self.document_storage / "DocumentQA_new.json"
        self.output_dir = Path("QAOutput")
        self.logs_dir = Path("QALogs")
        self.vector_store_dir = Path("VectorStores")
        self.system_prompt_path = "QAPrompts/system_prompt.txt"
        
        # API Configuration
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        
        # Processing Configuration
        self.concurrent_tasks = int(os.getenv("CONCURRENT_TASKS", "15"))
        self.max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
        self.max_rag_iterations = int(os.getenv("MAX_RAG_ITERATIONS", "3"))
        self.min_relevance_threshold = float(os.getenv("MIN_RELEVANCE_THRESHOLD", "0.3"))
        
        
        # Generate timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.api_key or not self.endpoint:
            print("Error: Missing Azure OpenAI API credentials")
            return False
        
        if not self.docs_folder.exists():
            print(f"Error: Documents folder not found: {self.docs_folder}")
            return False
        
        if not self.qa_file_path.exists():
            print(f"Error: QA file not found: {self.qa_file_path}")
            return False
        
        return True

class QAProcessor:
    """Main QA processor with iterative RAG"""
    
    def __init__(self, config: QAProcessorConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.endpoint)
        
        # Initialize components
        self.vector_store_manager = None
        self.rag_retriever = None
        self.qa_validator = None
        self.batch_processor = None
        
        # Data storage
        self.document_chunks = {}
        self.qa_pairs_by_document = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_file = self.config.logs_dir / f"{self.config.deployment_name}_{self.config.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
        
        # Reduce noise from HTTP libraries
        for noisy in ["httpx", "openai", "azure", "urllib3"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
        logger.info(f"QA Processor initialized with config timestamp: {self.config.timestamp}")
        return logger
    
    async def load_documents_and_qa_pairs(self):
        """Load and split documents, and load QA pairs"""
        self.logger.info("Loading documents and QA pairs...")
        
        # Load and split documents by pages
        self.logger.info(f"Splitting documents by pages")
        self.document_chunks = load_and_split_documents(self.config.docs_folder)
        
        total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())
        self.logger.info(f"Loaded {len(self.document_chunks)} documents with {total_chunks} total chunks")
        
        # Load QA pairs
        with open(self.config.qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        self.qa_pairs_by_document = {}
        for item in qa_data:
            doc_name = item["document_name"]
            self.qa_pairs_by_document[doc_name] = item["qa_pairs"]
        
        total_qa_pairs = sum(len(pairs) for pairs in self.qa_pairs_by_document.values())
        self.logger.info(f"Loaded QA pairs for {len(self.qa_pairs_by_document)} documents")
        self.logger.info(f"Total QA pairs: {total_qa_pairs}")
    
    async def setup_vector_stores(self, force_rebuild: bool = False):
        """Setup vector stores for documents"""
        self.logger.info("Setting up vector stores...")
        
        # Check if vector stores already exist
        store_index_path = self.config.vector_store_dir / "store_index.json"
        
        if store_index_path.exists() and not force_rebuild:
            self.logger.info("Loading existing vector stores...")
            self.vector_store_manager = MultiDocumentVectorStore()
            try:
                self.vector_store_manager.load_all(self.config.vector_store_dir)
                self.logger.info("Successfully loaded existing vector stores")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load existing vector stores: {e}")
                self.logger.info("Will rebuild vector stores...")
        
        # Create new vector stores
        self.logger.info("Creating new vector stores...")
        self.vector_store_manager = MultiDocumentVectorStore()
        
        await self.vector_store_manager.create_all_stores(self.document_chunks, self.client)
        
        # Save vector stores
        self.vector_store_manager.save_all(self.config.vector_store_dir)
        self.logger.info("Vector stores created and saved")
    
    def setup_rag_components(self):
        """Setup RAG retriever and QA validator"""
        self.logger.info("Setting up RAG components...")
        
        # Initialize RAG retriever
        self.rag_retriever = IterativeRAGRetriever(
            vector_store_manager=self.vector_store_manager,
            max_context_chars=self.config.max_context_chars,
            max_iterations=self.config.max_rag_iterations,
            min_relevance_threshold=self.config.min_relevance_threshold
        )
        
        # Initialize QA validator
        self.qa_validator = QAValidatorRAG(
            rag_retriever=self.rag_retriever,
            system_prompt_path=self.config.system_prompt_path,
            deployment_name=self.config.deployment_name,
            concurrent_limit=self.config.concurrent_tasks
        )
        
        # Initialize batch processor
        self.batch_processor = BatchQAProcessor(
            qa_validator=self.qa_validator,
            concurrent_tasks=self.config.concurrent_tasks
        )
        
        self.logger.info("RAG components setup completed")
    
    async def process_qa_pairs(self, max_docs: Optional[int] = None, 
                              max_questions_per_doc: Optional[int] = None) -> List[QAValidationResult]:
        """Process QA pairs using iterative RAG"""
        self.logger.info(f"Starting QA processing (max_docs={max_docs}, max_questions={max_questions_per_doc})")
        
        # Filter documents that have vector stores
        available_documents = {
            doc_name: qa_pairs 
            for doc_name, qa_pairs in self.qa_pairs_by_document.items()
            if self.vector_store_manager.get_document_store(doc_name) is not None
        }
        
        if not available_documents:
            self.logger.error("No documents with vector stores found!")
            return []
        
        self.logger.info(f"Processing {len(available_documents)} documents with vector stores")
        
        # Process all QA pairs
        results = await self.batch_processor.process_all_documents(
            qa_pairs_by_document=available_documents,
            client=self.client,
            max_docs=max_docs,
            max_questions_per_doc=max_questions_per_doc
        )
        
        return results
    
    def save_results(self, results: List[QAValidationResult], filename: Optional[str] = None):
        """Save processing results to file"""
        if filename is None:
            filename = f"qa_results_rag_{self.config.deployment_name}_{self.config.timestamp}.json"
        
        output_path = self.config.output_dir / filename
        
        # Prepare results for JSON serialization
        serializable_results = []
        for result in results:
            result_dict = {
                "document_name": result.document_name,
                "question": result.question,
                "answer": result.answer,
                "judgment": result.judgment,
                "reasoning": result.reasoning,
                "confidence_numeric": result.confidence_numeric,
                "confidence_verbal": result.confidence_verbal,
                "step_analysis": result.step_analysis,
                "rag_context_summary": result.rag_context_summary,
                "token_usage": result.token_usage,
                "processing_time": result.processing_time,
                "timestamp": self.config.timestamp
            }
            serializable_results.append(result_dict)
        
        # Calculate summary statistics
        total_tokens = sum(r.token_usage.get("total_tokens", 0) for r in results)
        avg_processing_time = sum(r.processing_time for r in results) / len(results) if results else 0
        avg_confidence = sum(r.confidence_numeric for r in results) / len(results) if results else 0
        
        # Create output data
        output_data = {
            "metadata": {
                "timestamp": self.config.timestamp,
                "model": self.config.deployment_name,
                "total_results": len(serializable_results),
                "concurrent_tasks": self.config.concurrent_tasks,
                "max_context_chars": self.config.max_context_chars,
                "max_rag_iterations": self.config.max_rag_iterations,
                "processing_summary": {
                    "total_tokens": total_tokens,
                    "avg_processing_time": avg_processing_time,
                    "avg_confidence": avg_confidence
                }
            },
            "results": serializable_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_sample_results(self, results: List[QAValidationResult], num_samples: int = 3):
        """Print sample results for verification"""
        print(f"\n=== SAMPLE RESULTS ({len(results)} total) ===")
        
        for i, result in enumerate(results[:num_samples]):
            print(f"\n--- Result {i+1} ---")
            print(f"Document: {result.document_name}")
            print(f"Question: {result.question[:100]}...")
            print(f"Judgment: {result.judgment}")
            print(f"Confidence: {result.confidence_numeric:.2f} ({result.confidence_verbal})")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Context Summary: {result.rag_context_summary}")
            print(f"Reasoning: {result.reasoning[:200]}...")
    
    async def run_full_pipeline(self, max_docs: Optional[int] = None,
                               max_questions_per_doc: Optional[int] = None,
                               force_rebuild_vectors: bool = False):
        """Run the complete QA processing pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Load documents and QA pairs
            await self.load_documents_and_qa_pairs()
            
            # Step 2: Setup vector stores
            await self.setup_vector_stores(force_rebuild=force_rebuild_vectors)
            
            # Step 3: Setup RAG components
            self.setup_rag_components()
            
            # Step 4: Process QA pairs
            results = await self.process_qa_pairs(max_docs, max_questions_per_doc)
            
            # Step 5: Save results
            output_path = self.save_results(results)
            
            # Step 6: Print summary
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
            self.logger.info(f"Processed {len(results)} QA pairs successfully")
            
            # Print sample results
            self.print_sample_results(results)
            
            return results, output_path
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

async def main():
    """Main entry point"""
    # Initialize configuration
    config = QAProcessorConfig()
    
    if not config.validate():
        return
    
    # Initialize processor
    processor = QAProcessor(config)
    
    # Configuration for this run
    MAX_DOCS = 2  # Process first 2 documents for testing
    MAX_QUESTIONS_PER_DOC = 5  # Process 5 questions per document
    FORCE_REBUILD_VECTORS = False  # Set to True to rebuild vector stores
    
    print(f"Starting QA processing with iterative RAG...")
    print(f"Configuration: {MAX_DOCS} docs, {MAX_QUESTIONS_PER_DOC} questions per doc")
    print(f"Max context chars: {config.max_context_chars}")
    print(f"Concurrent tasks: {config.concurrent_tasks}")
    
    try:
        results, output_path = await processor.run_full_pipeline(
            max_docs=MAX_DOCS,
            max_questions_per_doc=MAX_QUESTIONS_PER_DOC,
            force_rebuild_vectors=FORCE_REBUILD_VECTORS
        )
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📊 Processed {len(results)} QA pairs")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        logging.error(f"Processing failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
