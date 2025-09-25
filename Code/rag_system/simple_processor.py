"""
Simplified QA Processor

Minimal requirements implementation with clean architecture.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .document_splitter import load_and_split_documents
from .vector_store import MultiDocumentVectorStore
from .simple_rag import SimpleRAGRetriever, SimpleRAGContext

class SimpleQAProcessor:
    """Simplified QA processor with minimal complexity"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simplified processor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set logging levels to reduce noise
        logging.getLogger("pdfminer").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.psparser").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfinterp").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfpage").setLevel(logging.WARNING)
        
        # Initialize OpenAI client with environment variables
        self.client = AsyncOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        
        # Initialize vector store manager with environment variables
        embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.vector_store_manager = MultiDocumentVectorStore(embedding_model=embedding_model)
        
        # Initialize RAG retriever
        self.rag_retriever = SimpleRAGRetriever(
            vector_store_manager=self.vector_store_manager,
            max_iterations=config.get("max_iterations", 3),
            surrounding_pages=config.get("surrounding_pages", 1),
            confidence_threshold=config.get("confidence_threshold", 0.8),
            grace_period=config.get("grace_period", 2)
        )
    
    async def process_qa_pairs(self, document_name: str, qa_pairs: List[Dict], 
                             verbose: bool = False) -> List[Dict[str, Any]]:
        """Process QA pairs for a single document concurrently"""
        self.logger.info(f"Processing {len(qa_pairs)} QA pairs for {document_name} concurrently")
        
        # Create concurrent tasks for all QA pairs
        tasks = []
        for i, qa_pair in enumerate(qa_pairs):
            task = self._process_single_qa_pair(
                document_name=document_name,
                qa_pair=qa_pair,
                pair_index=i,
                total_pairs=len(qa_pairs),
                verbose=verbose
            )
            tasks.append(task)
        
        # Execute all QA pairs concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        error_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing QA pair {i+1}: {result}")
                error_count += 1
            else:
                valid_results.append(result)
        
            self.logger.info(f"[COMPLETED] {document_name}: {len(valid_results)} successful, {error_count} errors")
        return valid_results
    
    async def _process_single_qa_pair(self, document_name: str, qa_pair: Dict, 
                                     pair_index: int, total_pairs: int, 
                                     verbose: bool = False) -> Dict[str, Any]:
        """Process a single QA pair"""
        self.logger.info(f"Processing QA pair {pair_index+1}/{total_pairs}")
        
        try:
            # Process the QA pair
            context = await self.rag_retriever.process_qa_pair(
                document_name=document_name,
                question=qa_pair["question"],
                answer=qa_pair["answer"],
                client=self.client,
                verbose=verbose
            )
            
            # Create result
            result = {
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "document_name": document_name,
                "rag_context_summary": self.rag_retriever.get_context_summary(context, verbose),
                "supporting_evidence": context.supporting_evidence,
                "contradicting_evidence": context.contradicting_evidence,
                "missing_information": context.missing_information,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            
            if verbose:
                result["rag_context"] = {
                    "content": context.content,
                    "pages": context.pages,
                    "iterations": context.iterations,
                    "judgment": context.judgment,
                    "confidence": context.confidence,
                    "reasoning": context.reasoning,
                    "supporting_evidence": context.supporting_evidence,
                    "contradicting_evidence": context.contradicting_evidence,
                    "missing_information": context.missing_information
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing QA pair {pair_index+1}: {e}")
            return {
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "document_name": document_name,
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
    
    async def _process_document_with_qa_pairs(self, document_name: str, 
                                            qa_pairs: List[Dict], 
                                            verbose: bool = False) -> List[Dict[str, Any]]:
        """Process a single document with its QA pairs"""
        self.logger.info(f"Processing document: {document_name}")
        try:
            results = await self.process_qa_pairs(document_name, qa_pairs, verbose)
            self.logger.info(f"[COMPLETED] document: {document_name} ({len(results)} QA pairs processed)")
            return results
        except Exception as e:
            self.logger.error(f"[FAILED] document: {document_name} - {e}")
            return []
    
    async def run_full_pipeline(self, documents_folder: Path, qa_folder: Path, 
                              verbose: bool = False) -> Dict[str, Any]:
        """Run the full simplified pipeline"""
        self.logger.info("Starting simplified RAG pipeline")
        
        # Load documents
        self.logger.info("Loading documents...")
        document_chunks = load_and_split_documents(documents_folder)
        
        # Create vector stores
        self.logger.info("Creating vector stores...")
        await self.vector_store_manager.create_all_stores(document_chunks, self.client)
        
        # Load QA pairs from DocumentQA_new.json
        qa_file = qa_folder / "DocumentQA_new.json"
        if not qa_file.exists():
            self.logger.error(f"QA file not found: {qa_file}")
            return {"error": f"QA file not found: {qa_file}"}
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # Process documents concurrently (with limits)
        max_documents = self.config.get("max_documents", None)
        max_questions = self.config.get("max_questions_per_document", None)
        document_name_filter = self.config.get("document_name_filter", None)
        
        self.logger.info(f"Processing documents concurrently (max_docs={max_documents}, max_questions_per_doc={max_questions}, filter={document_name_filter})...")
        
        # Create tasks for each document (with limits)
        document_tasks = []
        document_count = 0
        for document_name, chunks in document_chunks.items():
            if max_documents and document_count >= max_documents:
                self.logger.info(f"Reached document limit ({max_documents}), stopping")
                break
            if not chunks:
                continue
            
            # Apply document name filter if specified
            if document_name_filter and document_name_filter.lower() not in document_name.lower():
                self.logger.info(f"Skipping document {document_name} (doesn't match filter: {document_name_filter})")
                continue
                
            # Find QA pairs for this document
            document_qa_pairs = []
            for qa_item in qa_data:
                if qa_item.get("document_name") == document_name:
                    # Extract QA pairs from the qa_pairs array
                    for qa_pair in qa_item.get("qa_pairs", []):
                        document_qa_pairs.append({
                            "question": qa_pair.get("Question"),
                            "answer": qa_pair.get("Detailed Answer")
                        })
            
            if not document_qa_pairs:
                self.logger.warning(f"No QA pairs found for {document_name}")
                continue
            
            # Limit questions per document if specified
            if max_questions and len(document_qa_pairs) > max_questions:
                self.logger.info(f"Limiting {document_name} to {max_questions} questions (found {len(document_qa_pairs)})")
                document_qa_pairs = document_qa_pairs[:max_questions]
            
            # Create task for this document
            task = self._process_document_with_qa_pairs(
                document_name, document_qa_pairs, verbose
            )
            document_tasks.append(task)
            document_count += 1
        
        # Execute all documents concurrently
        document_results = await asyncio.gather(*document_tasks, return_exceptions=True)
        
        # Flatten results and filter exceptions
        all_results = []
        for i, result in enumerate(document_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing document {i+1}: {result}")
            else:
                all_results.extend(result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = Path("QAOutput") / f"simple_rag_results_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_file}")
        
        return {
            "total_qa_pairs": len(all_results),
            "output_file": str(output_file),
            "results": all_results
        }
