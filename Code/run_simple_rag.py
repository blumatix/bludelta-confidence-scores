#!/usr/bin/env python3
"""
Simplified RAG System Runner

Minimal requirements implementation.
"""

import argparse
import logging
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rag_system.simple_processor import SimpleQAProcessor

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_rag.log'),
            logging.StreamHandler()
        ]
    )

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simplified RAG System")
    parser.add_argument("--docs", type=int, default=1, help="Number of documents to process")
    parser.add_argument("--questions", type=int, default=10, help="Number of questions per document")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--surrounding-pages", type=int, default=1, help="Number of surrounding pages (n-x to n+x)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum search iterations")
    parser.add_argument("--grace-period", type=int, default=2, help="Grace period for False judgments")
    parser.add_argument("--document-name", type=str, help="Process only documents with this name (partial match)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Simplified RAG System")
    logger.info(f"Configuration: docs={args.docs}, questions={args.questions}, verbose={args.verbose}")
    logger.info(f"RAG settings: surrounding_pages={args.surrounding_pages}, max_iterations={args.max_iterations}, grace_period={args.grace_period}")
    
    # Configuration
    config = {
        "max_iterations": args.max_iterations,
        "surrounding_pages": args.surrounding_pages,
        "confidence_threshold": 0.8,
        "grace_period": args.grace_period,
        "max_documents": args.docs,
        "max_questions_per_document": args.questions,
        "document_name_filter": args.document_name
    }
    
    # Initialize processor
    processor = SimpleQAProcessor(config)
    
    # Set up paths from environment variables
    base_folder = Path(os.getenv("DOCUMENT_STORAGE_QA", "Documents"))
    documents_folder = base_folder / "docs"  # Documents are in ./docs subfolder
    qa_folder = base_folder  # QA file is in the parent folder
    
    if not documents_folder.exists():
        logger.error(f"Documents folder not found: {documents_folder}")
        return
    
    qa_file = qa_folder / "DocumentQA_new.json"
    if not qa_file.exists():
        logger.error(f"QA file not found: {qa_file}")
        return
    
    # Run pipeline
    try:
        results = await processor.run_full_pipeline(
            documents_folder=documents_folder,
            qa_folder=qa_folder,
            verbose=args.verbose
        )
        
        logger.info(f"Pipeline completed successfully")
        logger.info(f"Processed {results['total_qa_pairs']} QA pairs")
        logger.info(f"Results saved to {results['output_file']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
