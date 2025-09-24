"""
Document Splitter Module

This module handles splitting PDF documents into pages and chunks for vector storage.
Supports multiple splitting strategies for optimal RAG retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pdfplumber
import re
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    content: str
    page_number: int
    chunk_index: int
    document_name: str
    chunk_type: str  # 'page', 'paragraph', 'section'
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DocumentSplitter:
    """Handles splitting documents into manageable chunks for vector storage"""
    
    def __init__(self):
        """Initialize the document splitter for page-based splitting only"""
        self.logger = logging.getLogger(__name__)
    
    def split_pdf_by_pages(self, pdf_path: Path) -> List[DocumentChunk]:
        """
        Split PDF document into individual pages
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of DocumentChunk objects, one per page
        """
        chunks = []
        document_name = pdf_path.name
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    
                    if text.strip():  # Only create chunk if page has content
                        chunk = DocumentChunk(
                            content=text.strip(),
                            page_number=page_num,
                            chunk_index=0,  # Single chunk per page
                            document_name=document_name,
                            chunk_type='page',
                            metadata={
                                'total_pages': len(pdf.pages),
                                'page_bbox': page.bbox if hasattr(page, 'bbox') else None
                            }
                        )
                        chunks.append(chunk)
                        
        except Exception as e:
            self.logger.error(f"Error splitting PDF {pdf_path}: {e}")
            return []
        
        self.logger.info(f"Split {document_name} into {len(chunks)} page chunks")
        return chunks
    
    
    
    
    def get_document_summary(self, chunks: List[DocumentChunk]) -> Dict:
        """
        Generate a summary of the document chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with summary statistics
        """
        if not chunks:
            return {}
        
        total_chars = sum(len(chunk.content) for chunk in chunks)
        pages = set(chunk.page_number for chunk in chunks)
        chunk_types = {}
        
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        return {
            'document_name': chunks[0].document_name,
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'pages_covered': len(pages),
            'page_range': f"{min(pages)}-{max(pages)}" if pages else "0",
            'chunk_types': chunk_types,
            'avg_chunk_size': total_chars / len(chunks) if chunks else 0
        }

def load_and_split_documents(document_folder: Path) -> Dict[str, List[DocumentChunk]]:
    """
    Load and split all PDF documents in a folder by pages
    
    Args:
        document_folder: Path to folder containing PDF files
        
    Returns:
        Dictionary mapping document names to their page chunks
    """
    splitter = DocumentSplitter()
    document_chunks = {}
    
    pdf_files = list(document_folder.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            chunks = splitter.split_pdf_by_pages(pdf_file)
            document_chunks[pdf_file.name] = chunks
            
            # Log summary
            summary = splitter.get_document_summary(chunks)
            logging.info(f"Processed {pdf_file.name}: {summary}")
            
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {e}")
            document_chunks[pdf_file.name] = []
    
    return document_chunks

