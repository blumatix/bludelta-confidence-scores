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
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document splitter
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
    
    def split_text_by_paragraphs(self, text: str, document_name: str, page_number: int = 1) -> List[DocumentChunk]:
        """
        Split text into paragraph-based chunks
        
        Args:
            text: Text content to split
            document_name: Name of the source document
            page_number: Page number this text comes from
            
        Returns:
            List of DocumentChunk objects based on paragraphs
        """
        chunks = []
        
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size, create a new chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    page_number=page_number,
                    chunk_index=chunk_index,
                    document_name=document_name,
                    chunk_type='paragraph',
                    metadata={'paragraph_count': len(current_chunk.split('\n\n'))}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap if specified
                if self.chunk_overlap > 0:
                    # Take last part of current chunk as overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk if it has content
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                page_number=page_number,
                chunk_index=chunk_index,
                document_name=document_name,
                chunk_type='paragraph',
                metadata={'paragraph_count': len(current_chunk.split('\n\n'))}
            )
            chunks.append(chunk)
        
        return chunks
    
    def split_pdf_by_paragraphs(self, pdf_path: Path) -> List[DocumentChunk]:
        """
        Split PDF into paragraph-based chunks across all pages
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of DocumentChunk objects based on paragraphs
        """
        all_chunks = []
        document_name = pdf_path.name
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    
                    if text.strip():
                        page_chunks = self.split_text_by_paragraphs(text, document_name, page_num)
                        all_chunks.extend(page_chunks)
                        
        except Exception as e:
            self.logger.error(f"Error splitting PDF {pdf_path} by paragraphs: {e}")
            return []
        
        self.logger.info(f"Split {document_name} into {len(all_chunks)} paragraph chunks")
        return all_chunks
    
    def split_by_sections(self, text: str, document_name: str, page_number: int = 1) -> List[DocumentChunk]:
        """
        Split text by detected sections (headers, numbered lists, etc.)
        
        Args:
            text: Text content to split
            document_name: Name of the source document
            page_number: Page number this text comes from
            
        Returns:
            List of DocumentChunk objects based on sections
        """
        chunks = []
        
        # Patterns to detect section breaks
        section_patterns = [
            r'^\d+\.\s+.+$',  # Numbered sections (1. Section Title)
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
            r'^\s*[A-Z][^.!?]*:$',  # Headers ending with colon
            r'^\s*#+\s+.+$',  # Markdown-style headers
        ]
        
        lines = text.split('\n')
        current_section = []
        chunk_index = 0
        
        for line in lines:
            is_section_break = any(re.match(pattern, line.strip(), re.MULTILINE) for pattern in section_patterns)
            
            # If we found a section break and have accumulated content, create a chunk
            if is_section_break and current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    chunk = DocumentChunk(
                        content=section_text,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        document_name=document_name,
                        chunk_type='section',
                        metadata={'line_count': len(current_section)}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new section
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add the final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                chunk = DocumentChunk(
                    content=section_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    document_name=document_name,
                    chunk_type='section',
                    metadata={'line_count': len(current_section)}
                )
                chunks.append(chunk)
        
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

def load_and_split_documents(document_folder: Path, splitting_strategy: str = 'pages') -> Dict[str, List[DocumentChunk]]:
    """
    Load and split all PDF documents in a folder
    
    Args:
        document_folder: Path to folder containing PDF files
        splitting_strategy: Strategy to use ('pages', 'paragraphs', 'sections')
        
    Returns:
        Dictionary mapping document names to their chunks
    """
    splitter = DocumentSplitter()
    document_chunks = {}
    
    pdf_files = list(document_folder.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            if splitting_strategy == 'pages':
                chunks = splitter.split_pdf_by_pages(pdf_file)
            elif splitting_strategy == 'paragraphs':
                chunks = splitter.split_pdf_by_paragraphs(pdf_file)
            elif splitting_strategy == 'sections':
                # For sections, we'll split by pages first, then by sections within each page
                page_chunks = splitter.split_pdf_by_pages(pdf_file)
                chunks = []
                for page_chunk in page_chunks:
                    section_chunks = splitter.split_by_sections(
                        page_chunk.content, 
                        page_chunk.document_name, 
                        page_chunk.page_number
                    )
                    chunks.extend(section_chunks)
            else:
                logging.warning(f"Unknown splitting strategy: {splitting_strategy}, using 'pages'")
                chunks = splitter.split_pdf_by_pages(pdf_file)
            
            document_chunks[pdf_file.name] = chunks
            
            # Log summary
            summary = splitter.get_document_summary(chunks)
            logging.info(f"Processed {pdf_file.name}: {summary}")
            
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {e}")
            document_chunks[pdf_file.name] = []
    
    return document_chunks

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test document splitting
    doc_folder = Path("../Documents/docs")  # Adjust path as needed
    if doc_folder.exists():
        chunks = load_and_split_documents(doc_folder, 'paragraphs')
        
        for doc_name, doc_chunks in chunks.items():
            print(f"\n{doc_name}: {len(doc_chunks)} chunks")
            if doc_chunks:
                print(f"  First chunk: {doc_chunks[0].content[:100]}...")
