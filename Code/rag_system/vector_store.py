"""
Vector Store Module

This module handles creating and managing vector stores for document chunks.
Supports embedding generation and similarity search for RAG retrieval.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, asdict
import asyncio
from openai import AsyncOpenAI
import faiss
from .document_splitter import DocumentChunk

@dataclass
class VectorSearchResult:
    """Represents a search result from the vector store"""
    chunk: DocumentChunk
    similarity_score: float
    rank: int

class DocumentVectorStore:
    """Vector store for document chunks with embedding-based similarity search"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small", 
                 embedding_dimension: int = 1536):
        """
        Initialize the vector store
        
        Args:
            embedding_model: OpenAI embedding model to use
            embedding_dimension: Dimension of the embedding vectors
        """
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.logger = logging.getLogger(__name__)
        
        # FAISS index for similarity search
        self.index = faiss.IndexFlatIP(embedding_dimension)  # Inner product for cosine similarity
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Metadata for persistence
        self.metadata = {
            'model': embedding_model,
            'dimension': embedding_dimension,
            'total_chunks': 0,
            'documents': {}
        }
    
    async def generate_embeddings(self, texts: List[str], client: AsyncOpenAI, 
                                 batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API
        
        Args:
            texts: List of text strings to embed
            client: AsyncOpenAI client instance
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = await client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                self.logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Fill with zero vectors as fallback
                batch_embeddings = [[0.0] * self.embedding_dimension] * len(batch_texts)
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def add_chunks(self, chunks: List[DocumentChunk], client: AsyncOpenAI):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of DocumentChunk objects to add
            client: AsyncOpenAI client instance
        """
        if not chunks:
            return
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract texts for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(texts, client)
        
        # Convert to numpy array and normalize for cosine similarity
        embedding_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embedding_array)  # Normalize for cosine similarity
        
        # Add to FAISS index
        self.index.add(embedding_array)
        
        # Store chunks and embeddings
        self.chunks.extend(chunks)
        
        if self.embeddings is None:
            self.embeddings = embedding_array
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_array])
        
        # Update metadata
        self.metadata['total_chunks'] = len(self.chunks)
        for chunk in chunks:
            doc_name = chunk.document_name
            if doc_name not in self.metadata['documents']:
                self.metadata['documents'][doc_name] = {
                    'chunks': 0,
                    'pages': set(),
                    'chunk_types': set()
                }
            
            self.metadata['documents'][doc_name]['chunks'] += 1
            self.metadata['documents'][doc_name]['pages'].add(chunk.page_number)
            self.metadata['documents'][doc_name]['chunk_types'].add(chunk.chunk_type)
        
        # Convert sets to lists for JSON serialization
        for doc_meta in self.metadata['documents'].values():
            doc_meta['pages'] = sorted(list(doc_meta['pages']))
            doc_meta['chunk_types'] = list(doc_meta['chunk_types'])
        
        self.logger.info(f"Vector store now contains {len(self.chunks)} chunks")
    
    async def search(self, query: str, client: AsyncOpenAI, 
                    top_k: int = 5, filter_doc: Optional[str] = None) -> List[VectorSearchResult]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query: Search query text
            client: AsyncOpenAI client instance
            top_k: Number of top results to return
            filter_doc: Optional document name to filter results
            
        Returns:
            List of VectorSearchResult objects
        """
        if len(self.chunks) == 0:
            return []
        
        # Generate embedding for query
        query_embeddings = await self.generate_embeddings([query], client)
        query_vector = np.array(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_vector, min(top_k * 2, len(self.chunks)))  # Get more results for filtering
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            chunk = self.chunks[idx]
            
            # Apply document filter if specified
            if filter_doc and chunk.document_name != filter_doc:
                continue
            
            result = VectorSearchResult(
                chunk=chunk,
                similarity_score=float(score),
                rank=rank
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_document_chunks(self, document_name: str) -> List[DocumentChunk]:
        """
        Get all chunks for a specific document
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of chunks for the document
        """
        return [chunk for chunk in self.chunks if chunk.document_name == document_name]
    
    def get_chunks_by_page(self, document_name: str, page_number: int) -> List[DocumentChunk]:
        """
        Get all chunks for a specific page of a document
        
        Args:
            document_name: Name of the document
            page_number: Page number
            
        Returns:
            List of chunks for the specific page
        """
        return [chunk for chunk in self.chunks 
                if chunk.document_name == document_name and chunk.page_number == page_number]
    
    def save(self, save_path: Path):
        """
        Save the vector store to disk
        
        Args:
            save_path: Directory path to save the vector store
        """
        save_path = Path(save_path)
        
        # Ensure the directory exists
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to create directory {save_path}: {e}")
            raise
        
        # Verify directory exists and is writable
        if not save_path.exists():
            raise FileNotFoundError(f"Directory was not created: {save_path}")
        
        if not save_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {save_path}")
        
        # Save FAISS index
        faiss_path = save_path / "faiss_index.idx"
        self.logger.info(f"Saving FAISS index to: {faiss_path}")
        try:
            faiss.write_index(self.index, str(faiss_path))
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index to {faiss_path}: {e}")
            raise
        
        # Save chunks and metadata
        with open(save_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(save_path / "embeddings.npy", self.embeddings)
        
        self.logger.info(f"Vector store saved to {save_path}")
    
    def load(self, load_path: Path):
        """
        Load the vector store from disk
        
        Args:
            load_path: Directory path containing the saved vector store
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store path not found: {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "faiss_index.idx"))
        
        # Load chunks
        with open(load_path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load embeddings
        embeddings_path = load_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        
        self.logger.info(f"Vector store loaded from {load_path}")
        self.logger.info(f"Loaded {len(self.chunks)} chunks from {len(self.metadata['documents'])} documents")

class MultiDocumentVectorStore:
    """Manages separate vector stores for multiple documents"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the multi-document vector store manager
        
        Args:
            embedding_model: OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self.document_stores: Dict[str, DocumentVectorStore] = {}
        self.logger = logging.getLogger(__name__)
    
    def _create_safe_filename(self, doc_name: str) -> str:
        """
        Create a safe filename from document name
        
        Args:
            doc_name: Original document name
            
        Returns:
            Safe filename for filesystem (ASCII only)
        """
        import re
        import unicodedata
        
        # Remove file extension and clean up name
        clean_name = doc_name.replace('.pdf', '').replace('.PDF', '')
        
        # Convert Unicode characters to ASCII equivalents
        # This converts ä->a, ö->o, ü->u, ß->ss, etc.
        ascii_name = unicodedata.normalize('NFKD', clean_name)
        ascii_name = ''.join(c for c in ascii_name if ord(c) < 128)
        
        # Replace any remaining problematic characters with underscores
        # Only allow alphanumeric, hyphens, underscores, and periods
        safe_name = re.sub(r'[^\w\-_\.]', '_', ascii_name)
        
        # Remove multiple underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # Limit length and remove leading/trailing underscores
        safe_name = safe_name[:50].strip('_')
        
        # Ensure it doesn't start with a dot or hyphen
        safe_name = re.sub(r'^[\.\-]+', '', safe_name)
        
        if not safe_name or len(safe_name) < 3:  # Fallback if name becomes empty or too short
            safe_name = f"document_{hash(doc_name) % 10000}"
        
        return safe_name
    
    async def create_document_store(self, document_name: str, chunks: List[DocumentChunk], 
                                   client: AsyncOpenAI) -> DocumentVectorStore:
        """
        Create a vector store for a specific document
        
        Args:
            document_name: Name of the document
            chunks: List of chunks for the document
            client: AsyncOpenAI client instance
            
        Returns:
            Created DocumentVectorStore instance
        """
        store = DocumentVectorStore(embedding_model=self.embedding_model)
        await store.add_chunks(chunks, client)
        self.document_stores[document_name] = store
        
        self.logger.info(f"Created vector store for {document_name} with {len(chunks)} chunks")
        return store
    
    async def create_all_stores(self, document_chunks: Dict[str, List[DocumentChunk]], 
                               client: AsyncOpenAI):
        """
        Create vector stores for all documents
        
        Args:
            document_chunks: Dictionary mapping document names to their chunks
            client: AsyncOpenAI client instance
        """
        self.logger.info(f"Creating vector stores for {len(document_chunks)} documents")
        
        for doc_name, chunks in document_chunks.items():
            if chunks:  # Only create store if document has chunks
                await self.create_document_store(doc_name, chunks, client)
        
        self.logger.info(f"Created {len(self.document_stores)} document vector stores")
    
    async def search_document(self, document_name: str, query: str, client: AsyncOpenAI, 
                             top_k: int = 5) -> List[VectorSearchResult]:
        """
        Search within a specific document's vector store
        
        Args:
            document_name: Name of the document to search
            query: Search query
            client: AsyncOpenAI client instance
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if document_name not in self.document_stores:
            self.logger.warning(f"No vector store found for document: {document_name}")
            return []
        
        return await self.document_stores[document_name].search(query, client, top_k)
    
    def get_document_store(self, document_name: str) -> Optional[DocumentVectorStore]:
        """Get the vector store for a specific document"""
        return self.document_stores.get(document_name)
    
    def save_all(self, base_path: Path):
        """
        Save all document vector stores
        
        Args:
            base_path: Base directory to save all stores
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for doc_name, store in self.document_stores.items():
            safe_name = self._create_safe_filename(doc_name)
            doc_path = base_path / safe_name
            self.logger.info(f"Saving vector store for '{doc_name}' to: {doc_path}")
            store.save(doc_path)
        
        # Save index of all stores
        index_data = {
            'embedding_model': self.embedding_model,
            'documents': list(self.document_stores.keys()),
            'total_stores': len(self.document_stores)
        }
        
        with open(base_path / "store_index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        self.logger.info(f"Saved {len(self.document_stores)} vector stores to {base_path}")
    
    def load_all(self, base_path: Path):
        """
        Load all document vector stores
        
        Args:
            base_path: Base directory containing saved stores
        """
        base_path = Path(base_path)
        
        # Load index
        index_path = base_path / "store_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Store index not found: {index_path}")
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        self.embedding_model = index_data['embedding_model']
        
        # Load each document store
        for doc_name in index_data['documents']:
            safe_name = self._create_safe_filename(doc_name)
            doc_path = base_path / safe_name
            if doc_path.exists():
                store = DocumentVectorStore(embedding_model=self.embedding_model)
                store.load(doc_path)
                self.document_stores[doc_name] = store
        
        self.logger.info(f"Loaded {len(self.document_stores)} vector stores from {base_path}")

if __name__ == "__main__":
    # Example usage
    import asyncio
    from dotenv import load_dotenv
    import os
    
    async def test_vector_store():
        load_dotenv()
        
        client = AsyncOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Create some test chunks
        chunks = [
            DocumentChunk("This is about artificial intelligence and machine learning.", 1, 0, "test.pdf", "page"),
            DocumentChunk("The document discusses neural networks and deep learning algorithms.", 2, 0, "test.pdf", "page"),
            DocumentChunk("Natural language processing is an important field in AI.", 3, 0, "test.pdf", "page"),
        ]
        
        # Create vector store
        store = DocumentVectorStore()
        await store.add_chunks(chunks, client)
        
        # Test search
        results = await store.search("machine learning algorithms", client, top_k=2)
        
        for result in results:
            print(f"Score: {result.similarity_score:.3f}")
            print(f"Content: {result.chunk.content}")
            print()
    
    # Run test
    if Path("../.env").exists():
        asyncio.run(test_vector_store())
