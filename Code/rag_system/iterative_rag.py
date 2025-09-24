"""
Iterative RAG Retrieval System

This module implements an iterative approach to RAG that gradually adds more context
from the document based on the relevance to the question and answer being evaluated.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI

from .document_splitter import DocumentChunk
from .vector_store import MultiDocumentVectorStore, VectorSearchResult

@dataclass
class RAGContext:
    """Represents the context retrieved for RAG"""
    chunks: List[DocumentChunk]
    total_chars: int
    retrieval_iterations: int
    search_queries: List[str]
    relevance_scores: List[float]

class IterativeRAGRetriever:
    """
    Implements iterative RAG retrieval that builds context incrementally
    """
    
    def __init__(self, vector_store_manager: MultiDocumentVectorStore,
                 max_context_chars: int = 8000,
                 max_iterations: int = 3,
                 min_relevance_threshold: float = 0.3):
        """
        Initialize the iterative RAG retriever
        
        Args:
            vector_store_manager: Manager for document vector stores
            max_context_chars: Maximum characters in final context
            max_iterations: Maximum number of retrieval iterations
            min_relevance_threshold: Minimum similarity score to include chunks
        """
        self.vector_store_manager = vector_store_manager
        self.max_context_chars = max_context_chars
        self.max_iterations = max_iterations
        self.min_relevance_threshold = min_relevance_threshold
        self.logger = logging.getLogger(__name__)
    
    async def generate_search_queries(self, question: str, answer: str, 
                                     client: AsyncOpenAI, iteration: int = 0) -> List[str]:
        """
        Generate search queries based on the question and answer being evaluated
        
        Args:
            question: The question being asked
            answer: The answer being evaluated
            client: AsyncOpenAI client instance
            iteration: Current iteration number (affects query generation strategy)
            
        Returns:
            List of search query strings
        """
        if iteration == 0:
            # First iteration: focus on the main question
            queries = [
                question,
                f"Information related to: {question}",
                # Extract key terms from question
                " ".join([word for word in question.split() if len(word) > 3])
            ]
        elif iteration == 1:
            # Second iteration: focus on answer content and verification
            queries = [
                f"Facts mentioned in: {answer[:200]}",
                f"Details about: {answer[:100]}",
                question,  # Still include original question
            ]
        else:
            # Later iterations: look for specific details and contradictions
            queries = [
                f"Additional context for: {question}",
                f"Related information: {answer[:150]}",
                "Supporting evidence or contradicting information"
            ]
        
        # Remove empty queries and duplicates
        queries = [q.strip() for q in queries if q.strip()]
        queries = list(dict.fromkeys(queries))  # Remove duplicates while preserving order
        
        return queries
    
    async def retrieve_relevant_chunks(self, document_name: str, search_queries: List[str],
                                     client: AsyncOpenAI, chunks_per_query: int = 3,
                                     exclude_chunks: Optional[List[DocumentChunk]] = None) -> List[VectorSearchResult]:
        """
        Retrieve relevant chunks for given search queries
        
        Args:
            document_name: Name of the document to search
            search_queries: List of queries to search for
            client: AsyncOpenAI client instance
            chunks_per_query: Number of chunks to retrieve per query
            exclude_chunks: Chunks to exclude from results (already retrieved)
            
        Returns:
            List of unique VectorSearchResult objects
        """
        all_results = []
        exclude_contents = set()
        
        if exclude_chunks:
            exclude_contents = {chunk.content for chunk in exclude_chunks}
        
        for query in search_queries:
            try:
                results = await self.vector_store_manager.search_document(
                    document_name, query, client, top_k=chunks_per_query * 2  # Get more to filter
                )
                
                # Filter out chunks we already have and low relevance scores
                filtered_results = []
                for result in results:
                    if (result.chunk.content not in exclude_contents and 
                        result.similarity_score >= self.min_relevance_threshold):
                        filtered_results.append(result)
                        exclude_contents.add(result.chunk.content)  # Prevent duplicates within this query
                
                # Take top chunks for this query
                all_results.extend(filtered_results[:chunks_per_query])
                
            except Exception as e:
                self.logger.error(f"Error searching with query '{query}': {e}")
        
        # Remove duplicates and sort by relevance
        unique_results = {}
        for result in all_results:
            chunk_id = f"{result.chunk.document_name}_{result.chunk.page_number}_{result.chunk.chunk_index}"
            if chunk_id not in unique_results or result.similarity_score > unique_results[chunk_id].similarity_score:
                unique_results[chunk_id] = result
        
        # Sort by similarity score (descending)
        sorted_results = sorted(unique_results.values(), key=lambda x: x.similarity_score, reverse=True)
        
        return sorted_results
    
    async def build_context_iteratively(self, document_name: str, question: str, answer: str,
                                       client: AsyncOpenAI) -> RAGContext:
        """
        Build context iteratively by performing multiple retrieval rounds
        
        Args:
            document_name: Name of the document to retrieve from
            question: The question being evaluated
            answer: The answer being evaluated
            client: AsyncOpenAI client instance
            
        Returns:
            RAGContext with accumulated chunks and metadata
        """
        accumulated_chunks = []
        accumulated_chars = 0
        all_search_queries = []
        relevance_scores = []
        
        self.logger.info(f"Starting iterative RAG for document: {document_name}")
        
        for iteration in range(self.max_iterations):
            # Check if we've reached the character limit
            if accumulated_chars >= self.max_context_chars:
                self.logger.info(f"Reached character limit ({accumulated_chars}) at iteration {iteration}")
                break
            
            # Generate search queries for this iteration
            search_queries = await self.generate_search_queries(question, answer, client, iteration)
            all_search_queries.extend(search_queries)
            
            self.logger.info(f"Iteration {iteration + 1}: Searching with {len(search_queries)} queries")
            
            # Retrieve relevant chunks
            results = await self.retrieve_relevant_chunks(
                document_name, search_queries, client,
                chunks_per_query=2 if iteration == 0 else 1,  # More chunks in first iteration
                exclude_chunks=accumulated_chunks
            )
            
            if not results:
                self.logger.info(f"No new relevant chunks found at iteration {iteration + 1}")
                break
            
            # Add chunks to context, respecting character limit
            iteration_chunks = []
            iteration_chars = 0
            
            for result in results:
                chunk_chars = len(result.chunk.content)
                
                if accumulated_chars + iteration_chars + chunk_chars <= self.max_context_chars:
                    iteration_chunks.append(result.chunk)
                    iteration_chars += chunk_chars
                    relevance_scores.append(result.similarity_score)
                else:
                    # Partial chunk to fill remaining space
                    remaining_chars = self.max_context_chars - accumulated_chars - iteration_chars
                    if remaining_chars > 100:  # Only add if we have meaningful space
                        partial_content = result.chunk.content[:remaining_chars] + "..."
                        partial_chunk = DocumentChunk(
                            content=partial_content,
                            page_number=result.chunk.page_number,
                            chunk_index=result.chunk.chunk_index,
                            document_name=result.chunk.document_name,
                            chunk_type=f"{result.chunk.chunk_type}_partial",
                            metadata=result.chunk.metadata
                        )
                        iteration_chunks.append(partial_chunk)
                        iteration_chars += len(partial_content)
                        relevance_scores.append(result.similarity_score)
                    break
            
            accumulated_chunks.extend(iteration_chunks)
            accumulated_chars += iteration_chars
            
            self.logger.info(f"Iteration {iteration + 1}: Added {len(iteration_chunks)} chunks "
                           f"({iteration_chars} chars), total: {accumulated_chars} chars")
            
            # If we didn't add much content, we can stop early
            if iteration_chars < 200:
                self.logger.info(f"Low content added ({iteration_chars} chars), stopping early")
                break
        
        # Sort chunks by page number and chunk index for coherent reading
        accumulated_chunks.sort(key=lambda x: (x.page_number, x.chunk_index))
        
        context = RAGContext(
            chunks=accumulated_chunks,
            total_chars=accumulated_chars,
            retrieval_iterations=iteration + 1,
            search_queries=all_search_queries,
            relevance_scores=relevance_scores
        )
        
        self.logger.info(f"Built context with {len(accumulated_chunks)} chunks, "
                        f"{accumulated_chars} chars in {context.retrieval_iterations} iterations")
        
        return context
    
    def format_context(self, rag_context: RAGContext, include_metadata: bool = True) -> str:
        """
        Format the RAG context into a readable string for the LLM
        
        Args:
            rag_context: The RAG context to format
            include_metadata: Whether to include chunk metadata
            
        Returns:
            Formatted context string
        """
        if not rag_context.chunks:
            return "No relevant context found."
        
        formatted_parts = []
        
        if include_metadata:
            formatted_parts.append(f"=== DOCUMENT CONTEXT ({len(rag_context.chunks)} sections) ===")
            formatted_parts.append(f"Retrieved in {rag_context.retrieval_iterations} iterations")
            formatted_parts.append("")
        
        current_page = None
        for i, chunk in enumerate(rag_context.chunks):
            # Add page separator if we're on a new page
            if chunk.page_number != current_page:
                if current_page is not None:
                    formatted_parts.append("")
                formatted_parts.append(f"--- Page {chunk.page_number} ---")
                current_page = chunk.page_number
            
            # Add chunk content
            if include_metadata:
                formatted_parts.append(f"[Section {i+1}, {chunk.chunk_type}]")
            formatted_parts.append(chunk.content.strip())
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
    
    async def get_context_for_qa(self, document_name: str, question: str, answer: str,
                                client: AsyncOpenAI) -> str:
        """
        Get formatted context for QA evaluation using iterative RAG
        
        Args:
            document_name: Name of the document
            question: The question being evaluated
            answer: The answer being evaluated
            client: AsyncOpenAI client instance
            
        Returns:
            Formatted context string ready for LLM consumption
        """
        try:
            rag_context = await self.build_context_iteratively(document_name, question, answer, client)
            return self.format_context(rag_context)
        except Exception as e:
            self.logger.error(f"Error building RAG context for {document_name}: {e}")
            return f"Error retrieving context: {e}"
    
    def get_context_summary(self, rag_context: RAGContext) -> Dict[str, Any]:
        """
        Get a summary of the RAG context for analysis
        
        Args:
            rag_context: The RAG context to summarize
            
        Returns:
            Dictionary with context summary statistics
        """
        if not rag_context.chunks:
            return {"status": "no_context"}
        
        pages = sorted(set(chunk.page_number for chunk in rag_context.chunks))
        chunk_types = {}
        for chunk in rag_context.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        return {
            "status": "success",
            "total_chunks": len(rag_context.chunks),
            "total_characters": rag_context.total_chars,
            "pages_covered": pages,
            "page_count": len(pages),
            "chunk_types": chunk_types,
            "iterations_used": rag_context.retrieval_iterations,
            "search_queries_count": len(rag_context.search_queries),
            "avg_relevance_score": sum(rag_context.relevance_scores) / len(rag_context.relevance_scores) if rag_context.relevance_scores else 0,
            "character_utilization": rag_context.total_chars / self.max_context_chars
        }

if __name__ == "__main__":
    # Example usage
    import asyncio
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    
    async def test_iterative_rag():
        """Test the iterative RAG system"""
        load_dotenv()
        
        client = AsyncOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # This would normally be loaded from your vector stores
        print("Iterative RAG system initialized")
        print("To use, first create vector stores with document_splitter.py and vector_store.py")
    
    # Run test
    if Path("../.env").exists():
        asyncio.run(test_iterative_rag())
