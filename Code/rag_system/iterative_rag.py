"""
Iterative RAG Retrieval System

This module implements an iterative approach to RAG that gradually adds more context
from the document based on the relevance to the question and answer being evaluated.
"""

import json
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
    termination_reason: str
    assessment_history: List[Dict[str, Any]]

class IterativeRAGRetriever:
    """
    Implements iterative RAG retrieval that builds context incrementally
    """
    
    def __init__(self, vector_store_manager: MultiDocumentVectorStore,
                 max_context_chars: int = 8000,
                 max_iterations: int = 3,
                 min_relevance_threshold: float = 0.3,
                 model_name: str = "gpt-4",
                 confidence_threshold: float = 0.8):
        """
        Initialize the iterative RAG retriever
        
        Args:
            vector_store_manager: Manager for document vector stores
            max_context_chars: Maximum characters in final context
            max_iterations: Maximum number of retrieval iterations
            min_relevance_threshold: Minimum similarity score to include chunks
            model_name: Name of the model to use for assessments
            confidence_threshold: Threshold (0-1) for judgment confidence before stopping
        """
        self.vector_store_manager = vector_store_manager
        self.max_context_chars = max_context_chars
        self.max_iterations = max_iterations
        self.min_relevance_threshold = min_relevance_threshold
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
        # Error tracking
        self.json_parse_errors = 0
        self.assessment_errors = 0
        self.total_assessments = 0
        
        # Grace period tracking for False judgments
        self.false_judgment_grace_period = 2
        self.current_grace_attempts = 0
    
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
        Retrieve relevant chunks for given search queries with surrounding pages
        
        Args:
            document_name: Name of the document to search
            search_queries: List of queries to search for
            client: AsyncOpenAI client instance
            chunks_per_query: Number of chunks to retrieve per query
            exclude_chunks: Chunks to exclude from results (already retrieved)
            
        Returns:
            List of unique VectorSearchResult objects with surrounding pages
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
        
        # Add surrounding pages for better context
        enhanced_results, surrounding_info = await self._add_surrounding_pages(
            document_name, sorted_results, exclude_chunks, client
        )
        
        # Store surrounding pages info for later use in context summary
        self._last_surrounding_info = surrounding_info
        
        return enhanced_results
    
    async def _add_surrounding_pages(self, document_name: str, results: List[VectorSearchResult],
                                   exclude_chunks: Optional[List[DocumentChunk]], 
                                   client: AsyncOpenAI) -> Tuple[List[VectorSearchResult], Dict[str, Any]]:
        """
        Add surrounding pages (n-1, n+1) for each retrieved chunk to provide better context
        
        Args:
            document_name: Name of the document
            results: List of search results
            exclude_chunks: Chunks to exclude from results
            client: AsyncOpenAI client instance
            
        Returns:
            Enhanced results with surrounding pages
        """
        if not results:
            return results
        
        enhanced_results = list(results)  # Start with original results
        exclude_contents = {chunk.content for chunk in exclude_chunks} if exclude_chunks else set()
        
        # Get all chunks for the document to find page boundaries
        try:
            document_store = self.vector_store_manager.document_stores.get(document_name)
            if not document_store:
                return enhanced_results
            
            # Get all chunks for this document
            all_document_chunks = document_store.get_document_chunks(document_name)
            
            # Find min and max page numbers
            if not all_document_chunks:
                return enhanced_results
            
            page_numbers = [chunk.page_number for chunk in all_document_chunks]
            min_page = min(page_numbers)
            max_page = max(page_numbers)
            
            # For each result, add surrounding pages
            pages_to_add = set()
            found_pages = {result.chunk.page_number for result in results}
            
            for result in results:
                page_num = result.chunk.page_number
                surrounding_pages = []
                
                # Add n-1 and n+1 pages if they exist
                if page_num > min_page:
                    pages_to_add.add(page_num - 1)
                    surrounding_pages.append(page_num - 1)
                if page_num < max_page:
                    pages_to_add.add(page_num + 1)
                    surrounding_pages.append(page_num + 1)
                
                # Log surrounding pages for this found page
                if surrounding_pages:
                    self.logger.info(f"Found page {page_num}: adding surrounding pages {surrounding_pages}")
                else:
                    self.logger.info(f"Found page {page_num}: no surrounding pages (at document boundary)")
            
            # Add chunks from surrounding pages
            for page_num in pages_to_add:
                page_chunks = document_store.get_chunks_by_page(document_name, page_num)
                
                for chunk in page_chunks:
                    # Skip if already in results or excluded
                    if chunk.content in exclude_contents:
                        continue
                    
                    # Check if this chunk is already in our results
                    chunk_id = f"{chunk.document_name}_{chunk.page_number}_{chunk.chunk_index}"
                    already_included = any(
                        f"{r.chunk.document_name}_{r.chunk.page_number}_{r.chunk.chunk_index}" == chunk_id
                        for r in enhanced_results
                    )
                    
                    if not already_included:
                        # Create a search result for the surrounding page chunk
                        # Use a lower similarity score since it's context, not direct match
                        surrounding_result = VectorSearchResult(
                            chunk=chunk,
                            similarity_score=0.5,  # Lower score for context chunks
                            rank=len(enhanced_results)  # Assign rank based on current position
                        )
                        enhanced_results.append(surrounding_result)
                        exclude_contents.add(chunk.content)
            
            # Sort by similarity score (descending) with original results first
            enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Log summary of surrounding pages logic
            found_pages_list = sorted(found_pages)
            surrounding_pages_list = sorted(pages_to_add)
            self.logger.info(f"Surrounding pages summary: Found pages {found_pages_list} -> Added surrounding pages {surrounding_pages_list}")
            self.logger.info(f"Total enhanced chunks: {len(enhanced_results)} (original: {len(results)}, surrounding: {len(enhanced_results) - len(results)})")
            
            # Prepare surrounding pages info for context summary
            surrounding_info = {
                "found_pages": found_pages_list,
                "surrounding_pages_added": surrounding_pages_list,
                "original_chunks": len(results),
                "surrounding_chunks": len(enhanced_results) - len(results),
                "total_enhanced_chunks": len(enhanced_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error adding surrounding pages: {e}")
            return results, {"error": str(e)}  # Return original results if enhancement fails
        
        return enhanced_results, surrounding_info
    
    async def _expand_search_window(self, document_name: str, search_queries: List[str],
                                  client: AsyncOpenAI, exclude_chunks: Optional[List[DocumentChunk]] = None) -> List[VectorSearchResult]:
        """
        Expand search window by retrieving more chunks with lower relevance threshold
        
        Args:
            document_name: Name of the document to search
            search_queries: List of search queries
            client: AsyncOpenAI client instance
            exclude_chunks: Chunks to exclude from results
            
        Returns:
            List of additional chunks with lower relevance threshold
        """
        try:
            # Use a lower relevance threshold for expanded search
            original_threshold = self.min_relevance_threshold
            self.min_relevance_threshold = max(0.1, original_threshold - 0.1)  # Lower threshold
            
            # Get more chunks per query
            expanded_results = await self.retrieve_relevant_chunks(
                document_name, search_queries, client,
                chunks_per_query=5,  # More chunks per query
                exclude_chunks=exclude_chunks
            )
            
            # Restore original threshold
            self.min_relevance_threshold = original_threshold
            
            # Filter out chunks that are too similar to what we already have
            if exclude_chunks:
                exclude_contents = {chunk.content for chunk in exclude_chunks}
                filtered_results = []
                
                for result in expanded_results:
                    # Check if this chunk is too similar to existing chunks
                    is_duplicate = False
                    for existing_chunk in exclude_chunks:
                        # Simple similarity check based on content overlap
                        if self._chunks_are_similar(result.chunk.content, existing_chunk.content):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and result.chunk.content not in exclude_contents:
                        filtered_results.append(result)
                        exclude_contents.add(result.chunk.content)
                
                expanded_results = filtered_results
            
            self.logger.info(f"Expanded search found {len(expanded_results)} additional chunks")
            return expanded_results
            
        except Exception as e:
            self.logger.error(f"Error in expanded search: {e}")
            return []
    
    def _chunks_are_similar(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """
        Check if two chunks are too similar based on content overlap
        
        Args:
            content1: First chunk content
            content2: Second chunk content
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if chunks are similar enough to be considered duplicates
        """
        # Simple word-based similarity check
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold
    
    async def assess_context_sufficiency(self, current_context: str, question: str, answer: str, 
                                        client: AsyncOpenAI) -> Dict[str, Any]:
        """
        Assess the QA pair using threshold-based judgment categories
        
        Args:
            current_context: Current accumulated context
            question: The question being evaluated
            answer: The answer being evaluated
            client: AsyncOpenAI client instance
            
        Returns:
            Dict with assessment results including judgment and confidence
        """
        assessment_prompt = f"""
Based on the current context, evaluate the answer against the question using these specific judgment categories:

CONTEXT:
{current_context if current_context.strip() else "No context available yet."}

QUESTION: {question}

ANSWER: {answer}

Evaluate and respond with ONLY a JSON object:
{{
  "judgment": "True/False/Insufficient_Details/Undeterminable",
  "confidence": <numeric value 0.0 to 1.0>,
  "reasoning": "detailed explanation of your assessment",
  "supporting_evidence": ["list", "of", "evidence", "from", "context"],
  "contradicting_evidence": ["list", "of", "contradictions", "if", "any"],
  "missing_information": ["what", "details", "are", "missing", "if", "any"],
  "needs_more_context": true/false
}}

JUDGMENT CRITERIA:
- "True": The answer as it is is correct, fully supported, and no major information in the document is missing. High confidence required.
- "False": The answer has provably wrong claims or is irrelevant to the question. List specific contradictions.
- "Insufficient_Details": The answer is not as detailed as the document. The answer is correct but lacks important information that the document contains.
- "Unfinished_Research": The answer contains claims that could not yet be verified as true or false, more documents are needed.

CONFIDENCE: Rate 0.0-1.0 how certain you are of this judgment based on available evidence.

IMPORTANT: Return ONLY valid JSON. No additional text, explanations, or formatting outside the JSON object.
"""

        self.total_assessments += 1

        try:
            response = await client.responses.create(
                model=self.model_name,
                instructions="You are an expert AI assistant that evaluates whether provided context is sufficient to answer questions confidently. You must respond with valid JSON only.",
                input=assessment_prompt,
                reasoning={"effort": "low"},
                text={"verbosity": "low"}
            )
            
            # Clean the response text before parsing
            response_text = response.output_text.strip()
            
            # Try to extract JSON from the response if it's wrapped in other text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Try to find JSON object in the response
            if '{' in response_text and '}' in response_text:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                response_text = response_text[start_idx:end_idx]
            
            assessment = json.loads(response_text)
            
            # Validate required fields
            if not isinstance(assessment, dict):
                raise ValueError("Response is not a dictionary")
            
            required_fields = ["judgment", "confidence", "reasoning"]
            for field in required_fields:
                if field not in assessment:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure confidence is a number
            if not isinstance(assessment.get("confidence"), (int, float)):
                assessment["confidence"] = 0.0
            
            # Ensure judgment is a string and valid
            valid_judgments = ["True", "False", "Insufficient_Details", "Unfinished_Research"]
            judgment = assessment.get("judgment", "Unfinished_Research")
            if not isinstance(judgment, str) or judgment not in valid_judgments:
                assessment["judgment"] = "Unfinished_Research"
            
            return assessment
            
        except json.JSONDecodeError as e:
            self.json_parse_errors += 1
            self.logger.error(f"JSON parsing error in context assessment: {e}")
            self.logger.error(f"Raw response text: {response.output_text if 'response' in locals() else 'No response'}")
            self.logger.warning(f"JSON parse error count: {self.json_parse_errors}/{self.total_assessments}")
            return {
                "judgment": "Unfinished_Research",
                "confidence": 0.0,
                "reasoning": f"JSON parsing failed: {e}",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "missing_information": ["JSON parsing error occurred"],
                "needs_more_context": True
            }
        except Exception as e:
            self.assessment_errors += 1
            self.logger.error(f"Error in context assessment: {e}")
            self.logger.warning(f"Assessment error count: {self.assessment_errors}/{self.total_assessments}")
            return {
                "judgment": "Unfinished_Research",
                "confidence": 0.0,
                "reasoning": f"Assessment failed: {e}",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "missing_information": ["assessment error occurred"],
                "needs_more_context": True
            }

    async def build_context_iteratively(self, document_name: str, question: str, answer: str,
                                       client: AsyncOpenAI) -> RAGContext:
        """
        Build context iteratively by checking sufficiency after each retrieval round
        
        Args:
            document_name: Name of the document to retrieve from
            question: The question being evaluated
            answer: The answer being evaluated
            client: AsyncOpenAI client instance
            
        Returns:
            RAGContext with accumulated chunks and metadata
        """
        # Reset grace period for new QA pair
        self.reset_grace_period()
        
        accumulated_chunks = []
        accumulated_chars = 0
        all_search_queries = []
        relevance_scores = []
        termination_reason = "completed_all_iterations"  # Default reason
        assessment_history = []
        
        self.logger.info(f"Starting adaptive RAG for document: {document_name}")
        
        for iteration in range(self.max_iterations):
            # Check if we've reached the character limit
            if accumulated_chars >= self.max_context_chars:
                termination_reason = f"character_limit_reached_{accumulated_chars}_chars"
                self.logger.info(f"Reached character limit ({accumulated_chars}) at iteration {iteration}")
                break
            
            # Assess current context after retrieving information (except first iteration)
            if iteration > 0:
                current_context = self.format_context(RAGContext(
                    chunks=accumulated_chunks,
                    total_chars=accumulated_chars,
                    retrieval_iterations=iteration,
                    search_queries=all_search_queries,
                    relevance_scores=relevance_scores,
                    termination_reason="in_progress",
                    assessment_history=assessment_history
                ), include_metadata=False)
                
                assessment = await self.assess_context_sufficiency(current_context, question, answer, client)
                assessment_history.append(assessment)
                
                judgment = assessment.get("judgment", "Undeterminable")
                confidence = assessment.get("confidence", 0.0)
                
                self.logger.info(f"Assessment at iteration {iteration + 1}: "
                               f"judgment={judgment}, confidence={confidence:.3f}")
                
                # Handle False judgments with grace period (regardless of confidence)
                if judgment == "False":
                    # For False: Implement grace period system
                    if self.current_grace_attempts < self.false_judgment_grace_period:
                        self.current_grace_attempts += 1
                        termination_reason = f"false_judgment_grace_period_iteration_{iteration + 1}_attempt_{self.current_grace_attempts}_confidence_{confidence:.3f}"
                        contradictions = assessment.get("contradicting_evidence", [])
                        self.logger.info(f"False judgment - grace period attempt {self.current_grace_attempts}/{self.false_judgment_grace_period}. Contradictions: {contradictions}")
                        self.logger.info(f"Continuing search to find more relevant context...")
                        # Continue to next iteration for grace period
                    else:
                        # Grace period exhausted, stop with False judgment
                        termination_reason = f"false_judgment_confirmed_after_grace_period_iteration_{iteration + 1}_confidence_{confidence:.3f}"
                        contradictions = assessment.get("contradicting_evidence", [])
                        self.logger.info(f"False judgment confirmed after grace period - contradictions: {contradictions}")
                        break
                
                # Apply threshold-based stopping logic for other judgments
                elif confidence >= self.confidence_threshold:
                    if judgment == "True":
                        # For True: Do one more round to check for missing details
                        if iteration >= 2:  # But only if we've already done extra checking
                            termination_reason = f"true_judgment_confirmed_iteration_{iteration + 1}_confidence_{confidence:.3f}"
                            self.logger.info(f"True judgment confirmed after extra verification: {assessment['reasoning']}")
                            break
                        else:
                            self.logger.info(f"True judgment found, doing one more round to check for missing details")
                            # Continue to next iteration for verification
                    
                    elif judgment == "Insufficient_Details":
                        # For Insufficient Details: Stop and report missing information
                        termination_reason = f"insufficient_details_iteration_{iteration + 1}_confidence_{confidence:.3f}"
                        missing = assessment.get("missing_information", [])
                        self.logger.info(f"Insufficient details - missing information: {missing}")
                        break
                
                    elif judgment == "Unfinished_Research":
                        # For Unfinished Research: Stop and report need for more documents
                        termination_reason = f"unfinished_research_iteration_{iteration + 1}_confidence_{confidence:.3f}"
                        missing = assessment.get("missing_information", [])
                        self.logger.info(f"Unfinished research - need more documents: {missing}")
                        break
                
                # For Unfinished_Research or low confidence: continue searching
                if judgment == "Unfinished_Research" or confidence < self.confidence_threshold:
                    missing_info = assessment.get("missing_information", ["additional context"])
                    self.logger.info(f"Continuing search - need more info about: {missing_info}")
                    # Continue to next iteration
            
            # Generate search queries for this iteration
            if iteration == 0:
                # First iteration: broad search
                search_queries = await self.generate_search_queries(question, answer, client, iteration)
            else:
                # Later iterations: targeted search based on missing information
                if assessment_history:
                    last_assessment = assessment_history[-1]
                    missing_info = last_assessment.get("missing_information", ["additional context"])
                    
                    # Create targeted search queries based on specific missing information
                    search_queries = []
                    for info in missing_info[:3]:  # Limit to top 3 missing items
                        search_queries.extend([
                            f"{question} {info}",
                            f"Details about {info}",
                            f"{info} related to {answer[:50]}"
                        ])
                    
                    # Remove duplicates and limit
                    search_queries = list(dict.fromkeys(search_queries))[:3]
                else:
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
                termination_reason = f"no_relevant_chunks_iteration_{iteration + 1}"
                self.logger.info(f"No new relevant chunks found at iteration {iteration + 1}")
                break
            
            # Add chunks to context, respecting character limit
            iteration_chunks = []
            iteration_chars = 0
            
            for result in results:
                chunk_chars = len(result.chunk.content)
                
                # Always add complete pages - never split them
                if accumulated_chars + iteration_chars + chunk_chars <= self.max_context_chars:
                    iteration_chunks.append(result.chunk)
                    iteration_chars += chunk_chars
                    relevance_scores.append(result.similarity_score)
                else:
                    # If adding this complete page would exceed the limit, stop here
                    # We never split pages - always keep them complete
                    self.logger.info(f"Stopping at complete page {result.chunk.page_number} to respect context limit")
                    break
            
            # Check if we found relevant context and reset grace period if needed
            if iteration_chunks and iteration_chars > 200:  # Meaningful new content found
                # Check if we're in a grace period for False judgments
                if self.current_grace_attempts > 0:
                    # Reset grace period since we found relevant context
                    self.logger.info(f"Relevant context found - resetting grace period from {self.current_grace_attempts} to 0")
                    self.current_grace_attempts = 0
            
            accumulated_chunks.extend(iteration_chunks)
            accumulated_chars += iteration_chars
            
            self.logger.info(f"Iteration {iteration + 1}: Added {len(iteration_chunks)} chunks "
                           f"({iteration_chars} chars), total: {accumulated_chars} chars")
            
            # If we didn't add much content, expand search window instead of stopping
            if iteration_chars < 200:
                self.logger.info(f"Low content added ({iteration_chars} chars), expanding search window...")
                # Try to get more chunks by increasing the search scope
                expanded_results = await self._expand_search_window(
                    document_name, search_queries, client, exclude_chunks=accumulated_chunks
                )
                
                if expanded_results:
                    # Add expanded results to context
                    expanded_chunks = []
                    expanded_chars = 0
                    
                    for result in expanded_results:
                        chunk_chars = len(result.chunk.content)
                        
                        if accumulated_chars + expanded_chars + chunk_chars <= self.max_context_chars:
                            expanded_chunks.append(result.chunk)
                            expanded_chars += chunk_chars
                            relevance_scores.append(result.similarity_score)
                        else:
                            break
                    
                    if expanded_chunks:
                        accumulated_chunks.extend(expanded_chunks)
                        accumulated_chars += expanded_chars
                        iteration_chars += expanded_chars  # Update iteration chars for logging
                        
                        self.logger.info(f"Expanded search added {len(expanded_chunks)} chunks "
                                       f"({expanded_chars} chars), total: {accumulated_chars} chars")
                    else:
                        # Still no meaningful content, stop
                        termination_reason = f"low_content_after_expansion_{iteration_chars}_chars_iteration_{iteration + 1}"
                        self.logger.info(f"Still low content after expansion ({iteration_chars} chars), stopping")
                        break
                else:
                    # No expanded results found, stop
                    termination_reason = f"no_expanded_content_{iteration_chars}_chars_iteration_{iteration + 1}"
                    self.logger.info(f"No expanded content found ({iteration_chars} chars), stopping")
                break
        
        # Sort chunks by page number and chunk index for coherent reading
        accumulated_chunks.sort(key=lambda x: (x.page_number, x.chunk_index))
        
        context = RAGContext(
            chunks=accumulated_chunks,
            total_chars=accumulated_chars,
            retrieval_iterations=iteration + 1,
            search_queries=all_search_queries,
            relevance_scores=relevance_scores,
            termination_reason=termination_reason,
            assessment_history=assessment_history
        )
        
        self.logger.info(f"Built context with {len(accumulated_chunks)} chunks, "
                        f"{accumulated_chars} chars in {context.retrieval_iterations} iterations")
        self.logger.info(f"RAG search terminated: {termination_reason}")
        
        return context
    
    def reset_grace_period(self):
        """Reset grace period for new QA pair"""
        self.current_grace_attempts = 0
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            "total_assessments": self.total_assessments,
            "json_parse_errors": self.json_parse_errors,
            "assessment_errors": self.assessment_errors,
            "error_rate": (self.json_parse_errors + self.assessment_errors) / max(self.total_assessments, 1),
            "json_parse_error_rate": self.json_parse_errors / max(self.total_assessments, 1),
            "assessment_error_rate": self.assessment_errors / max(self.total_assessments, 1),
            "grace_period_attempts": self.current_grace_attempts,
            "grace_period_max": self.false_judgment_grace_period
        }
    
    def format_context(self, rag_context: RAGContext, include_metadata: bool = True) -> str:
        """
        Format the RAG context into a readable string for the LLM with clear page numbering
        
        Args:
            rag_context: The RAG context to format
            include_metadata: Whether to include chunk metadata
            
        Returns:
            Formatted context string with page order information
        """
        if not rag_context.chunks:
            return "No relevant context found."
        
        formatted_parts = []
        
        if include_metadata:
            formatted_parts.append(f"=== DOCUMENT CONTEXT ({len(rag_context.chunks)} complete pages) ===")
            formatted_parts.append(f"Retrieved in {rag_context.retrieval_iterations} iterations")
            formatted_parts.append("")
        
        # Sort chunks by page number to ensure proper reading order
        sorted_chunks = sorted(rag_context.chunks, key=lambda x: x.page_number)
        
        # Get page range information
        page_numbers = [chunk.page_number for chunk in sorted_chunks]
        min_page = min(page_numbers)
        max_page = max(page_numbers)
        
        if include_metadata:
            formatted_parts.append(f"PAGE ORDER: Pages {min_page}-{max_page} (complete pages only)")
            formatted_parts.append("")
        
        current_page = None
        for i, chunk in enumerate(sorted_chunks):
            # Add page separator if we're on a new page
            if chunk.page_number != current_page:
                if current_page is not None:
                    formatted_parts.append("")
                formatted_parts.append(f"=== PAGE {chunk.page_number} (Complete Page) ===")
                current_page = chunk.page_number
            
            # Add chunk content (this should be the complete page content)
            formatted_parts.append(chunk.content.strip())
            formatted_parts.append("")
        
        # Add reading order instruction for LLM
        if include_metadata:
            formatted_parts.append("=== READING INSTRUCTIONS ===")
            formatted_parts.append("The above pages are complete and should be read in numerical order.")
            formatted_parts.append("Each page contains the full content from that page of the document.")
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
        
        # Get final assessment if available
        final_assessment = rag_context.assessment_history[-1] if rag_context.assessment_history else None
        
        # Get surrounding pages info if available
        surrounding_info = getattr(self, '_last_surrounding_info', {})
        
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
            "character_utilization": rag_context.total_chars / self.max_context_chars,
            "termination_reason": rag_context.termination_reason,
            "confidence_threshold": self.confidence_threshold,
            "assessment_count": len(rag_context.assessment_history),
            "final_assessment": final_assessment,
            "all_assessments": rag_context.assessment_history,
            "surrounding_pages_info": surrounding_info
        }

