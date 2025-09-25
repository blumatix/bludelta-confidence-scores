"""
Simplified RAG Retrieval System

Minimal requirements:
1. Find most relevant page using cosine similarity
2. Add n-x to n+x pages around the found page
3. Pass pages, question, and answer to LLM
4. LLM decides category: True, False, Insufficient_Details, Unfinished_Research
5. Terminate based on result
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .document_splitter import DocumentChunk
from .vector_store import MultiDocumentVectorStore, VectorSearchResult

@dataclass
class SimpleRAGContext:
    """Simplified RAG context"""
    pages: List[int]
    content: str
    total_chars: int
    iterations: int
    judgment: str
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    missing_information: List[str]

class SimpleRAGRetriever:
    """Simplified RAG retriever with minimal complexity"""
    
    def __init__(self, vector_store_manager: MultiDocumentVectorStore,
                 max_iterations: int = 3,
                 surrounding_pages: int = 1,
                 confidence_threshold: float = 0.8,
                 grace_period: int = 2):
        """
        Initialize simplified RAG retriever
        
        Args:
            vector_store_manager: Manager for document vector stores
            max_iterations: Maximum number of search iterations
            surrounding_pages: Number of pages to add around found page (n-x to n+x)
            confidence_threshold: Confidence threshold for judgments
            grace_period: Number of additional attempts for False judgments
        """
        self.vector_store_manager = vector_store_manager
        self.max_iterations = max_iterations
        self.surrounding_pages = surrounding_pages
        self.confidence_threshold = confidence_threshold
        self.grace_period = grace_period
        self.logger = logging.getLogger(__name__)
        
        # Grace period tracking
        self.current_grace_attempts = 0
    
    async def process_qa_pair(self, document_name: str, question: str, answer: str, 
                            client: AsyncOpenAI, verbose: bool = False) -> SimpleRAGContext:
        """
        Process a single QA pair with simplified RAG
        
        Args:
            document_name: Name of the document
            question: The question
            answer: The answer to evaluate
            client: AsyncOpenAI client
            verbose: Whether to include verbose logging
            
        Returns:
            SimpleRAGContext with results
        """
        self.logger.info(f"Processing QA pair for {document_name}")
        
        # Reset grace period for new QA pair
        self.current_grace_attempts = 0
        actual_iterations = 0
        
        # Get all relevant pages above similarity threshold
        all_relevant_pages = await self._get_all_relevant_pages(
            document_name, question, answer, client
        )
        
        accumulated_pages = set()
        accumulated_content = ""
        all_pages = []
        
        for iteration in range(self.max_iterations):
            actual_iterations += 1
            if verbose:
                self.logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Find next-best page from sorted list
            next_page = await self._find_next_best_page(
                all_relevant_pages, accumulated_pages
            )
            
            if next_page is None:
                if verbose:
                    self.logger.info("No more relevant pages found")
                break
            
            if verbose:
                self.logger.info(f"Found next-best page: {next_page}")
            
            # Add surrounding pages (n-x to n+x)
            new_pages = await self._add_surrounding_pages(
                document_name, next_page, accumulated_pages
            )
            
            # Get content for new pages
            new_content = await self._get_pages_content(document_name, new_pages)
            
            # Add to accumulated content
            accumulated_pages.update(new_pages)
            all_pages.extend(new_pages)
            accumulated_content += new_content
            
            if verbose:
                self.logger.info(f"Added pages {sorted(new_pages)}, total pages: {len(accumulated_pages)}")
            
            # Get LLM judgment using WHOLE context
            judgment, confidence, reasoning, supporting_evidence, contradicting_evidence, missing_information = await self._get_llm_judgment(
                accumulated_content, question, answer, client
            )
            
            if verbose:
                self.logger.info(f"LLM judgment: {judgment} (confidence: {confidence:.3f})")
            
            # Handle judgment
            if judgment == "True":
                if verbose:
                    self.logger.info("True judgment - terminating")
                break
            elif judgment == "False":
                if self.current_grace_attempts < self.grace_period:
                    self.current_grace_attempts += 1
                    if verbose:
                        self.logger.info(f"False judgment - grace period attempt {self.current_grace_attempts}/{self.grace_period}")
                        self.logger.info(f"Current pages: {sorted(accumulated_pages)}")
                    continue
                else:
                    if verbose:
                        self.logger.info("False judgment - grace period exhausted")
                    break
            elif judgment == "Insufficient_Details":
                if verbose:
                    self.logger.info("Insufficient details - terminating")
                break
            elif judgment == "Unfinished_Research":
                if verbose:
                    self.logger.info("Unfinished research - continuing search")
                continue
        
        return SimpleRAGContext(
            pages=sorted(accumulated_pages),
            content=accumulated_content,
            total_chars=len(accumulated_content),
            iterations=actual_iterations,
            judgment=judgment,
            confidence=confidence,
            reasoning=reasoning,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            missing_information=missing_information
        )
    
    async def _get_all_relevant_pages(self, document_name: str, question: str, answer: str,
                                    client: AsyncOpenAI) -> List[Tuple[int, float]]:
        """Get all pages above similarity threshold, sorted by relevance"""
        try:
            # Create search query
            search_query = f"{question} {answer}"
            
            # Get all results with generous threshold
            results = await self.vector_store_manager.search_document(
                document_name, search_query, client, top_k=100  # Get many results
            )
            
            # Filter by similarity threshold and extract page numbers
            relevant_pages = []
            for result in results:
                if result.similarity_score >= 0.3:  # Generous threshold
                    relevant_pages.append((result.chunk.page_number, result.similarity_score))
            
            # Sort by similarity score (descending)
            relevant_pages.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"Found {len(relevant_pages)} relevant pages above threshold")
            return relevant_pages
            
        except Exception as e:
            self.logger.error(f"Error getting relevant pages: {e}")
            return []
    
    async def _find_next_best_page(self, all_relevant_pages: List[Tuple[int, float]], 
                                  exclude_pages: set) -> Optional[int]:
        """Find the next-best page from sorted list that's not already included"""
        for page_num, similarity_score in all_relevant_pages:
            if page_num not in exclude_pages:
                self.logger.info(f"Selected page {page_num} with similarity {similarity_score:.3f}")
                return page_num
        return None

    async def _find_most_relevant_page(self, document_name: str, question: str, answer: str,
                                     client: AsyncOpenAI, exclude_pages: set, search_expansion: bool = False) -> Optional[int]:
        """Find the most relevant page using cosine similarity"""
        try:
            # Create search query
            search_query = f"{question} {answer}"
            
            # Search for relevant chunks
            results = await self.vector_store_manager.search_document(
                document_name, search_query, client, top_k=20  # Get more results to find next-best
            )
            
            # Find the next-best page not already included
            for result in results:
                page_num = result.chunk.page_number
                if page_num not in exclude_pages:
                    return page_num
            
            # Debug: log when no new pages are found
            self.logger.info(f"No new pages found. Excluded pages: {sorted(exclude_pages)}")
            self.logger.info(f"Search results pages: {[r.chunk.page_number for r in results[:5]]}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding relevant page: {e}")
            return None
    
    async def _add_surrounding_pages(self, document_name: str, center_page: int, 
                                   exclude_pages: set) -> List[int]:
        """Add n-x to n+x pages around the center page"""
        try:
            # Get document boundaries
            document_store = self.vector_store_manager.document_stores.get(document_name)
            if not document_store:
                return [center_page]
            
            all_chunks = document_store.get_document_chunks(document_name)
            if not all_chunks:
                return [center_page]
            
            page_numbers = [chunk.page_number for chunk in all_chunks]
            min_page = min(page_numbers)
            max_page = max(page_numbers)
            
            # Calculate page range
            start_page = max(min_page, center_page - self.surrounding_pages)
            end_page = min(max_page, center_page + self.surrounding_pages)
            
            # Get pages in range
            pages_to_add = []
            for page_num in range(start_page, end_page + 1):
                if page_num not in exclude_pages:
                    pages_to_add.append(page_num)
            
            return pages_to_add
            
        except Exception as e:
            self.logger.error(f"Error adding surrounding pages: {e}")
            return [center_page]
    
    async def _get_pages_content(self, document_name: str, pages: List[int]) -> str:
        """Get content for specified pages"""
        try:
            document_store = self.vector_store_manager.document_stores.get(document_name)
            if not document_store:
                return ""
            
            content_parts = []
            for page_num in sorted(pages):
                page_chunks = document_store.get_chunks_by_page(document_name, page_num)
                for chunk in page_chunks:
                    content_parts.append(f"=== PAGE {page_num} ===\n{chunk.content}\n")
            
            return "\n".join(content_parts)
            
        except Exception as e:
            self.logger.error(f"Error getting pages content: {e}")
            return ""
    
    async def _get_llm_judgment(self, content: str, question: str, answer: str, 
                              client: AsyncOpenAI) -> Tuple[str, float, str, List[str], List[str], List[str]]:
        """Get LLM judgment for the content"""
        try:
            prompt = f"""
You are evaluating a question-answer pair against document content.

DOCUMENT CONTENT:
{content}

QUESTION: {question}
ANSWER: {answer}

Categorize this QA pair into one of these 4 categories:

1. **True**: The answer is correct, fully supported, and no major information is missing
2. **False**: The answer has provably wrong claims, contradicts the document, or contains unsupported details not found in the document
3. **Insufficient_Details**: The answer is correct but the document contains important details that are missing from the answer
4. **Unfinished_Research**: The answer contains claims that could not yet be verified as true or false, more documents are needed

IMPORTANT LOGIC:
- If the DOCUMENT has important details MISSING FROM THE ANSWER → "Insufficient_Details"
- If the ANSWER has details MISSING FROM THE DOCUMENT (and not public knowledge) → "False"
- If you find ANY contradicting evidence, the judgment MUST be "False"
- Only use "True" if the answer is fully supported with no missing information

EVIDENCE EXTRACTION RULES:
- For "False" judgments: List ALL unsupported or contradicting claims from the answer in contradicting_evidence
- For "Insufficient_Details" judgments: List ALL important details from the document that are missing from the answer in missing_information
- Always provide specific examples, not just general statements
- ALWAYS include page numbers and exact quotes in your evidence

EVIDENCE FORMATTING:
- supporting_evidence: "Page X: \"exact quote from document\""
- contradicting_evidence: "Answer states: \"claim\" — the document does not mention/state..."
- missing_information: "Page X: \"exact quote from document\""

EXAMPLE for "False" judgment:
If answer claims "DSGVO compliance" but document doesn't mention DSGVO, then contradicting_evidence should include: "Answer states: 'DSGVO compliance' — the document does not mention DSGVO"

IMPORTANT: Return ONLY valid JSON. No additional text, explanations, or formatting outside the JSON object.

Return your response as JSON with these exact fields:
{{
    "judgment": "True|False|Insufficient_Details|Unfinished_Research",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your judgment",
    "supporting_evidence": ["List of specific evidence from the document that supports the answer - format as 'Page X: \"exact quote from document\"'"],
    "contradicting_evidence": ["List of specific unsupported or contradicting claims in the answer that are not found in the document - format as 'Answer states: \"claim\" — the document does not mention/state...' - if this list is not empty, judgment MUST be False"],
    "missing_information": ["List of important information from the document that is missing from the answer - format as 'Page X: \"exact quote from document\" - only relevant for Insufficient_Details judgments"]
}}
"""

            response = await client.responses.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                instructions="You are an expert AI assistant that evaluates question-answer pairs against document content. You must respond with valid JSON only.",
                input=prompt,
            )
            
            # Parse JSON response
            response_text = response.output_text.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            result = json.loads(json_text)
            
            judgment = result.get("judgment", "Unfinished_Research")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")
            supporting_evidence = result.get("supporting_evidence", [])
            contradicting_evidence = result.get("contradicting_evidence", [])
            missing_information = result.get("missing_information", [])
            
            return judgment, confidence, reasoning, supporting_evidence, contradicting_evidence, missing_information
            
        except Exception as e:
            self.logger.error(f"Error getting LLM judgment: {e}")
            return "Unfinished_Research", 0.0, f"Error: {str(e)}", [], [], []
    
    def get_context_summary(self, context: SimpleRAGContext, verbose: bool = False) -> Dict[str, Any]:
        """Get context summary for JSON output"""
        summary = {
            "status": "success",
            "pages_covered": context.pages,
            "page_count": len(context.pages),
            "total_characters": context.total_chars,
            "iterations_used": context.iterations,
            "judgment": context.judgment,
            "confidence": context.confidence,
            "reasoning": context.reasoning
        }
        
        if verbose:
            summary.update({
                "content_preview": context.content[:500] + "..." if len(context.content) > 500 else context.content,
                "grace_period_attempts": self.current_grace_attempts,
                "grace_period_max": self.grace_period
            })
        
        return summary
