"""
QA Validation with Iterative RAG

This module implements the 3-step QA validation process using iterative RAG
instead of loading full document content into memory.
"""

import time
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from openai import AsyncOpenAI, BadRequestError

from .iterative_rag import IterativeRAGRetriever, RAGContext

@dataclass
class QAValidationResult:
    """Result of QA validation process"""
    document_name: str
    question: str
    answer: str
    judgment: str
    reasoning: str
    confidence_numeric: float
    confidence_verbal: str
    step_analysis: Dict[str, str]
    rag_context_summary: Dict[str, Any]
    llm_logs: Dict[str, Dict[str, Any]]
    token_usage: Dict[str, int]
    processing_time: float

class QAValidatorRAG:
    """QA Validator that uses iterative RAG for context retrieval"""
    
    def __init__(self, rag_retriever: IterativeRAGRetriever, 
                 system_prompt_path: str = "../QAPrompts/system_prompt.txt",
                 deployment_name: str = "gpt-4",
                 concurrent_limit: int = 15):
        """
        Initialize the QA validator with RAG
        
        Args:
            rag_retriever: Iterative RAG retriever instance
            system_prompt_path: Path to system prompt file
            deployment_name: Name of the model deployment
            concurrent_limit: Maximum concurrent requests
        """
        self.rag_retriever = rag_retriever
        self.system_prompt_path = system_prompt_path
        self.deployment_name = deployment_name
        self.semaphore = asyncio.Semaphore(concurrent_limit)
        self.logger = logging.getLogger(__name__)
        
        # Load system prompt
        self._load_system_prompt()
        
        # Confidence classes for verbal descriptions
        self.confidence_classes = [
            "Almost no chance",
            "Highly unlikely", 
            "Chances are slight",
            "Unlikely",
            "Less than even",
            "Better than even",
            "Likely",
            "Very good chance",
            "Highly likely",
            "Almost certain"
        ]
    
    def _load_system_prompt(self):
        """Load the system prompt from file"""
        try:
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            self.logger.error(f"System prompt file not found: {self.system_prompt_path}")
            self.system_prompt = "You are an expert AI assistant that evaluates question-answer pairs for accuracy and completeness."
    
    def score_to_verbal(self, score: float) -> str:
        """Convert numeric confidence score to verbal description"""
        s = 0.0 if score is None else float(score)
        s = max(0.0, min(1.0, s))
        bounds = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        idx = max(i for i, b in enumerate(bounds) if s >= b)
        return self.confidence_classes[idx]
    
    async def _timed_request(self, client: AsyncOpenAI, instructions: str, input_text: str, 
                           reasoning: Optional[Dict] = None,
                           text: Optional[Dict] = None, user: str = "qa-judge", 
                           store: bool = True, timeout_sec: int = 60) -> tuple:
        """
        Make a timed request to the OpenAI API with semaphore control
        
        Args:
            client: AsyncOpenAI client instance
            instructions: System instructions
            input_text: Input text for the model
            reasoning: Reasoning configuration
            text: Text configuration
            user: User identifier
            store: Whether to store the request
            timeout_sec: Timeout in seconds
            
        Returns:
            Tuple of (response, logs)
        """
        async with self.semaphore:
            start = time.time()
            try:
                resp = await asyncio.wait_for(
                    client.responses.create(
                        model=self.deployment_name,
                        instructions=instructions,
                        input=input_text,
                        reasoning=reasoning,
                        text=text,
                        user=user,
                        store=store
                    ), 
                    timeout=timeout_sec
                )
                
            except (asyncio.TimeoutError, BadRequestError) as e:
                runtime = round(time.time() - start, 2)
                self.logger.error(f"Request failed with error: {e}")
                return None, {
                    "runtime_sec": runtime,
                    "input_tokens": None,
                    "cached_input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "timed_out": True,
                }
            except Exception as e:
                runtime = round(time.time() - start, 2)
                self.logger.error(f"Request failed with error: {e}")
                return None, {
                    "runtime_sec": runtime,
                    "input_tokens": None,
                    "cached_input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "timed_out": False,
                }
            
            runtime = round(time.time() - start, 2)
            usage = getattr(resp, "usage", None)
            ptd = getattr(usage, "prompt_tokens_details", None) if usage else None
            cached = getattr(ptd, "cached_tokens", None) if ptd else None
            
            return resp, {
                "runtime_sec": runtime,
                "input_tokens": getattr(usage, "input_tokens", None),
                "cached_input_tokens": cached,
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "timed_out": False,
            }
    
    def _extract_reasoning_text(self, response) -> str:
        """Extract reasoning text from response"""
        if response and hasattr(response, 'output_text'):
            return response.output_text
        return ""
    
    async def validate_qa_with_rag(self, question: str, answer: str, document_name: str,
                                  client: AsyncOpenAI) -> Optional[QAValidationResult]:
        """
        Validate QA pair using iterative RAG for context retrieval
        
        Args:
            question: The question to validate
            answer: The answer to validate
            document_name: Name of the source document
            client: AsyncOpenAI client instance
            
        Returns:
            QAValidationResult or None if validation failed
        """
        start_time = time.time()
        
        try:
            # Step 1: Build context using iterative RAG
            self.logger.info(f"Building RAG context for: {document_name}")
            rag_context_text = await self.rag_retriever.get_context_for_qa(
                document_name, question, answer, client
            )
            
            # Get context summary for analysis
            rag_context = await self.rag_retriever.build_context_iteratively(
                document_name, question, answer, client
            )
            context_summary = self.rag_retriever.get_context_summary(rag_context)
            
            # Step 2: Context Analysis
            resp1, log1 = await self._timed_request(
                client=client,
                instructions=self.system_prompt,
                input_text=(
                    "Step 1 — CONTEXT ANALYSIS:\n"
                    "Analyze the provided document context to understand what information is available to answer the question. "
                    "Identify key facts, data points, and relevant sections. "
                    "Do NOT evaluate the answer yet - just understand what the document says.\n\n"
                    f"--- DOCUMENT CONTEXT START ---\n{rag_context_text}\n--- DOCUMENT CONTEXT END ---\n\n"
                    f"--- QUESTION ---\n{question}\n"
                ),
                reasoning={"effort": "low"},
                text={"verbosity": "low"}
            )
            
            if resp1 is None:
                return None
            step1_notes = self._extract_reasoning_text(resp1)
            
            # Step 3: Answer Evaluation  
            resp2, log2 = await self._timed_request(
                client=client,
                instructions=self.system_prompt,
                input_text=(
                    "Step 2 — ANSWER EVALUATION:\n"
                    "Now evaluate the provided answer against the question and document context. "
                    "Check for accuracy, completeness, and whether it properly addresses the question. "
                    "Consider if the answer contains incorrect information, missing key points, or irrelevant details. "
                    "Do NOT provide your final judgment yet.\n\n"
                    f"--- PREVIOUS CONTEXT ANALYSIS ---\n{step1_notes}\n\n"
                    f"--- QUESTION ---\n{question}\n\n"
                    f"--- ANSWER TO EVALUATE ---\n{answer}\n"
                ),
                reasoning={"effort": "low"},
                text={"verbosity": "low"}
            )
            
            if resp2 is None:
                return None
            step2_notes = self._extract_reasoning_text(resp2)
            
            # Step 4: Final Judgment
            resp3, log3 = await self._timed_request(
                client=client,
                instructions=self.system_prompt,
                input_text=(
                    "Step 3 — FINAL JUDGMENT:\n"
                    "Based on your analysis, provide your final evaluation. Return ONLY a single valid JSON object with this exact structure:\n"
                    "{\n"
                    '  "judgment": "<True|False|Insufficient_Details|Unfinished_Research>",\n'
                    '  "reasoning": "<detailed explanation of your decision>",\n'
                    '  "confidence": <numeric value between 0.0 and 1.0>\n'
                    "}\n\n"
                    "Where:\n"
                    "- True: The answer as it is is correct, fully supported, and no major information in the document is missing\n"
                    "- False: The answer has provably wrong claims or is irrelevant to the question\n"
                    "- Insufficient_Details: The answer is not as detailed as the document. The answer is correct but lacks important information that the document contains\n"
                    "- Unfinished_Research: The answer contains claims that could not yet be verified as true or false, more documents are needed\n"
                    "- confidence: How certain you are of your judgment (0.0 = very uncertain, 1.0 = very certain)\n\n"
                    f"--- CONTEXT ANALYSIS ---\n{step1_notes}\n\n"
                    f"--- ANSWER EVALUATION ---\n{step2_notes}\n"
                ),
                reasoning={"effort": "low"},
                text={"verbosity": "low"}
            )
            
            if resp3 is None:
                return None
            
            # Parse the final judgment
            try:
                result = json.loads(resp3.output_text)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Add verbal confidence description
                confidence_verbal = self.score_to_verbal(result.get("confidence", 0.0))
                
                # Calculate total tokens
                total_input_tokens = (log1.get("input_tokens", 0) + log2.get("input_tokens", 0) + log3.get("input_tokens", 0))
                total_output_tokens = (log1.get("output_tokens", 0) + log2.get("output_tokens", 0) + log3.get("output_tokens", 0))
                
                return QAValidationResult(
                    document_name=document_name,
                    question=question,
                    answer=answer,
                    judgment=result.get("judgment"),
                    reasoning=result.get("reasoning"),
                    confidence_numeric=result.get("confidence", 0.0),
                    confidence_verbal=confidence_verbal,
                    step_analysis={
                        "step1_context": step1_notes,
                        "step2_evaluation": step2_notes
                    },
                    rag_context_summary=context_summary,
                    llm_logs={
                        "step1": log1,
                        "step2": log2, 
                        "step3": log3
                    },
                    token_usage={
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    },
                    processing_time=processing_time
                )
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                self.logger.error(f"Response text: {resp3.output_text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error validating QA pair for {document_name}: {e}")
            return None

class BatchQAProcessor:
    """Processes multiple QA pairs in batches using RAG validation"""
    
    def __init__(self, qa_validator: QAValidatorRAG, concurrent_tasks: int = 15):
        """
        Initialize batch processor
        
        Args:
            qa_validator: QA validator instance
            concurrent_tasks: Maximum concurrent tasks
        """
        self.qa_validator = qa_validator
        self.concurrent_tasks = concurrent_tasks
        self.logger = logging.getLogger(__name__)
    
    async def process_qa_pair(self, qa_pair: Dict[str, Any], document_name: str,
                             client: AsyncOpenAI) -> Optional[QAValidationResult]:
        """Process a single QA pair"""
        try:
            return await self.qa_validator.validate_qa_with_rag(
                question=qa_pair["Question"],
                answer=qa_pair["Detailed Answer"],
                document_name=document_name,
                client=client
            )
        except Exception as e:
            self.logger.error(f"Error processing QA pair for {document_name}: {e}")
            return None
    
    async def process_document_qa_pairs(self, document_name: str, qa_pairs: List[Dict[str, Any]],
                                       client: AsyncOpenAI, max_questions: int = None) -> List[QAValidationResult]:
        """Process all QA pairs for a document concurrently"""
        
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]
        
        self.logger.info(f"Processing {len(qa_pairs)} QA pairs for document: {document_name}")
        
        # Create tasks for concurrent processing
        tasks = [
            self.process_qa_pair(qa_pair, document_name, client)
            for qa_pair in qa_pairs
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Exception in QA processing: {result}")
                failed_count += 1
            elif result is None:
                failed_count += 1
            else:
                successful_results.append(result)
        
        self.logger.info(f"Document {document_name}: {len(successful_results)} successful, {failed_count} failed")
        return successful_results
    
    async def process_all_documents(self, qa_pairs_by_document: Dict[str, List[Dict[str, Any]]],
                                   client: AsyncOpenAI, max_docs: int = None,
                                   max_questions_per_doc: int = None) -> List[QAValidationResult]:
        """Process all documents and their QA pairs"""
        
        docs_to_process = list(qa_pairs_by_document.keys())
        if max_docs:
            docs_to_process = docs_to_process[:max_docs]
        
        self.logger.info(f"Starting concurrent processing of {len(docs_to_process)} documents...")
        start_time = time.time()
        
        # Create tasks for all documents
        document_tasks = [
            self.process_document_qa_pairs(doc_name, qa_pairs_by_document[doc_name], client, max_questions_per_doc)
            for doc_name in docs_to_process
        ]
        
        # Process all documents concurrently
        all_results = await asyncio.gather(*document_tasks, return_exceptions=True)
        
        # Flatten results
        final_results = []
        for doc_results in all_results:
            if isinstance(doc_results, Exception):
                self.logger.error(f"Exception processing document: {doc_results}")
            else:
                final_results.extend(doc_results)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Total results: {len(final_results)}")
        
        return final_results

