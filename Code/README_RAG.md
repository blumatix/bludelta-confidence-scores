# QA Processing with Iterative RAG

This system replaces the Jupyter notebook with a Python-based solution that uses iterative RAG (Retrieval-Augmented Generation) to solve the context length problem when processing QA pairs.

## Overview

Instead of loading entire documents into memory, this system:

1. **Splits documents** into manageable chunks (pages, paragraphs, or sections)
2. **Creates vector stores** for each document using embeddings
3. **Uses iterative RAG** to gradually build context by retrieving the most relevant chunks
4. **Processes QA pairs** with intelligent context retrieval rather than full document loading

## Architecture

```
Documents → Document Splitter → Vector Stores → Iterative RAG → QA Validation
    ↓              ↓                 ↓              ↓              ↓
PDF Files      Chunks           Embeddings    Smart Context   3-Step Analysis
```

## Key Components

### 1. `document_splitter.py`
- Splits PDF documents into chunks using different strategies
- Supports page-based, paragraph-based, and section-based splitting
- Preserves metadata for each chunk

### 2. `vector_store.py`
- Creates FAISS-based vector stores for document chunks
- Generates embeddings using OpenAI's API
- Supports per-document vector stores for efficient retrieval

### 3. `iterative_rag.py`
- Implements iterative context building
- Gradually adds relevant chunks based on question and answer content
- Respects context length limits while maximizing relevance

### 4. `qa_validation_rag.py`
- 3-step QA validation process using RAG context
- Async processing with semaphore-controlled concurrency
- Detailed logging and error handling

### 5. `main_qa_processor.py`
- Complete pipeline orchestration
- Configuration management
- Results saving and analysis

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_model_name
DOCUMENT_STORAGE_QA=path_to_your_documents
CONCURRENT_TASKS=15
MAX_CONTEXT_CHARS=8000
MAX_RAG_ITERATIONS=3
SPLITTING_STRATEGY=paragraphs
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main_qa_processor.py
```

### Testing

Test individual components:
```bash
python test_rag_system.py
```

### Configuration Options

**Document Splitting Strategies:**
- `pages`: Split by PDF pages (good for structured documents)
- `paragraphs`: Split by paragraph breaks (good for text-heavy documents)
- `sections`: Split by detected sections/headers (good for structured content)

**Context Management:**
- `MAX_CONTEXT_CHARS`: Maximum characters in final context (default: 8000)
- `MAX_RAG_ITERATIONS`: Maximum retrieval iterations (default: 3)
- `MIN_RELEVANCE_THRESHOLD`: Minimum similarity score for inclusion (default: 0.3)

**Performance:**
- `CONCURRENT_TASKS`: Number of concurrent API requests (default: 15)
- `CHUNK_SIZE`: Size of document chunks in characters (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks in characters (default: 200)

## How Iterative RAG Works

1. **Initial Query**: Uses the original question to find relevant chunks
2. **Answer Analysis**: Searches for content related to the answer being evaluated
3. **Context Refinement**: Adds additional context to fill knowledge gaps
4. **Iteration Control**: Stops when context limit is reached or no more relevant content is found

Each iteration uses different search strategies:
- **Iteration 1**: Focus on the main question
- **Iteration 2**: Focus on answer content and verification
- **Iteration 3+**: Look for supporting evidence or contradictions

## Output

The system generates:

1. **JSON Results File**: Complete validation results with metadata
2. **Log Files**: Detailed processing logs for debugging
3. **Vector Stores**: Saved embeddings for reuse (cached)

### Sample Output Structure
```json
{
  "metadata": {
    "timestamp": "2025-09-24_13-34-25",
    "model": "gpt-4",
    "total_results": 10,
    "processing_summary": {
      "total_tokens": 150000,
      "avg_processing_time": 3.2,
      "avg_confidence": 0.85
    }
  },
  "results": [
    {
      "document_name": "example.pdf",
      "question": "What is the main topic?",
      "answer": "The document discusses...",
      "judgment": "True",
      "confidence_numeric": 0.92,
      "confidence_verbal": "Highly likely",
      "rag_context_summary": {
        "total_chunks": 5,
        "pages_covered": [1, 3, 7],
        "iterations_used": 2,
        "character_utilization": 0.75
      }
    }
  ]
}
```

## Advantages Over Notebook Approach

1. **Scalability**: Handles large documents without memory issues
2. **Efficiency**: Only loads relevant context, not entire documents
3. **Speed**: Concurrent processing with intelligent caching
4. **Maintainability**: Modular code structure
5. **Reliability**: Better error handling and logging
6. **Flexibility**: Configurable strategies for different document types

## Performance Tips

1. **Vector Store Caching**: Vector stores are automatically saved and reused
2. **Batch Processing**: Documents are processed concurrently
3. **Context Optimization**: RAG system finds the most relevant content efficiently
4. **Token Management**: Smart context building respects token limits

## Troubleshooting

**Common Issues:**

1. **Missing Documents**: Ensure PDF files are in the correct folder
2. **API Errors**: Check Azure OpenAI credentials and quotas
3. **Memory Issues**: Reduce `CONCURRENT_TASKS` or `MAX_CONTEXT_CHARS`
4. **Vector Store Errors**: Delete the `VectorStores` folder to force rebuild

**Performance Tuning:**

- Increase `CONCURRENT_TASKS` for faster processing (watch API limits)
- Adjust `MAX_CONTEXT_CHARS` based on your model's context window
- Use `paragraphs` strategy for most documents
- Set `MIN_RELEVANCE_THRESHOLD` higher to reduce noise

## Integration

To integrate with existing workflows:

1. Import the main components:
```python
from main_qa_processor import QAProcessor, QAProcessorConfig

config = QAProcessorConfig()
processor = QAProcessor(config)
results = await processor.run_full_pipeline()
```

2. Use individual components:
```python
from iterative_rag import IterativeRAGRetriever
from vector_store import MultiDocumentVectorStore

# Custom RAG configuration
rag_retriever = IterativeRAGRetriever(
    vector_store_manager=your_vector_store,
    max_context_chars=10000,
    max_iterations=4
)
```

## Future Enhancements

- Support for more document formats (Word, HTML, etc.)
- Advanced chunk splitting algorithms
- Multi-modal document processing
- Real-time document updates
- Performance analytics dashboard
