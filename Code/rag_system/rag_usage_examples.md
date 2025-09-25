# RAG System Usage Examples

## Basic Usage

```bash
# Run with default settings (2 docs, 5 questions each)
python run_rag_system.py

# Run with custom document and question limits
python run_rag_system.py --docs 3 --questions 10

# Force rebuild vector stores
python run_rag_system.py --rebuild

# Build vector stores only (no QA processing)
python run_rag_system.py --build-vectors

# Run with detailed metrics in output JSON
python run_rag_system.py --verbose
```

## Advanced Configuration

```bash
# High-performance run with more concurrent tasks
python run_rag_system.py --docs 5 --questions 15 --concurrent 25

# Large context processing
python run_rag_system.py --context-chars 12000 --iterations 5

# Quick test run
python run_rag_system.py --docs 1 --questions 3 --concurrent 5

# Full processing with custom settings
python run_rag_system.py --docs 10 --questions 20 --concurrent 30 --context-chars 10000 --iterations 4
```

## Available Arguments

- `--docs N`: Maximum number of documents to process (default: 2)
- `--questions N`: Maximum questions per document (default: 5)
- `--rebuild`: Force rebuild vector stores (default: False)
- `--concurrent N`: Number of concurrent tasks (default: 15)
- `--context-chars N`: Maximum context characters (default: 8000)
- `--iterations N`: Maximum RAG iterations (default: 3)
- `--verbose`: Include detailed metrics in output JSON (default: False)
- `--build-vectors`: Build vector stores only (no QA processing)
- `--test`: Run tests instead of processing

## Examples for Different Scenarios

### Quick Testing
```bash
python run_rag_system.py --docs 1 --questions 2 --concurrent 5
```

### Production Run
```bash
python run_rag_system.py --docs 20 --questions 50 --concurrent 25 --context-chars 12000
```

### Debugging with Rebuild
```bash
python run_rag_system.py --rebuild --docs 2 --questions 5
```

### Memory-Constrained Environment
```bash
python run_rag_system.py --concurrent 5 --context-chars 4000 --iterations 2
```

### Output Format Control
```bash
# Default: Clean output without detailed metrics
python run_rag_system.py --docs 2 --questions 5

# Verbose: Include all metrics (tokens, processing time, etc.)
python run_rag_system.py --docs 2 --questions 5 --verbose
```

### Grace Period System
The system now includes a grace period for "False" judgments:
- **2 additional attempts** to find relevant context when "False" is detected
- **Automatic reset** when relevant context is found
- **Reduced false negatives** from insufficient context
- **Detailed logging** of grace period decisions
