# RAG System Judgment Categories

## Standard Judgment Categories

The RAG system uses four standardized judgment categories for QA validation:

### 1. **True**
- **Definition**: The answer as it is is correct, fully supported, and no major information in the document is missing
- **Criteria**: 
  - Answer is factually correct and complete
  - Document provides sufficient evidence to support all claims
  - No important information from the document is missing from the answer
- **Confidence**: High confidence required (typically >0.8)

### 2. **False**
- **Definition**: The answer has provably wrong claims or is irrelevant to the question
- **Criteria**:
  - Answer contains factual errors that contradict the document
  - Answer is irrelevant to the question being asked
  - Answer makes claims that are demonstrably false
- **Confidence**: High confidence required (typically >0.8)
- **Grace Period**: 2 additional attempts to find supporting context

### 3. **Insufficient_Details**
- **Definition**: The answer is not as detailed as the document. The answer is correct but lacks important information that the document contains
- **Criteria**:
  - Answer is factually correct but incomplete
  - Document contains more relevant information than what's in the answer
  - Answer lacks important details that the document provides
- **Confidence**: Medium to high confidence (typically >0.6)

### 4. **Unfinished_Research**
- **Definition**: The answer contains claims that could not yet be verified as true or false, more documents are needed
- **Criteria**:
  - Answer makes claims that cannot be verified with current context
  - Additional documents or sources would be needed to verify claims
  - Current document doesn't contain enough information to assess the claims
- **Confidence**: Low to medium confidence (typically <0.7)

## Implementation Notes

- **Consistency**: All components (iterative RAG, QA validation, results processing) use these same categories
- **JSON Format**: Judgments are returned as strings in JSON responses
- **Error Handling**: Invalid judgments default to "Unfinished_Research"
- **Confidence Scoring**: Each judgment includes a numeric confidence score (0.0-1.0)

## Usage in Code

```python
# Valid judgment values
VALID_JUDGMENTS = ["True", "False", "Insufficient_Details", "Unfinished_Research"]

# Example JSON response
{
    "judgment": "True",
    "reasoning": "The answer correctly addresses the question...",
    "confidence": 0.85
}
```
