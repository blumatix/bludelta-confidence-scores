# Unified Judgment System

## Overview

The RAG system now uses a unified judgment system across all components (iterative RAG, QA validation, and results processing) with four standardized categories.

## Judgment Categories

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

## Key Changes

### 1. **Unified Criteria**
- Both RAG context assessment and final QA validation use identical judgment criteria
- No more discrepancies between different assessment stages
- Consistent reasoning across all components

### 2. **Grace Period for False Judgments**
- **False** judgments trigger a grace period of 2 additional attempts
- Grace period resets when relevant context is found
- Prevents false negatives from insufficient context

### 3. **Replaced "Undeterminable" with "Unfinished_Research"**
- More descriptive and actionable category
- Indicates need for additional documents/sources
- Better reflects the actual state of assessment

### 4. **Enhanced Error Handling**
- Invalid judgments default to "Unfinished_Research"
- Better validation of judgment categories
- Consistent error handling across components

## Implementation Details

### RAG Context Assessment
```python
# Judgment criteria in iterative_rag.py
JUDGMENT_CRITERIA = {
    "True": "The answer as it is is correct, fully supported, and no major information in the document is missing",
    "False": "The answer has provably wrong claims or is irrelevant to the question",
    "Insufficient_Details": "The answer is not as detailed as the document",
    "Unfinished_Research": "The answer contains claims that could not yet be verified as true or false, more documents are needed"
}
```

### QA Validation
```python
# Same criteria in qa_validation_rag.py
FINAL_JUDGMENT_CRITERIA = {
    "True": "The answer as it is is correct, fully supported, and no major information in the document is missing",
    "False": "The answer has provably wrong claims or is irrelevant to the question", 
    "Insufficient_Details": "The answer is not as detailed as the document",
    "Unfinished_Research": "The answer contains claims that could not yet be verified as true or false, more documents are needed"
}
```

## Benefits

### 1. **Consistency**
- No more discrepancies between RAG assessment and final validation
- Unified reasoning across all components
- Predictable judgment behavior

### 2. **Better Categorization**
- "Unfinished_Research" is more descriptive than "Undeterminable"
- Clear distinction between incomplete answers and unverifiable claims
- Better guidance for users on next steps

### 3. **Improved Accuracy**
- Grace period reduces false negatives
- More nuanced assessment of answer quality
- Better handling of complex documents

### 4. **Enhanced Debugging**
- Consistent judgment categories across all logs
- Clear termination reasons
- Better error tracking and analysis

## Usage Examples

### True Judgment
```json
{
  "judgment": "True",
  "reasoning": "The answer correctly addresses the question with all claims supported by the document",
  "confidence": 0.9
}
```

### False Judgment (with Grace Period)
```json
{
  "judgment": "False", 
  "reasoning": "The answer contains factual errors that contradict the document",
  "confidence": 0.85
}
```

### Insufficient_Details
```json
{
  "judgment": "Insufficient_Details",
  "reasoning": "The answer is correct but lacks important details that the document contains",
  "confidence": 0.7
}
```

### Unfinished_Research
```json
{
  "judgment": "Unfinished_Research",
  "reasoning": "The answer contains claims that cannot be verified with current context",
  "confidence": 0.4
}
```

## Migration Notes

- **Old "Undeterminable"** → **New "Unfinished_Research"**
- **Grace period** now applies to "False" judgments
- **Unified criteria** across all assessment stages
- **Enhanced error handling** for invalid judgments
