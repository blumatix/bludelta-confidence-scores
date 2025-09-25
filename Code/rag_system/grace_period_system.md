# False Judgment Grace Period System

## Overview

The RAG system now implements a grace period system for "False" judgments to reduce false negatives caused by insufficient context. When a "False" judgment is made, the system gets 2 additional chances to find relevant context that might change the assessment.

## How It Works

### 1. **Grace Period Trigger**
- When a "False" judgment is made with high confidence (≥ confidence threshold)
- System enters grace period mode instead of immediately stopping
- Gets 2 additional attempts to find relevant context

### 2. **Grace Period Reset**
- **Automatic Reset**: When relevant context is found (>200 characters of new content)
- **Manual Reset**: At the start of each new QA pair
- **Smart Detection**: System detects meaningful new content and resets counter

### 3. **Grace Period Exhaustion**
- After 2 grace period attempts, system confirms "False" judgment
- Provides detailed logging of why judgment was confirmed
- Includes contradiction evidence in final result

## Implementation Details

### Configuration
```python
# Grace period settings
self.false_judgment_grace_period = 2  # Number of additional attempts
self.current_grace_attempts = 0       # Current attempt counter
```

### Logging Output
```
False judgment - grace period attempt 1/2. Contradictions: [...]
Continuing search to find more relevant context...
Relevant context found - resetting grace period from 1 to 0
```

### Termination Reasons
- `false_judgment_grace_period_iteration_X_attempt_Y_confidence_Z`
- `false_judgment_confirmed_after_grace_period_iteration_X_confidence_Y`

## Benefits

### 1. **Reduced False Negatives**
- System doesn't give up too early on "False" judgments
- Additional context retrieval can reveal supporting evidence
- Better handling of complex documents with scattered information

### 2. **Smart Context Detection**
- Automatically resets when relevant content is found
- Prevents unnecessary iterations when context is already sufficient
- Balances thoroughness with efficiency

### 3. **Detailed Tracking**
- Grace period attempts are tracked in error statistics
- Clear logging of grace period decisions
- Termination reasons indicate grace period usage

## Example Scenarios

### Scenario 1: False Judgment with Grace Period
```
1. Initial assessment: "False" (confidence: 0.85)
2. Grace period attempt 1: Continue searching
3. Find relevant context: Reset grace period
4. New assessment: "True" (confidence: 0.90)
5. Result: Corrected judgment
```

### Scenario 2: Confirmed False Judgment
```
1. Initial assessment: "False" (confidence: 0.85)
2. Grace period attempt 1: Continue searching
3. No relevant context found
4. Grace period attempt 2: Continue searching
5. Still no relevant context
6. Result: Confirmed "False" judgment
```

### Scenario 3: Grace Period Reset
```
1. Initial assessment: "False" (confidence: 0.85)
2. Grace period attempt 1: Continue searching
3. Find 500 characters of relevant context
4. Grace period reset to 0
5. New assessment: "True" (confidence: 0.88)
6. Result: Corrected judgment with fresh grace period
```

## Monitoring

### Error Statistics
The system tracks grace period usage:
```json
{
  "grace_period_attempts": 1,
  "grace_period_max": 2,
  "total_assessments": 150
}
```

### Logging
- Grace period attempts are logged with attempt numbers
- Context resets are logged when relevant content is found
- Termination reasons indicate grace period usage

## Configuration Options

### Adjusting Grace Period
```python
# In iterative_rag.py
self.false_judgment_grace_period = 2  # Default: 2 attempts
```

### Context Threshold
```python
# Minimum characters to trigger grace period reset
if iteration_chars > 200:  # Default: 200 characters
```

## Best Practices

1. **Monitor Grace Period Usage**: Check if grace periods are frequently used
2. **Adjust Thresholds**: Fine-tune context detection thresholds
3. **Review Logs**: Analyze grace period decisions for optimization
4. **Balance Performance**: Grace periods add processing time but improve accuracy

## Expected Impact

- **Reduced False Negatives**: Fewer incorrect "False" judgments
- **Better Context Utilization**: More thorough document analysis
- **Improved Accuracy**: Better handling of complex documents
- **Slightly Increased Processing Time**: Additional iterations when needed
