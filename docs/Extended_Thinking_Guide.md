# Extended Thinking Mode Guide

## Overview

Extended thinking is a feature in Anthropic's Claude API that gives the model additional computation budget for internal reasoning before generating a response. This can significantly improve Claude's ability to solve complex problems, especially those requiring:

- Multi-step reasoning
- Mathematical calculations
- Code generation and analysis
- Complex logical deductions
- Detailed planning

## Implementation in BotWithMemory

Extended thinking has been integrated into the LLMBridge class to allow any component that interacts with the LLM to utilize this capability when needed.

### Configuration

Extended thinking can be configured globally or per-request:

1. **Global Configuration** (in `config.py`):
   ```python
   # In config.py, ApiConfig class
   extended_thinking: bool = False  # Whether extended thinking is enabled by default
   extended_thinking_budget: int = 4096  # Default token budget (min: 1024)
   ```

2. **Per-Request Configuration** (when calling `generate_response`):
   ```python
   response = llm_bridge.generate_response(
       messages=messages,
       extended_thinking=True,  # Override global setting
       extended_thinking_budget=8192  # Override global budget
   )
   ```

### Usage Guidelines

1. **When to Use Extended Thinking:**
   - Complex problem-solving tasks
   - Multi-step reasoning
   - Mathematical calculations
   - Code generation and analysis
   - Tasks where Claude struggles with direct responses

2. **Token Budget Considerations:**
   - Minimum budget: 1024 tokens
   - Recommended range: 4096-16384 tokens for most tasks
   - Larger budgets (e.g., 16384) for extremely complex reasoning

3. **Performance Impact:**
   - Extended thinking may increase response time
   - Consider using streaming for better user experience
   - The model may not use the entire allocated budget

4. **Cost Considerations:**
   - Extended thinking tokens count towards your API usage
   - Use judiciously for tasks that truly benefit from it

## Example Usage

```python
from api.llm_bridge import LLMBridge

# Initialize the bridge
llm = LLMBridge()

# Example: Complex mathematical problem
messages = [
    {"role": "user", "content": "Derive the formula for the sum of the first n cubes."}
]

# Enable extended thinking with a larger budget for this complex task
response = llm.generate_response(
    messages=messages,
    extended_thinking=True,
    extended_thinking_budget=8192,
    max_tokens=2000  # Ensure max_tokens accommodates thinking budget
)

result = llm.extract_text_content(response)
print(result)
```

## Best Practices

1. **Start Conservative:** Begin with the minimum budget (1024 tokens) and increase as needed.

2. **Monitor Usage:** Track how much of the allocated thinking budget Claude actually uses to optimize future requests.

3. **Use Streaming:** For better user experience, always use streaming when extended thinking is enabled to provide immediate feedback.

4. **Targeted Application:** Apply extended thinking selectively to specific components or tools that handle complex reasoning tasks.

5. **Testing:** Test with varied inputs to ensure extended thinking consistently improves results for your specific use cases.

## Technical Details

The implementation in LLMBridge adds the `thinking` parameter to the API request when extended thinking is enabled:

```json
{
  "model": "claude-3-5-sonnet-20240229",
  "messages": [ ... ],
  "max_tokens": 4000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 8192
  }
}
```

When extended thinking is enabled, the system automatically ensures that `max_tokens` is large enough to accommodate both the thinking budget and the desired response length.