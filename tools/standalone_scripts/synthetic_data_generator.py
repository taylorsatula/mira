"""
Synthetic Data Generator Vol2

A modular, maintainable system for generating synthetic training data
for tool classifiers through analysis of tool code and LLM generation.
"""

import json
import os
import argparse
import time
import sys
import logging
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

# Add parent directory to path for standalone execution
parent_dir = str(Path(__file__).parent.parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# Try to import from project config, but allow standalone operation
try:
    from config.config_manager import config
    HAS_PROJECT_CONFIG = True
except ImportError:
    HAS_PROJECT_CONFIG = False
    
# Set up root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class Configuration:
    """Configuration settings for synthetic data generation."""
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with sensible defaults, overridden by provided values.
        
        Args:
            **kwargs: Configuration overrides
        """
        # Set up API key
        self.api_key = kwargs.get('api_key')
        
        # Try project config
        if HAS_PROJECT_CONFIG and not self.api_key:
            try:
                self.api_key = config.api_key
            except Exception:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            
        if not self.api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY environment variable or pass as argument.")
            
        # Model settings
        if HAS_PROJECT_CONFIG:
            self.analysis_model = kwargs.get('analysis_model') or getattr(config.tools, 'synthetic_data_analysis_model', "claude-3-7-sonnet-20250219")
            self.generation_model = kwargs.get('generation_model') or getattr(config.tools, 'synthetic_data_generation_model', "claude-3-5-haiku-20240307")
            self.embedding_model = kwargs.get('embedding_model') or getattr(config.tools, 'synthetic_data_embedding_model', "all-MiniLM-L6-v2")
            self.data_dir = kwargs.get('data_dir') or getattr(config.paths, 'data_dir', "data")
        else:
            self.analysis_model = kwargs.get('analysis_model') or "claude-3-7-sonnet-20250219"  
            self.generation_model = kwargs.get('generation_model') or "claude-3-5-haiku-20240307"
            self.embedding_model = kwargs.get('embedding_model') or "all-MiniLM-L6-v2"
            self.data_dir = kwargs.get('data_dir') or "data"
            
        # Generation settings
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.99)  # 0.99 means only nearly identical examples are removed
        self.examples_per_temp = kwargs.get('examples_per_temp', 30)
        self.temperatures = kwargs.get('temperatures', [0.2, 0.8])  # Using lower and higher temperatures for better variety
        self.skip_llm_review = kwargs.get('skip_llm_review', False)


class LLMClient:
    """Client for interacting with the LLM API."""
    
    def __init__(self, api_key: str, model: str, logger=None):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM service
            model: Model identifier
            logger: Optional logger instance
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.logger = logger or logging.getLogger("LLMClient")
        
    def generate(self, prompt: str, system_prompt: str = None, 
               temperature: float = 0.7, max_tokens: int = 4000,
               thinking: bool = False) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens to generate
            thinking: Whether to enable extended thinking
            
        Returns:
            Generated text
        """
        try:
            system = system_prompt or "You are an expert in creating diverse, realistic training data for tool classification systems."
            
            # Parameters for the API call
            params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add thinking parameter if enabled
            if thinking:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 4000
                }
                self.logger.info("Using extended thinking mode with 4000 token budget")
            
            # Make the API call
            response = self.client.messages.create(**params)
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Error in LLM API call: {e}")
            raise


class ToolAnalyzer:
    """Analyzes tool code to extract operations and structure."""
    
    def __init__(self, llm_client, logger=None):
        """
        Initialize the analyzer.
        
        Args:
            llm_client: LLM client for operation extraction
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger("ToolAnalyzer")
        
    def analyze_tool(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a tool file to extract all relevant information.
        
        Args:
            file_path: Path to the tool file
            
        Returns:
            Dictionary with tool analysis
        """
        tool_code = self.read_tool_file(file_path)
        tool_name = self.extract_tool_name(file_path)
        
        self.logger.info(f"Analyzing tool: {tool_name}")
        
        # Step 1: Extract basic operations from code - no thinking needed
        self.logger.info(f"Step 1: Extracting operations from code")
        basic_analysis = self.extract_operations(tool_code, tool_name)
        tool_description = basic_analysis.get("description", "")
        operations = basic_analysis.get("operations", [])
        
        # Step 2: Analyze tool complexity and operation importance with extended thinking
        self.logger.info(f"Step 2: Analyzing tool complexity and operation importance")
        complexity_analysis = self.analyze_tool_complexity(
            tool_name=tool_name,
            tool_description=tool_description,
            operations=operations
        )
        
        # Merge the analyses
        result = {
            "tool_name": tool_name,
            "tool_code": tool_code,
            "description": tool_description,
            "operations": complexity_analysis.get("operations", operations),
            "complexity_category": complexity_analysis.get("tool_complexity", "standard"),
            "recommended_examples": complexity_analysis.get("recommended_examples", 45)
        }
        
        self.logger.info(f"Tool complexity category: {result['complexity_category']}")
        self.logger.info(f"Recommended examples: {result['recommended_examples']}")
        
        return result
    
    def read_tool_file(self, file_path: str) -> str:
        """
        Read a tool's Python file.
        
        Args:
            file_path: Path to the tool file
            
        Returns:
            String containing the file contents
        """
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading tool file {file_path}: {e}")
            raise
            
    def extract_tool_name(self, file_path: str) -> str:
        """
        Extract tool name from file path.
        
        Args:
            file_path: Path to the tool file
            
        Returns:
            Tool name (without .py extension)
        """
        return Path(file_path).stem
            
    def extract_operations(self, tool_code: str, tool_name: str) -> Dict[str, Any]:
        """
        Use LLM to extract operations from tool code without extended thinking.
        
        Args:
            tool_code: The tool's code as string
            tool_name: Name of the tool
            
        Returns:
            Dictionary with basic operations analysis
        """
        prompt = f"""Analyze this Python tool code and extract a comprehensive, structured summary of its operations.

```python
{tool_code}
```

Create a detailed analysis with the following format:

1. Tool Name: The name of the tool
2. Tool Description: A concise 1-2 sentence description of the overall tool purpose
3. Operations: A list of all operations the tool supports, with for each operation:
   - Name: The operation name
   - Description: Clear explanation of what this operation does (1-2 sentences)
   - Required Parameters: List each required parameter with its type and description
   - Optional Parameters: List each optional parameter with its type, description, and default value

Format your response as a JSON object with this structure:
```json
{{
  "tool_name": "string",
  "description": "string",
  "operations": [
    {{
      "name": "string",
      "description": "string",
      "required_parameters": [
        {{ "name": "string", "type": "string", "description": "string" }}
      ],
      "optional_parameters": [
        {{ "name": "string", "type": "string", "description": "string", "default": "any" }}
      ]
    }}
  ]
}}
```

IMPORTANT: 
1. Include ALL operations supported by the tool
2. Be comprehensive but concise in your descriptions
3. For parameters, include precise information on types and purposes
4. Ensure the JSON is valid and properly formatted"""

        system_prompt = "You are an expert Python code analyzer specializing in extracting structured information about tool functionality."
        
        try:
            # Call LLM for basic extraction without thinking budget
            response_text = self.llm_client.generate(
                prompt=prompt, 
                system_prompt=system_prompt,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                analysis = json.loads(json_str)
                return analysis
            else:
                self.logger.warning(f"Failed to extract JSON from LLM response for {tool_name}")
                return {
                    "tool_name": tool_name,
                    "description": "",
                    "operations": []
                }
                
        except Exception as e:
            self.logger.error(f"Error in operation extraction: {e}")
            return {
                "tool_name": tool_name,
                "description": "",
                "operations": []
            }
    
    def analyze_tool_complexity(self, tool_name: str, tool_description: str, operations: List[Dict]) -> Dict[str, Any]:
        """
        Analyze tool complexity and operation importance using extended thinking.
        
        Args:
            tool_name: Name of the tool
            tool_description: Description of the tool's purpose
            operations: List of operations extracted from code
            
        Returns:
            Dictionary with tool complexity category and classified operations
        """
        if not operations:
            return {
                "tool_complexity": "simple",
                "recommended_examples": 30,
                "operations": []
            }
            
        # Prepare operations for analysis
        operations_json = json.dumps(operations, indent=2)
        
        prompt = f"""Analyze the "{tool_name}" tool to determine:
1. Overall complexity category (simple, standard, or complex)
2. Each operation's relative importance

Tool description: {tool_description}

Here are all the operations extracted from the tool's code:
```json
{operations_json}
```

Part 1: Categorize the ENTIRE TOOL as:
- "simple" (30 examples needed): Few operations, straightforward functionality
- "standard" (45 examples needed): Average complexity, moderate number of operations
- "complex" (60 examples needed): Many operations or sophisticated functionality

Part 2: For each operation, classify its importance as:
- "core": Essential operations users frequently request (examples: checking email, sending messages)
- "standard": Regular functionality used occasionally
- "fringe": Specialized or rarely used functions

Part 3: Assign each operation a relative weight on a scale of 1-10:
- 8-10: Core operations that will be frequently requested by users
- 4-7: Standard operations that will be occasionally requested
- 1-3: Fringe operations that will rarely be directly requested

Analyze how users would actually interact with this tool in practice.
Consider which operations represent the tool's main purpose vs. supporting functions.
Core operations should get more examples to ensure good coverage.

Return your analysis as JSON:
{{
  "tool_complexity": "simple|standard|complex",
  "recommended_examples": 30-60,
  "operations": [
    {{
      "name": "operation_name",
      "importance": "core|standard|fringe",
      "relative_weight": 1-10,
      "justification": "Brief explanation of classification"
    }}
  ]
}}"""

        system_prompt = "You are an expert in analyzing software tools and classifying functions by importance. Prioritize user-facing operations that align with the tool's main purpose."
        
        try:
            # Use the generate method with thinking enabled
            response_text = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=4000,
                thinking=True  # Enable extended thinking
            )
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                complexity_analysis = json.loads(json_str)
                
                # Map complexity to examples count if not provided
                if "recommended_examples" not in complexity_analysis:
                    complexity = complexity_analysis.get("tool_complexity", "standard").lower()
                    if complexity == "simple":
                        complexity_analysis["recommended_examples"] = 30
                    elif complexity == "complex":
                        complexity_analysis["recommended_examples"] = 60
                    else:
                        complexity_analysis["recommended_examples"] = 45
                
                # Validate and normalize weights
                self._normalize_operation_weights(complexity_analysis)
                
                # Log results
                self.logger.info(f"Tool complexity: {complexity_analysis.get('tool_complexity', 'standard')}")
                self.logger.info(f"Recommended examples: {complexity_analysis.get('recommended_examples', 45)}")
                for op in complexity_analysis.get("operations", []):
                    self.logger.info(f"Operation '{op.get('name', '')}': {op.get('importance', 'standard')} " +
                                   f"(weight: {op.get('relative_weight', 0)})")
                
                return complexity_analysis
            else:
                self.logger.warning("Failed to extract complexity analysis")
                return {
                    "tool_complexity": "standard",
                    "recommended_examples": 45,
                    "operations": operations
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing tool complexity: {e}")
            return {
                "tool_complexity": "standard",
                "recommended_examples": 45,
                "operations": operations
            }
    
    def _normalize_operation_weights(self, analysis: Dict[str, Any]) -> None:
        """
        Ensure operation weights are valid and normalized.
        
        Args:
            analysis: Tool complexity analysis
        """
        if "operations" not in analysis:
            return
            
        operations = analysis["operations"]
        
        # Ensure all operations have valid weights
        for op in operations:
            if "relative_weight" not in op or not isinstance(op["relative_weight"], (int, float)):
                op["relative_weight"] = 5  # Default to middle weight
            else:
                # Ensure weight is in range 1-10
                op["relative_weight"] = max(1, min(10, op["relative_weight"]))
                
            # Ensure all operations have importance
            if "importance" not in op:
                weight = op["relative_weight"]
                if weight >= 8:
                    op["importance"] = "core"
                elif weight >= 4:
                    op["importance"] = "standard"
                else:
                    op["importance"] = "fringe"


class PromptBuilder:
    """Builds prompts for different generation scenarios."""
    
    def create_operation_prompt(self, tool_name: str, operation: Dict[str, Any], 
                              tool_description: str = "",
                              num_examples: int = 10, temp: float = 0.7) -> str:
        """
        Create a prompt for generating examples for a specific operation.
        
        Args:
            tool_name: Name of the tool
            operation: Operation details
            tool_description: Description of the tool
            num_examples: Number of examples to request
            temp: Temperature setting
            
        Returns:
            Formatted prompt
        """
        # Style guide based on temperature
        style_guide = self._get_style_guide(temp)
        
        # Extract operation details
        operation_name = operation.get("name", "")
        operation_description = operation.get("description", "")
        
        # Format parameters
        parameters_text = self._format_parameters(operation)
        
        # Create the prompt
        prompt = f"""You are creating a training dataset for a classifier that identifies when users request the '{operation_name}' operation of the '{tool_name}' tool.

## Tool & Operation Details
Tool: {tool_name}
Tool Description: {tool_description}

Operation: {operation_name}
Operation Description: {operation_description}
{parameters_text}

## Your Task
Generate {num_examples} diverse, realistic user queries that would trigger this SPECIFIC operation.

{style_guide}

## Guidelines for Creating Effective Training Data
1. Ensure each example CLEARLY belongs to the '{operation_name}' operation with no ambiguity
2. Create natural-sounding queries that real users would actually write
3. Vary the phrasing, structure, and vocabulary across examples
4. Include various ways users might specify the operation's parameters
5. Create a mix of direct/explicit requests and more indirect/implicit requests
6. Vary query length - include both short, direct queries and longer ones with context
7. IMPORTANT: Do NOT excessively use the operation name or tool name in examples - users rarely refer to operations by their exact technical names
8. Focus on the functional need rather than naming the operation explicitly
9. Avoid patterns that would make examples easy to classify using simplistic rules

## Important Classifier Considerations
1. Your examples will be used to train a classifier to recognize when users need this specific operation
2. Examples should describe what the user wants to accomplish, not necessarily using the exact operation terminology
3. Real users describe their goals and needs, not the technical operation names
4. Do not rely on explicit mentions of the operation name - that makes classification too easy and unrealistic

## Output Format
Return a JSON array of objects, each with a "query" field containing the user query:
```json
[
  {{"query": "Example query describing the need"}},
  {{"query": "Another example query"}},
  ...
]
```

Remember that each query MUST clearly indicate the user wants to perform the functionality of the '{operation_name}' operation, but should vary naturally in how this intent is expressed using everyday language."""

        return prompt
    
    def create_tool_prompt(self, tool_analysis: Dict[str, Any], 
                         num_examples: int = 30, temp: float = 0.7,
                         variation: str = "") -> str:
        """
        Create a prompt for generating examples for an entire tool.
        
        Args:
            tool_analysis: Tool analysis
            num_examples: Number of examples to request
            temp: Temperature setting
            variation: Optional variation instructions
            
        Returns:
            Formatted prompt
        """
        # Style guide based on temperature
        style_guide = self._get_style_guide(temp)
        if variation:
            style_guide += f" {variation}"
        
        # Extract tool details
        tool_name = tool_analysis.get("tool_name", "")
        tool_description = tool_analysis.get("description", "")
        
        # Format operations section
        operations_text = self._format_operations_list(tool_analysis.get("operations", []))
        
        # Create the prompt
        prompt = f"""You are creating a training dataset for a classifier that identifies when users need the '{tool_name}' tool.

## Tool Details
Tool Name: {tool_name}
Tool Description: {tool_description}

{operations_text}

## Your Task
Generate {num_examples} diverse, realistic user queries that would clearly indicate the user needs this tool.
{style_guide}

## Guidelines for Creating Effective Training Data
1. Cover ALL operations in a BALANCED way - don't favor one operation over others
2. Create natural-sounding queries that real users would actually write
3. Vary the phrasing, structure, and vocabulary across examples
4. Mix specific technical requests with everyday language approaches
5. Include common real-world scenarios where this tool would be useful
6. Vary query length and complexity - include both simple and detailed requests
7. IMPORTANT: Do NOT excessively use the tool name in the examples - users rarely mention the tool by name
8. Focus on the functionality and intent rather than explicitly naming the tool
9. Avoid creating patterns that would make classification artificially easy

## Important Classifier Considerations
1. Your examples will be used to train a classifier to recognize when users need this tool
2. Examples should be clearly indicative of needing this tool's functionality
3. Avoid ambiguous examples that could apply to multiple different tools
4. Do not rely on the user explicitly asking for the tool by name - real users describe what they want to do, not which tool to use

## Output Format
Return a JSON array of objects, each with a "query" field containing the user query:
```json
[
  {{"query": "Example query that needs the functionality"}},
  {{"query": "Another example query"}},
  ...
]
```

The goal is to create training data that will help a classifier accurately recognize when users need this specific tool, regardless of how they phrase their request."""

        return prompt
    
    def create_multi_tool_prompt(self, tool_names: List[str], 
                               num_examples: int = 20, 
                               variation: str = "") -> str:
        """
        Create a prompt for generating multi-tool examples.
        
        Args:
            tool_names: List of tool names (typically a pair)
            num_examples: Number of examples to request
            variation: Optional variation instructions
            
        Returns:
            Formatted prompt
        """
        tools_formatted = "\n".join([f"{i+1}. {name}" for i, name in enumerate(tool_names)])
        
        variation_text = f"\n{variation}" if variation else ""
        
        prompt = f"""I need to create training data for scenarios where a user request requires MULTIPLE different tools to fulfill.

The tools are:
{tools_formatted}

Please generate {num_examples} diverse user queries that would require ALL of these tools working together.
These should be genuine cases where a single request naturally needs these tools, not forced combinations.{variation_text}

Some guidelines:
- Create realistic scenarios where all tools are genuinely needed
- Vary the phrasing, structure, and vocabulary
- Include different ways users might naturally request functionality that spans multiple tools
- Focus on requests that make practical sense in real-world usage

Format your response as a JSON array of objects, each with a "query" field:
```json
[
  {{"query": "Example multi-tool query 1"}},
  {{"query": "Example multi-tool query 2"}},
  ...
]
```"""

        return prompt
    
    def create_review_prompt(self, examples: List[Dict[str, str]], tool_name: str, tool_description: str = "") -> str:
        """
        Create a prompt for LLM to review examples for quality.
        
        Args:
            examples: List of examples to review
            tool_name: Name of the tool
            tool_description: Description of the tool (optional)
            
        Returns:
            Formatted prompt
        """
        # Format examples as numbered list
        examples_text = ""
        for i, example in enumerate(examples):
            query = example.get("query", "")
            examples_text += f"{i+1}. \"{query}\"\n"
        
        # Add tool description if available
        tool_context = f"Tool Description: {tool_description}\n\n" if tool_description else ""
        
        prompt = f"""I have generated training examples for a classifier that identifies when a user's message should trigger the '{tool_name}' tool.

## Tool Information
Tool Name: {tool_name}
{tool_context}

## Examples to Review
Below is a numbered list of {len(examples)} generated examples. Please review these and identify any examples that are clearly problematic.

{examples_text}

## Evaluation Guidelines
Only flag examples that have SIGNIFICANT issues such as:
1. Completely unrelated to the tool's likely functionality (based on its name and description)
2. So poorly worded that the meaning is unclear
3. Not a realistic user request (e.g., sounds like documentation rather than a query)
4. Malformed or contains errors that would confuse a classifier

IMPORTANT:
- Direct requests to control or manipulate something are VALID use cases for tools
- Requests for information about the status of something are VALID
- Requests to modify settings or parameters are VALID
- Accept examples that express user intent to control, monitor, or get information
- Both command-like statements and question-like queries can be valid
- A spectrum of example specificity helps build a robust classifier
- It's acceptable for examples to vary in complexity - some can be simple and direct
- Without detailed knowledge of the tool, be conservative in judging relevance
- Don't flag examples just because they explicitly mention the tool name - this is acceptable

## Output Format
Please respond with a JSON object that has a "removed_indices" array containing the indices of examples that should be removed:
```json
{{
  "removed_indices": [3, 7, 12],
  "reasons": {{
    "3": "Unrelated to tool functionality",
    "7": "Not a realistic user request",
    "12": "Meaning unclear"
  }}
}}
```

If all examples look good, return an empty array:
```json
{{
  "removed_indices": [],
  "reasons": {{}}
}}
```"""

        return prompt
    
    def _get_style_guide(self, temperature: float) -> str:
        """Get appropriate style guide based on temperature."""
        if temperature < 0.5:
            return "Focus on common, straightforward ways users would request this functionality. Use clear, direct language."
        elif temperature < 0.9:
            return "Create a mix of common and varied phrasings. Include some creative variations while keeping queries realistic."
        else:
            return "Create highly diverse phrasings with varied sentence structures. Include unusual but valid ways users might express these requests."
    
    def _format_parameters(self, operation: Dict[str, Any]) -> str:
        """Format operation parameters for prompt."""
        result = ""
        
        # Add required parameters
        if "required_parameters" in operation and operation["required_parameters"]:
            result += "Required parameters:\n"
            for param in operation["required_parameters"]:
                result += f"- {param.get('name', '')}: {param.get('description', '')}\n"
        
        # Add optional parameters
        if "optional_parameters" in operation and operation["optional_parameters"]:
            result += "Optional parameters:\n"
            for param in operation["optional_parameters"]:
                default = f" (default: {param.get('default', 'None')})"
                result += f"- {param.get('name', '')}: {param.get('description', '')}{default}\n"
        
        return result
    
    def _format_operations_list(self, operations: List[Dict[str, Any]]) -> str:
        """Format operations list for prompt."""
        if not operations:
            return ""
            
        result = "Tool Operations:\n\n"
        
        for i, op in enumerate(operations):
            result += f"{i+1}. {op.get('name', '')}: {op.get('description', '')}\n"
            
            # Add parameters
            if "required_parameters" in op and op["required_parameters"]:
                result += "   Required parameters:\n"
                for param in op["required_parameters"]:
                    result += f"   - {param.get('name', '')}: {param.get('description', '')}\n"
            
            if "optional_parameters" in op and op["optional_parameters"]:
                result += "   Optional parameters:\n"
                for param in op["optional_parameters"]:
                    default = f" (default: {param.get('default', 'None')})"
                    result += f"   - {param.get('name', '')}: {param.get('description', '')}{default}\n"
            
            result += "\n"
        
        return result


class ResponseParser:
    """Parses LLM responses into structured data."""
    
    def __init__(self, logger=None):
        """
        Initialize the parser.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("ResponseParser")
    
    def parse_examples(self, response_text: str, tool_name: str) -> List[Dict[str, str]]:
        """
        Parse examples from LLM response.
        
        Args:
            response_text: LLM response text
            tool_name: Name of the tool
            
        Returns:
            List of examples
        """
        examples = []
        
        try:
            # Extract JSON array from response
            json_array = self._extract_json_array(response_text)
            
            if json_array:
                # Convert to standard format
                for item in json_array:
                    if "query" in item:
                        examples.append({
                            "tool_name": tool_name,
                            "query": item["query"].strip()
                        })
            else:
                # Fallback: look for numbered list
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for patterns like "1. Query text" or "- Query text"
                    if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', '-', '*')) and 
                        len(line) > 2):
                        query = line.split(' ', 1)[1].strip()
                        if query and len(query) > 5:  # Basic validation
                            examples.append({
                                "tool_name": tool_name,
                                "query": query
                            })
        
        except Exception as e:
            self.logger.error(f"Error parsing examples: {e}")
        
        return examples
    
    def parse_multi_tool_examples(self, response_text: str, tool_names: List[str]) -> List[Dict]:
        """
        Parse multi-tool examples from LLM response.
        
        Args:
            response_text: LLM response text
            tool_names: List of tool names used
            
        Returns:
            List of multi-tool examples
        """
        examples = []
        
        try:
            # Extract JSON array from response
            json_array = self._extract_json_array(response_text)
            
            if json_array:
                # Convert to multi-tool format
                for item in json_array:
                    if "query" in item:
                        examples.append({
                            "tool_names": tool_names,
                            "query": item["query"].strip()
                        })
            
        except Exception as e:
            self.logger.error(f"Error parsing multi-tool examples: {e}")
        
        return examples
    
    def parse_review_results(self, response_text: str) -> Set[int]:
        """
        Parse review results to get indices to remove.
        
        Args:
            response_text: LLM response text
            
        Returns:
            Set of indices to remove
        """
        removed_indices = set()
        
        try:
            # Extract JSON object from response
            json_obj = self._extract_json_object(response_text)
            
            if json_obj and "removed_indices" in json_obj:
                removed_indices = set(json_obj["removed_indices"])
                
                # Print reasons if available
                if "reasons" in json_obj:
                    for idx, reason in json_obj["reasons"].items():
                        self.logger.info(f"Removing example {idx}: {reason}")
            else:
                # Fallback: look for numbers in the response
                import re
                numbers = re.findall(r'\b\d+\b', response_text)
                for num in numbers:
                    try:
                        removed_indices.add(int(num))
                    except ValueError:
                        pass
                        
        except Exception as e:
            self.logger.error(f"Error parsing review results: {e}")
        
        return removed_indices
    
    def _extract_json_array(self, text: str) -> List[Dict]:
        """Extract JSON array from text."""
        import json
        import re
        
        # Find JSON array in the text
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON array")
        
        return []
    
    def _extract_json_object(self, text: str) -> Dict:
        """Extract JSON object from text."""
        import json
        import re
        
        # Find JSON object in the text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON object")
        
        return {}


class ExampleProcessor:
    """Processes, deduplicates, and filters examples."""
    
    def __init__(self, embedding_model, similarity_threshold: float = 0.99, logger=None):
        """
        Initialize the processor.
        
        Args:
            embedding_model: Model for calculating embeddings
            similarity_threshold: Threshold for similarity detection (0.99 means only nearly identical examples are removed)
            logger: Optional logger instance
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.logger = logger or logging.getLogger("ExampleProcessor")
    
    def deduplicate(self, examples: List[Dict], query_field: str = "query") -> List[Dict]:
        """
        Remove near-duplicate examples based on semantic similarity.
        
        Args:
            examples: List of examples
            query_field: Field containing the query text
            
        Returns:
            Deduplicated examples
        """
        if not examples:
            return []
        
        self.logger.info(f"Deduplicating {len(examples)} examples...")
        
        # Extract queries
        queries = [ex[query_field] for ex in examples]
        
        # Calculate embeddings
        embeddings = self.embedding_model.encode(queries)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Track which examples to keep and discard
        keep_indices = set()
        discard_indices = set()
        
        # Greedy deduplication algorithm
        for i in range(len(examples)):
            # Skip if we've already decided to discard this example
            if i in discard_indices:
                continue
                
            # Keep this example
            keep_indices.add(i)
            
            # Compare with remaining examples
            for j in range(i+1, len(examples)):
                # Skip if we've already decided to keep or discard this example
                if j in keep_indices or j in discard_indices:
                    continue
                    
                # Calculate similarity
                similarity = np.dot(embeddings[i], embeddings[j])
                
                # Discard if too similar
                if similarity > self.similarity_threshold:
                    discard_indices.add(j)
        
        # Create filtered list
        filtered_examples = [examples[i] for i in sorted(keep_indices)]
        
        # Log the actual counts
        self.logger.info(f"Deduplication complete. Kept {len(filtered_examples)} examples, discarded {len(discard_indices)} duplicates.")
        return filtered_examples
    
    def filter_by_review(self, examples: List[Dict], removed_indices: Set[int]) -> List[Dict]:
        """
        Filter examples based on review results.
        
        Args:
            examples: List of examples
            removed_indices: Indices to remove (1-based)
            
        Returns:
            Filtered examples
        """
        if not examples or not removed_indices:
            return examples
            
        filtered = []
        removed_count = 0
        
        for i, example in enumerate(examples):
            if i+1 in removed_indices:  # +1 because indices from review are 1-based
                removed_count += 1
            else:
                filtered.append(example)
        
        self.logger.info(f"Filtered {removed_count} examples based on review.")
        return filtered
    
    def batch_examples(self, examples: List[Dict], batch_size: int = 50) -> List[List[Dict]]:
        """
        Split examples into batches for processing.
        
        Args:
            examples: List of examples
            batch_size: Size of each batch
            
        Returns:
            List of batches
        """
        return [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]


class OutputManager:
    """Manages output files and directories."""
    
    def __init__(self, data_dir: str, logger=None):
        """
        Initialize the output manager.
        
        Args:
            data_dir: Base data directory
            logger: Optional logger instance
        """
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger("OutputManager")
    
    def save_tool_analysis(self, tool_name: str, analysis: Dict) -> str:
        """
        Save tool analysis to file.
        
        Args:
            tool_name: Name of the tool
            analysis: Analysis data
            
        Returns:
            Path to saved file
        """
        # Create directory
        tool_dir = Path(self.data_dir) / "tools" / tool_name
        os.makedirs(tool_dir, exist_ok=True)
        
        # Save file
        file_path = tool_dir / "tool_analysis.json"
        with open(file_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        self.logger.info(f"Saved tool analysis to {file_path}")
        return str(file_path)
    
    def save_examples(self, tool_name: str, examples: List[Dict], 
                    prefix: str = "autogen") -> str:
        """
        Save examples to file.
        
        Args:
            tool_name: Name of the tool
            examples: Examples to save
            prefix: File name prefix
            
        Returns:
            Path to saved file
        """
        # Create directory
        tool_dir = Path(self.data_dir) / "tools" / tool_name
        os.makedirs(tool_dir, exist_ok=True)
        
        # Create filename
        filename = f"{prefix}_classifier_examples.json"
        file_path = tool_dir / filename
        
        # Save file
        with open(file_path, 'w') as f:
            json.dump(examples, f, indent=2)
            
        self.logger.info(f"Saved {len(examples)} examples to {file_path}")
        return str(file_path)
    
    def save_multi_tool_examples(self, examples: List[Dict]) -> str:
        """
        Save multi-tool examples to file.
        
        Args:
            examples: Examples to save
            
        Returns:
            Path to saved file
        """
        # Create directory
        multi_tool_dir = Path(self.data_dir) / "tools" / "multi_tool"
        os.makedirs(multi_tool_dir, exist_ok=True)
        
        # Create filename
        file_path = multi_tool_dir / "multi_tool_examples.json"
        
        # Save file
        with open(file_path, 'w') as f:
            json.dump(examples, f, indent=2)
            
        self.logger.info(f"Saved {len(examples)} multi-tool examples to {file_path}")
        return str(file_path)


class ExampleGenerator:
    """Generates examples using LLM."""
    
    def __init__(self, llm_client, prompt_builder, response_parser, logger=None):
        """
        Initialize the generator.
        
        Args:
            llm_client: LLM client
            prompt_builder: Prompt builder
            response_parser: Response parser
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.logger = logger or logging.getLogger("ExampleGenerator")
    
    def generate_for_operation(self, tool_name: str, operation: Dict[str, Any],
                             tool_description: str = "",
                             num_examples: int = 10, temperature: float = 0.7) -> List[Dict[str, str]]:
        """
        Generate examples for a specific operation.
        
        Args:
            tool_name: Name of the tool
            operation: Operation details
            tool_description: Description of the tool
            num_examples: Number of examples to generate
            temperature: Temperature setting
            
        Returns:
            List of examples
        """
        operation_name = operation.get("name", "unknown")
        self.logger.info(f"Generating examples for operation '{operation_name}' at temperature {temperature}")
        
        # Create prompt
        prompt = self.prompt_builder.create_operation_prompt(
            tool_name=tool_name,
            operation=operation,
            tool_description=tool_description,
            num_examples=num_examples,
            temp=temperature
        )
        
        # Generate examples
        try:
            response_text = self.llm_client.generate(
                prompt=prompt,
                temperature=temperature
            )
            
            # Parse examples
            examples = self.response_parser.parse_examples(response_text, tool_name)
            
            self.logger.info(f"Generated {len(examples)} examples for operation '{operation_name}'")
            return examples
            
        except Exception as e:
            self.logger.error(f"Error generating examples for operation: {e}")
            return []
    
    def generate_for_tool(self, tool_analysis: Dict[str, Any],
                        examples_per_temp: int = 30,
                        temperatures: List[float] = [0.3, 0.7, 1.0]) -> List[Dict[str, str]]:
        """
        Generate examples for an entire tool.
        
        Args:
            tool_analysis: Tool analysis
            examples_per_temp: Number of examples per temperature
            temperatures: List of temperatures
            
        Returns:
            List of examples
        """
        import time
        import concurrent.futures
        
        tool_name = tool_analysis.get("tool_name", "")
        tool_description = tool_analysis.get("description", "")
        operations = tool_analysis.get("operations", [])
        all_examples = []
        
        self.logger.info(f"Generating examples for tool '{tool_name}'")
        
        # If we have operations, generate examples per operation using the priority weights
        if operations:
            for temp in temperatures:
                self.logger.info(f"Generating at temperature {temp}")
                
                # Calculate examples based on operation priority weights
                operation_counts = self._calculate_operation_examples(
                    operations=operations, 
                    total_examples=examples_per_temp
                )
                
                # Log the distribution
                for op_name, count in operation_counts.items():
                    importance = next((op.get("importance", "standard") for op in operations 
                                     if op.get("name", "") == op_name), "standard")
                    self.logger.info(f"Operation '{op_name}' ({importance}): {count} examples")
                
                # Generate examples for each operation in parallel
                operation_examples = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(operations))) as executor:
                    futures = []
                    
                    for operation in operations:
                        op_name = operation.get("name", "unknown")
                        num_examples = operation_counts.get(op_name, 5) # Default to 5 if missing
                        
                        # Skip if no examples needed
                        if num_examples <= 0:
                            continue
                            
                        future = executor.submit(
                            self.generate_for_operation,
                            tool_name=tool_name,
                            operation=operation,
                            tool_description=tool_description,
                            num_examples=num_examples,
                            temperature=temp
                        )
                        futures.append((future, op_name))
                    
                    for future, op_name in futures:
                        try:
                            examples = future.result()
                            operation_examples.append(examples)
                            self.logger.info(f"Generated {len(examples)} examples for operation '{op_name}'")
                        except Exception as e:
                            self.logger.error(f"Error generating examples for operation '{op_name}': {e}")
                
                # Add all operation examples to the result
                for examples in operation_examples:
                    all_examples.extend(examples)
                
                # Generate some general examples (20% of total)
                general_count = max(5, int(examples_per_temp * 0.2))
                
                prompt = self.prompt_builder.create_tool_prompt(
                    tool_analysis=tool_analysis,
                    num_examples=general_count,
                    temp=temp,
                    variation="Create diverse examples that don't focus on any specific operation."
                )
                
                try:
                    response_text = self.llm_client.generate(
                        prompt=prompt,
                        temperature=temp
                    )
                    
                    # Parse examples
                    examples = self.response_parser.parse_examples(response_text, tool_name)
                    all_examples.extend(examples)
                    
                    self.logger.info(f"Generated {len(examples)} general examples")
                    
                except Exception as e:
                    self.logger.error(f"Error generating general examples: {e}")
        
        # If no operations, generate examples for the whole tool
        else:
            for temp in temperatures:
                self.logger.info(f"Generating at temperature {temp}")
                
                variations = [
                    "",  # No variation for first call
                    "Focus on different phrasings than previous examples.",
                    "Create examples with more complex syntax and varied vocabulary."
                ]
                
                # Generate examples for each variation in parallel
                variation_examples = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(variations)) as executor:
                    variation_futures = []
                    
                    for i, variation in enumerate(variations):
                        prompt = self.prompt_builder.create_tool_prompt(
                            tool_analysis=tool_analysis,
                            num_examples=examples_per_temp // len(variations),
                            temp=temp,
                            variation=variation
                        )
                        
                        future = executor.submit(self._process_variation, prompt, temp, tool_name, i)
                        variation_futures.append((future, i))
                    
                    for future, i in variation_futures:
                        try:
                            examples = future.result()
                            variation_examples.append(examples)
                            self.logger.info(f"Generated {len(examples)} examples (variation {i+1})")
                        except Exception as e:
                            self.logger.error(f"Error generating examples for variation {i+1}: {e}")
                
                # Add all variation examples to the result
                for examples in variation_examples:
                    all_examples.extend(examples)
        
        self.logger.info(f"Generated {len(all_examples)} total examples for tool '{tool_name}'")
        return all_examples
        
    def _calculate_operation_examples(self, operations: List[Dict], total_examples: int) -> Dict[str, int]:
        """
        Calculate how many examples to generate for each operation based on semantic coverage.
        
        Args:
            operations: List of operations with priority weights
            total_examples: Total examples to distribute
            
        Returns:
            Dictionary mapping operation names to example counts
        """
        # Constants for allocation control
        ABSOLUTE_MAX = 60  # Maximum examples regardless of tool complexity
        GENERAL_EXAMPLES_RATIO = 0.15  # Reduce from 20% to 15% for general examples
        
        # Adjust constants based on operation count to prevent example explosion
        if len(operations) > 15:
            self.logger.warning(f"Large number of operations detected ({len(operations)}). Adjusting allocation strategy.")
            GENERAL_EXAMPLES_RATIO = 0.1  # Further reduce general examples for large operation sets
        
        # Calculate minimum viable examples for operations by importance
        min_examples = {
            "core": 4,     # Core operations need good coverage
            "standard": 2,  # Standard operations need basic coverage
            "fringe": 1     # Fringe operations need minimal representation
        }
        
        # Step 1: Group operations by semantic similarity to detect overlap
        semantic_groups = self._group_similar_operations(operations)
        self.logger.info(f"Grouped {len(operations)} operations into {len(semantic_groups)} semantic clusters")
        
        # Step 2: Initial allocation - minimum viable examples per operation
        result = {}
        allocated = 0
        
        for op in operations:
            name = op.get("name", "unknown")
            importance = op.get("importance", "standard")
            min_count = min_examples.get(importance, 2)
            
            # Store initial allocation
            result[name] = min_count
            allocated += min_count
        
        # Calculate remaining budget after minimum allocations
        general_examples = int(total_examples * GENERAL_EXAMPLES_RATIO)
        remaining = max(0, total_examples - general_examples - allocated)
        
        # Step 3: Distribute remaining budget by weighted importance
        if remaining > 0:
            # Calculate weights from relative_weight scores
            weights = {}
            total_weight = 0
            
            for op in operations:
                name = op.get("name", "unknown")
                # Use relative_weight but scale down for operations in large semantic groups
                group_size = next((len(g) for g in semantic_groups if name in g), 1)
                group_factor = 1.0 / math.sqrt(group_size)  # Scale down by square root of group size
                
                # Calculate adjusted weight
                base_weight = op.get("relative_weight", 5)
                adjusted_weight = base_weight * group_factor
                
                weights[name] = adjusted_weight
                total_weight += adjusted_weight
            
            # Distribute remaining examples proportionally by adjusted weight
            for name, weight in weights.items():
                # Calculate proportional allocation
                additional = int(remaining * (weight / total_weight)) if total_weight > 0 else 0
                result[name] += additional
        
        # Step 4: Check if we've exceeded our absolute maximum and scale down if needed
        total_operation_examples = sum(result.values())
        if total_operation_examples > ABSOLUTE_MAX - general_examples:
            scale_factor = (ABSOLUTE_MAX - general_examples) / total_operation_examples
            
            # Scale down but preserve minimum allocations
            for name in result:
                op = next((o for o in operations if o.get("name", "") == name), {})
                importance = op.get("importance", "standard")
                min_count = min_examples.get(importance, 2)
                
                # Scale down but don't go below minimum for importance level
                result[name] = max(min_count, int(result[name] * scale_factor))
            
            # Adjust total examples to match our cap
            total_examples = min(ABSOLUTE_MAX, total_examples)
        
        self.logger.info(f"Example allocation complete: {sum(result.values())} operation examples, " +
                       f"{general_examples} general examples, {sum(result.values()) + general_examples} total")
        
        # Log allocation by importance
        core_count = sum(result.get(op.get("name", "")) for op in operations if op.get("importance", "") == "core")
        std_count = sum(result.get(op.get("name", "")) for op in operations if op.get("importance", "") == "standard")
        fringe_count = sum(result.get(op.get("name", "")) for op in operations if op.get("importance", "") == "fringe")
        
        self.logger.info(f"Distribution: {core_count} core, {std_count} standard, {fringe_count} fringe")
        
        return result
    
    def _group_similar_operations(self, operations: List[Dict]) -> List[Set[str]]:
        """
        Group operations by semantic similarity to identify functional overlap.
        
        Args:
            operations: List of operations
            
        Returns:
            List of sets, where each set contains names of semantically similar operations
        """
        # Simple implementation using name and description similarity
        # In a full implementation, we would use embeddings for better semantic grouping
        groups = []
        processed = set()
        
        import re
        import math
        
        # Helper function to normalize text for comparison
        def normalize(text):
            return re.sub(r'[^a-z0-9]', ' ', text.lower())
        
        # Helper function to check if two operations are semantically similar
        def is_similar(op1, op2):
            # Check name similarity
            name1 = normalize(op1.get("name", ""))
            name2 = normalize(op2.get("name", ""))
            
            # Operations on same entity are likely similar
            # e.g., get_user, update_user, delete_user would share 'user'
            words1 = set(name1.split('_'))
            words2 = set(name2.split('_'))
            shared_words = words1.intersection(words2)
            
            # If they share entity terms and one is a common verb (get/set/list/etc)
            common_verbs = {'get', 'set', 'list', 'update', 'delete', 'create', 'add', 'remove'}
            if (shared_words and 
                (words1.intersection(common_verbs) or words2.intersection(common_verbs))):
                return True
                
            # Description similarity as backup
            desc1 = normalize(op1.get("description", ""))
            desc2 = normalize(op2.get("description", ""))
            
            # Very basic heuristic: check word overlap ratio
            if desc1 and desc2:
                words1 = set(desc1.split())
                words2 = set(desc2.split())
                
                if not words1 or not words2:
                    return False
                    
                overlap = len(words1.intersection(words2))
                overlap_ratio = overlap / math.sqrt(len(words1) * len(words2))
                return overlap_ratio > 0.5
                
            return False
        
        # Group operations
        for i, op1 in enumerate(operations):
            name1 = op1.get("name", "")
            if name1 in processed:
                continue
                
            # Start a new group
            group = {name1}
            processed.add(name1)
            
            # Find similar operations
            for j, op2 in enumerate(operations):
                if i == j:
                    continue
                    
                name2 = op2.get("name", "")
                if name2 in processed:
                    continue
                    
                if is_similar(op1, op2):
                    group.add(name2)
                    processed.add(name2)
            
            groups.append(group)
        
        # Handle any remaining ungrouped operations
        ungrouped = {op.get("name", "") for op in operations} - processed
        for name in ungrouped:
            groups.append({name})
            
        return groups
        
    def _process_variation(self, prompt: str, temperature: float, tool_name: str, variation_index: int) -> List[Dict[str, str]]:
        """Process a single variation prompt."""
        try:
            response_text = self.llm_client.generate(
                prompt=prompt,
                temperature=temperature
            )
            
            # Parse examples
            return self.response_parser.parse_examples(response_text, tool_name)
            
        except Exception as e:
            self.logger.error(f"Error in variation {variation_index+1}: {e}")
            return []
    
    def generate_multi_tool(self, tool_combinations: List[Tuple[str, ...]],
                          examples_per_combo: int = 20) -> List[Dict]:
        """
        Generate examples for multi-tool combinations.
        
        Args:
            tool_combinations: List of tool name combinations
            examples_per_combo: Number of examples per combination
            
        Returns:
            List of multi-tool examples
        """
        import concurrent.futures
        
        all_examples = []
        tasks = []
        
        # Prepare all tasks for parallel execution
        for tool_combo in tool_combinations:
            self.logger.info(f"Preparing tasks for combination: {' + '.join(tool_combo)}")
            
            # Different variations for diversity
            variations = [
                "",  # No variation
                "Focus on scenarios where the tools are needed in sequence.",
                "Generate examples where the user implicitly needs all tools in a single request."
            ]
            
            for i, variation in enumerate(variations):
                examples_this_batch = examples_per_combo // len(variations)
                
                # Skip if we need 0 examples
                if examples_this_batch <= 0:
                    continue
                
                tasks.append({
                    "tool_combo": tool_combo,
                    "variation": variation,
                    "variation_index": i,
                    "examples_count": examples_this_batch
                })
        
        # Process tasks in parallel
        self.logger.info(f"Processing {len(tasks)} multi-tool generation tasks in parallel")
        
        # Define function to process each task
        def process_task(task):
            tool_combo = task["tool_combo"]
            variation = task["variation"]
            variation_index = task["variation_index"]
            examples_count = task["examples_count"]
            
            # Create prompt
            prompt = self.prompt_builder.create_multi_tool_prompt(
                tool_names=list(tool_combo),
                num_examples=examples_count,
                variation=variation
            )
            
            try:
                # Generate examples
                response_text = self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.7  # Fixed temperature for multi-tool
                )
                
                # Parse examples
                examples = self.response_parser.parse_multi_tool_examples(
                    response_text=response_text,
                    tool_names=list(tool_combo)
                )
                
                self.logger.info(f"Generated {len(examples)} examples for {' + '.join(tool_combo)} (variation {variation_index+1})")
                return examples
                
            except Exception as e:
                self.logger.error(f"Error generating multi-tool examples for {' + '.join(tool_combo)}: {e}")
                return []
        
        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(tasks))) as executor:
            future_to_task = {executor.submit(process_task, task): task for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    examples = future.result()
                    all_examples.extend(examples)
                except Exception as e:
                    tool_combo = ' + '.join(task["tool_combo"])
                    self.logger.error(f"Task for {tool_combo} (variation {task['variation_index']+1}) failed: {e}")
        
        self.logger.info(f"Generated {len(all_examples)} total multi-tool examples")
        return all_examples
    
    def review_examples(self, examples: List[Dict[str, str]], tool_name: str,
                      tool_description: str = "", batch_size: int = 50) -> List[Dict[str, str]]:
        """
        Review examples for quality using LLM.
        
        Args:
            examples: List of examples to review
            tool_name: Name of the tool
            tool_description: Description of the tool (optional)
            batch_size: Size of batches for review
            
        Returns:
            Filtered examples
        """
        import concurrent.futures
        
        if not examples:
            return []
            
        self.logger.info(f"Reviewing {len(examples)} examples for quality")
        
        # Batch examples for review
        batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
        
        # Define function to process each batch
        def process_batch(batch_data):
            batch_index, batch = batch_data
            self.logger.info(f"Reviewing batch {batch_index+1}/{len(batches)} ({len(batch)} examples)")
            
            # Create review prompt
            prompt = self.prompt_builder.create_review_prompt(batch, tool_name, tool_description)
            
            try:
                # Generate review
                response_text = self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.1,  # Low temperature for consistent reviews
                    system_prompt="You are an expert at evaluating the quality and relevance of training data for tool classification systems."
                )
                
                # Parse review results
                removed_indices = self.response_parser.parse_review_results(response_text)
                
                # Filter batch
                retained = []
                for j, example in enumerate(batch):
                    if j+1 not in removed_indices:  # +1 because indices are 1-based
                        retained.append(example)
                
                self.logger.info(f"Batch {batch_index+1} review: kept {len(retained)}/{len(batch)}")
                return retained
                
            except Exception as e:
                self.logger.error(f"Error reviewing batch {batch_index+1}: {e}")
                # If review fails, keep all examples from this batch
                return batch
        
        # Process batches in parallel
        all_retained = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(batches))) as executor:
            future_to_batch = {executor.submit(process_batch, (i, batch)): i for i, batch in enumerate(batches)}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    retained = future.result()
                    all_retained.extend(retained)
                except Exception as e:
                    self.logger.error(f"Batch {batch_index+1} processing failed: {e}")
                    # If the entire task fails, keep all examples from this batch
                    all_retained.extend(batches[batch_index])
        
        self.logger.info(f"Review complete. Kept {len(all_retained)}/{len(examples)} examples")
        return all_retained


class SyntheticDataGenerator:
    """
    Main class for generating synthetic training data for tools.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the synthetic data generator with optional configuration overrides.
        
        Args:
            **kwargs: Configuration overrides (api_key, model, embedding_model, etc.)
        """
        # Set up logging
        self._setup_logging()
        
        # Initialize configuration
        self.config = Configuration(**kwargs)
        self.logger.info(f"Initialized with analysis model: {self.config.analysis_model}")
        self.logger.info(f"Initialized with generation model: {self.config.generation_model}")
        
        # Initialize LLM clients for different purposes
        self.analysis_client = LLMClient(
            api_key=self.config.api_key,
            model=self.config.analysis_model,
            logger=self.logger
        )
        
        self.generation_client = LLMClient(
            api_key=self.config.api_key,
            model=self.config.generation_model,
            logger=self.logger
        )
        
        # Initialize embedding model or use provided one
        self.embedding_model = kwargs.get('embedding_model')
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
        else:
            self.logger.info(f"Using provided embedding model")
        
        # Initialize components with specialized clients
        self.tool_analyzer = ToolAnalyzer(self.analysis_client, logger=self.logger)
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser(logger=self.logger)
        self.example_processor = ExampleProcessor(
            embedding_model=self.embedding_model,
            similarity_threshold=self.config.similarity_threshold,
            logger=self.logger
        )
        self.output_manager = OutputManager(self.config.data_dir, logger=self.logger)
        self.example_generator = ExampleGenerator(
            llm_client=self.generation_client,  # Using the faster model for example generation
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser,
            logger=self.logger
        )
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger("SyntheticDataGenerator")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicate messages
        if self.logger.handlers:
            self.logger.handlers = []
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
    
    def generate_for_tool(self, tool_path: str, save_output: bool = True,
                         examples_per_temp: int = None, 
                         temperatures: List[float] = None,
                         skip_llm_review: bool = None) -> List[Dict[str, str]]:
        """
        Generate examples for a tool.
        
        Args:
            tool_path: Path to the tool file
            save_output: Whether to save output to file
            examples_per_temp: Number of examples per temperature (overrides config)
            temperatures: List of temperatures (overrides config)
            skip_llm_review: Whether to skip LLM review (overrides config)
            
        Returns:
            List of generated examples
        """
        # Use configuration with optional overrides
        examples_per_temp = examples_per_temp or self.config.examples_per_temp
        temperatures = temperatures or self.config.temperatures
        skip_llm_review = skip_llm_review if skip_llm_review is not None else self.config.skip_llm_review
        
        self.logger.info(f"Generating examples for tool: {tool_path}")
        self.logger.info(f"Initial configuration: {examples_per_temp} examples per temperature, temperatures: {temperatures}")
        
        # Step 1: Analyze tool with extended thinking to determine complexity and prioritize operations
        self.logger.info("Step 1: Analyzing tool with extended thinking...")
        tool_analysis = self.tool_analyzer.analyze_tool(tool_path)
        tool_name = tool_analysis["tool_name"]
        
        # Adjust examples count based on complexity analysis
        if "recommended_examples" in tool_analysis:
            recommended_count = tool_analysis["recommended_examples"]
            # Distribute the recommended count across temperatures
            dynamic_examples_per_temp = max(10, recommended_count // len(temperatures))
            
            # If we have a recommended count that's different from the default
            if dynamic_examples_per_temp != examples_per_temp:
                self.logger.info(f"Adjusting to complexity-based count: {dynamic_examples_per_temp} examples per temperature")
                self.logger.info(f"Total target examples: {dynamic_examples_per_temp * len(temperatures)}")
                examples_per_temp = dynamic_examples_per_temp
                
                # Log complexity category
                self.logger.info(f"Tool complexity category: {tool_analysis.get('complexity_category', 'standard')}")
        
        # Save tool analysis if requested
        if save_output:
            self.output_manager.save_tool_analysis(tool_name, tool_analysis)
        
        # Step 2: Generate examples with operation prioritization
        self.logger.info("Step 2: Generating examples with priority-weighted distribution...")
        examples = self.example_generator.generate_for_tool(
            tool_analysis=tool_analysis,
            examples_per_temp=examples_per_temp,
            temperatures=temperatures
        )
        
        # Step 3: Deduplicate examples
        self.logger.info("Step 3: Deduplicating examples...")
        deduped_examples = self.example_processor.deduplicate(examples)
        
        # Step 4: Review examples (unless skipped)
        final_examples = deduped_examples
        if not skip_llm_review:
            self.logger.info("Step 4: Reviewing examples for quality...")
            # Get tool description for better context in review
            tool_description = tool_analysis.get("description", "")
            final_examples = self.example_generator.review_examples(
                examples=deduped_examples,
                tool_name=tool_name,
                tool_description=tool_description
            )
        else:
            self.logger.info("Skipping LLM review as requested.")
        
        # Save results if requested
        if save_output:
            self.output_manager.save_examples(tool_name, final_examples)
        
        # Log summary
        self.logger.info(f"Generation complete for {tool_name}")
        self.logger.info(f"Initial examples: {len(examples)}")
        self.logger.info(f"After deduplication: {len(deduped_examples)}")
        self.logger.info(f"Final examples: {len(final_examples)}")
        
        return final_examples
    
    def generate_multi_tool(self, tool_combinations: List[Tuple[str, ...]],
                          examples_per_combo: int = 20,
                          save_output: bool = True) -> List[Dict]:
        """
        Generate multi-tool examples.
        
        Args:
            tool_combinations: List of tool name combinations
            examples_per_combo: Number of examples per combination
            save_output: Whether to save output to file
            
        Returns:
            List of multi-tool examples
        """
        self.logger.info(f"Generating multi-tool examples for {len(tool_combinations)} combinations")
        
        # Generate examples
        multi_tool_examples = self.example_generator.generate_multi_tool(
            tool_combinations=tool_combinations,
            examples_per_combo=examples_per_combo
        )
        
        # Deduplicate
        self.logger.info("Deduplicating multi-tool examples...")
        deduped_examples = self.example_processor.deduplicate(
            multi_tool_examples,
            query_field="query"
        )
        
        # Save results if requested
        if save_output:
            self.output_manager.save_multi_tool_examples(deduped_examples)
        
        # Log summary
        self.logger.info(f"Multi-tool generation complete")
        self.logger.info(f"Initial examples: {len(multi_tool_examples)}")
        self.logger.info(f"After deduplication: {len(deduped_examples)}")
        
        return deduped_examples
    
    def generate_for_directory(self, tool_dir: str, save_output: bool = True,
                             examples_per_temp: int = None,
                             temperatures: List[float] = None,
                             skip_llm_review: bool = None,
                             generate_multi_tool: bool = False,
                             parallel_tools: int = 3) -> Dict[str, List[Dict]]:
        """
        Generate examples for all tools in a directory.
        
        Args:
            tool_dir: Directory containing tool files
            save_output: Whether to save output
            examples_per_temp: Examples per temperature (overrides config)
            temperatures: Temperatures (overrides config)
            skip_llm_review: Whether to skip review (overrides config)
            generate_multi_tool: Whether to generate multi-tool examples
            parallel_tools: Number of tools to process in parallel
            
        Returns:
            Dictionary mapping tool names to examples
        """
        import os
        import concurrent.futures
        
        # Get all tool files
        self.logger.info(f"Scanning directory: {tool_dir}")
        tool_files = []
        tool_names = []
        
        for file in os.listdir(tool_dir):
            if file.endswith("_tool.py"):
                tool_files.append(os.path.join(tool_dir, file))
                tool_names.append(file[:-3])  # Remove .py
        
        self.logger.info(f"Found {len(tool_files)} tool files")
        
        # Process the tools in parallel batches
        results = {}
        
        # Define a function to analyze and prepare a tool (CPU-intensive part)
        def analyze_tool(tool_file):
            tool_name = self.tool_analyzer.extract_tool_name(tool_file)
            self.logger.info(f"Analyzing tool: {tool_name}")
            
            # Perform the tool analysis using extended thinking
            analysis = self.tool_analyzer.analyze_tool(tool_file)
            
            # Prepare the analysis result
            return {
                "tool_name": tool_name,
                "tool_file": tool_file,
                "analysis": analysis
            }
            
        # Define function to generate examples for a pre-analyzed tool
        def generate_examples_for_analyzed_tool(tool_info):
            tool_name = tool_info["tool_name"]
            tool_file = tool_info["tool_file"]
            analysis = tool_info["analysis"]
            
            self.logger.info(f"Generating examples for pre-analyzed tool: {tool_name}")
            
            # Adjust examples count based on complexity analysis
            local_examples_per_temp = examples_per_temp
            if "recommended_examples" in analysis:
                recommended_count = analysis["recommended_examples"]
                # Distribute the recommended count across temperatures
                temp_count = temperatures or self.config.temperatures
                dynamic_examples_per_temp = max(10, recommended_count // len(temp_count))
                
                if dynamic_examples_per_temp != local_examples_per_temp:
                    self.logger.info(f"Tool {tool_name}: Adjusting to {dynamic_examples_per_temp} examples per temperature")
                    local_examples_per_temp = dynamic_examples_per_temp
            
            # Skip tool analysis (already done) and generate examples
            try:
                # Generate examples (using pre-analyzed tool)
                examples = self.example_generator.generate_for_tool(
                    tool_analysis=analysis,
                    examples_per_temp=local_examples_per_temp,
                    temperatures=temperatures or self.config.temperatures
                )
                
                # Deduplicate examples
                deduped_examples = self.example_processor.deduplicate(examples)
                
                # Review examples if needed
                if not skip_llm_review and not self.config.skip_llm_review:
                    tool_description = analysis.get("description", "")
                    final_examples = self.example_generator.review_examples(
                        examples=deduped_examples,
                        tool_name=tool_name,
                        tool_description=tool_description
                    )
                else:
                    final_examples = deduped_examples
                
                # Save results if requested
                if save_output:
                    self.output_manager.save_tool_analysis(tool_name, analysis)
                    self.output_manager.save_examples(tool_name, final_examples)
                
                # Log summary
                self.logger.info(f"Completed {tool_name}: {len(final_examples)} examples")
                
                return tool_name, final_examples
                
            except Exception as e:
                self.logger.error(f"Error generating examples for {tool_name}: {e}")
                return tool_name, []
                
        # Step 1: Analyze all tools first (more CPU-intensive, less API-intensive)
        self.logger.info("Step 1: Analyzing all tools...")
        analyzed_tools = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(tool_files))) as executor:
            future_to_tool = {executor.submit(analyze_tool, tool_file): tool_file for tool_file in tool_files}
            
            for future in concurrent.futures.as_completed(future_to_tool):
                tool_file = future_to_tool[future]
                try:
                    tool_info = future.result()
                    analyzed_tools.append(tool_info)
                    self.logger.info(f"Completed analysis for {tool_info['tool_name']}")
                except Exception as e:
                    self.logger.error(f"Error analyzing tool {tool_file}: {e}")
        
        # Step 2: Generate examples with controlled parallelism (more API-intensive)
        self.logger.info(f"Step 2: Generating examples (processing {parallel_tools} tools in parallel)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_tools) as executor:
            future_to_tool = {executor.submit(generate_examples_for_analyzed_tool, tool_info): 
                             tool_info["tool_name"] for tool_info in analyzed_tools}
            
            for future in concurrent.futures.as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    _, examples = future.result()
                    results[tool_name] = examples
                except Exception as e:
                    self.logger.error(f"Error in final processing of {tool_name}: {e}")
        
        # Generate multi-tool examples if requested
        if generate_multi_tool and len(tool_names) >= 2:
            from itertools import combinations
            
            self.logger.info("Generating multi-tool examples...")
            # Limit to a reasonable number of combinations
            max_combinations = 10
            tool_combinations = list(combinations(tool_names, 2))[:max_combinations]
            
            multi_tool_examples = self.generate_multi_tool(
                tool_combinations=tool_combinations,
                examples_per_combo=examples_per_temp // 2 if examples_per_temp else 15,
                save_output=save_output
            )
            
            results["multi_tool"] = multi_tool_examples
        
        return results


def main():
    """Command-line interface for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic training data for tool classifiers")
    parser.add_argument("--tool_dir", type=str, help="Directory containing tool Python files")
    parser.add_argument("--tool_file", type=str, help="Single tool Python file")
    parser.add_argument("--output", type=str, help="Output directory (defaults to data/tools)")
    parser.add_argument("--examples_per_temp", type=int, default=20, help="Examples to generate per temperature (will be adjusted based on tool complexity)")
    parser.add_argument("--api_key", type=str, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--analysis_model", type=str, help="Model for code analysis (defaults to claude-3-7-sonnet)")
    parser.add_argument("--generation_model", type=str, help="Model for example generation (defaults to claude-3-5-haiku)")
    parser.add_argument("--embedding_model", type=str, help="Embedding model to use (defaults to config or MiniLM)")
    parser.add_argument("--multi_tool", action="store_true", help="Generate multi-tool examples")
    parser.add_argument("--skip_llm_review", action="store_true", help="Skip the LLM quality review step")
    parser.add_argument("--temperatures", type=str, default="0.2,0.8", help="Comma-separated list of temperatures")
    parser.add_argument("--similarity_threshold", type=float, default=0.85, help="Threshold for deduplication (lower = more aggressive)")
    parser.add_argument("--parallel_tools", type=int, default=3, help="Number of tools to process in parallel (default: 3)")
    
    args = parser.parse_args()
    
    # Parse temperatures
    temperatures = [float(t) for t in args.temperatures.split(",")]
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        api_key=args.api_key,
        analysis_model=args.analysis_model,
        generation_model=args.generation_model,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        examples_per_temp=args.examples_per_temp,
        temperatures=temperatures,
        skip_llm_review=args.skip_llm_review,
        data_dir=args.output
    )
    
    # Handle tool directory
    if args.tool_dir:
        generator.generate_for_directory(
            tool_dir=args.tool_dir,
            generate_multi_tool=args.multi_tool,
            parallel_tools=args.parallel_tools
        )
    # Handle single tool file
    elif args.tool_file:
        generator.generate_for_tool(
            tool_path=args.tool_file
        )
    else:
        print("Error: Must specify either --tool_dir or --tool_file")
        parser.print_help()


def generate_examples_for_tool(
    tool_path: str, 
    output_path: str = None,
    examples_per_temp: int = 30,
    similarity_threshold: float = 0.92,
    skip_llm_review: bool = False,
    api_key: str = None,
    model: str = None,
    embedding_model: str = None,
    use_autogen_suffix: bool = True  # Generate to autogen_classifier_examples.json by default
) -> List[Dict[str, str]]:
    """
    Simplified function to generate examples for a single tool programmatically.
    
    Args:
        tool_path: Path to the tool Python file
        output_path: Optional path to save JSON output (if None, uses data/tools/{tool_name}/classifier_examples.json)
        examples_per_temp: Number of examples to generate per temperature
        similarity_threshold: Threshold for considering examples too similar
        skip_llm_review: Whether to skip LLM quality review
        api_key: Anthropic API key (defaults to environment variable)
        model: Model to use (defaults to config)
        embedding_model: Embedding model to use (defaults to config)
        use_autogen_suffix: Whether to use "autogen_" prefix in output filename
        
    Returns:
        List of examples in the format {"tool_name": X, "query": Y}
    """
    # Initialize generator
    generator = SyntheticDataGenerator(
        api_key=api_key,
        analysis_model=model,
        generation_model=model,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        examples_per_temp=examples_per_temp,
        skip_llm_review=skip_llm_review
    )
    
    # Extract tool name
    tool_name = Path(tool_path).stem
    
    print(f"Generating examples for {tool_name}...")
    
    # Use default temperatures
    temperatures = [0.3, 0.7, 1.0]
    
    # Generate examples with extended thinking analysis
    final_examples = generator.generate_for_tool(
        tool_path=tool_path,
        examples_per_temp=examples_per_temp,
        temperatures=temperatures,
        skip_llm_review=skip_llm_review,
        save_output=False  # We'll handle saving ourselves
    )
    
    # Determine output path if not explicitly provided
    if output_path is None and HAS_PROJECT_CONFIG:
        # Use project data directory structure: data/tools/{tool_name}/classifier_examples.json
        data_dir = config.paths.data_dir
        tool_data_dir = os.path.join(data_dir, "tools", tool_name)
        os.makedirs(tool_data_dir, exist_ok=True)
        filename = "autogen_classifier_examples.json" if use_autogen_suffix else "classifier_examples.json"
        output_path = os.path.join(tool_data_dir, filename)
    
    # Save to file if path is available
    if output_path:
        # Make sure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save examples
        with open(output_path, 'w') as f:
            json.dump(final_examples, f, indent=2)
        print(f"Saved {len(final_examples)} examples to {output_path}")
    
    return final_examples

if __name__ == "__main__":
    main()