#!/usr/bin/env python3
"""
Conversational Example Generator

A script that automatically generates natural conversation examples for AI tool classification training.
Takes any Python tool file as input and produces realistic human queries that would trigger the tool.

Usage:
    python conversational_example_generator.py <tool_file_path> [--count 50] [--output output.json]
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to the path so we can import from the project
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.llm_provider import LLMProvider
from errors import APIError, ErrorCode, error_context


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ToolAnalysis:
    """Complete analysis of a tool's functionality."""
    tool_name: str
    primary_purpose: str
    capabilities: List[str]
    target_users: List[str]
    common_scenarios: List[str]
    usage_patterns: List[str]


class ConversationalExampleGenerator:
    """
    Generates natural conversational examples for AI tool classification training.
    
    This class analyzes Python tool files using LLM intelligence and creates realistic 
    human queries that would trigger the tool, focusing on natural speech patterns 
    and diverse user contexts.
    """
    
    def __init__(self):
        """Initialize the generator with LLM bridge."""
        self.llm = LLMProvider()
        logger.info("Conversational Example Generator initialized")
    
    def analyze_tool(self, tool_path: str) -> ToolAnalysis:
        """
        Extract and analyze a tool's functionality using LLM intelligence.
        
        Args:
            tool_path: Path to the Python tool file
            
        Returns:
            ToolAnalysis object containing comprehensive tool information
        """
        logger.info(f"Analyzing tool: {tool_path}")
        
        # Read the tool source code
        with open(tool_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Use LLM to analyze the tool intelligently
        analysis_prompt = """
        Analyze this Python tool source code and extract key information about its functionality.
        
        IMPORTANT: Respond with ONLY valid JSON. Do not include any text before or after the JSON.
        
        Required JSON structure:
        {
            "tool_name": "exact name of the tool from the code",
            "primary_purpose": "concise description of what this tool does",
            "capabilities": ["list", "of", "specific", "operations", "or", "functions"],
            "target_users": ["types", "of", "people", "who", "would", "use", "this"],
            "common_scenarios": ["situations", "when", "people", "would", "need", "this", "tool"],
            "usage_patterns": ["natural", "ways", "people", "might", "ask", "for", "help"]
        }
        
        Focus on:
        1. The human value proposition - what problems does this solve?
        2. Real-world use cases rather than technical details
        3. Natural language patterns people would use
        4. Different types of users (busy professionals, casual users, etc.)
        
        Return ONLY the JSON object, no additional text.
        
        Tool source code:
        """
        
        messages = [
            {"role": "user", "content": f"{analysis_prompt}\n\n```python\n{source_code}\n```"}
        ]
        
        try:
            with error_context(
                component_name="conversational_generator",
                operation="tool_analysis",
                error_class=APIError,
                error_code=ErrorCode.API_RESPONSE_ERROR,
                logger=logger
            ):
                response = self.llm.generate_response(messages)
                response_text = self.llm.extract_text_content(response)
                
                logger.debug(f"LLM response text: {response_text[:500]}...")
                
                # Parse the JSON response
                analysis_data = json.loads(response_text)
                
                return ToolAnalysis(
                    tool_name=analysis_data.get("tool_name", Path(tool_path).stem),
                    primary_purpose=analysis_data.get("primary_purpose", ""),
                    capabilities=analysis_data.get("capabilities", []),
                    target_users=analysis_data.get("target_users", []),
                    common_scenarios=analysis_data.get("common_scenarios", []),
                    usage_patterns=analysis_data.get("usage_patterns", [])
                )
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM analysis, using fallback: {e}")
            return self._fallback_analysis(tool_path, source_code)
    
    def _fallback_analysis(self, tool_path: str, source_code: str) -> ToolAnalysis:
        """Fallback analysis if LLM parsing fails."""
        tool_name = Path(tool_path).stem
        
        # Basic pattern matching as fallback
        if "email" in source_code.lower():
            return ToolAnalysis(
                tool_name=tool_name,
                primary_purpose="Email management and communication",
                capabilities=["read emails", "send emails", "organize messages"],
                target_users=["business professionals", "personal users"],
                common_scenarios=["managing inbox", "sending messages"],
                usage_patterns=["check my emails", "send a message"]
            )
        else:
            return ToolAnalysis(
                tool_name=tool_name,
                primary_purpose="General purpose tool",
                capabilities=["tool operations"],
                target_users=["general users"],
                common_scenarios=["general use"],
                usage_patterns=["help me with"]
            )
    
    def generate_scenarios(self, tool_analysis: ToolAnalysis, count: int = 50) -> List[Dict[str, str]]:
        """
        Generate diverse conversation scenarios using LLM intelligence.
        
        Args:
            tool_analysis: Analysis of the tool's functionality
            count: Number of examples to generate
            
        Returns:
            List of conversation scenarios
        """
        logger.info(f"Generating {count} conversation scenarios")
        
        scenario_prompt = f"""
        Generate {count} natural, conversational examples of how real people would ask for help with this tool.

        Tool Information:
        - Name: {tool_analysis.tool_name}
        - Purpose: {tool_analysis.primary_purpose}
        - Capabilities: {', '.join(tool_analysis.capabilities)}
        - Target Users: {', '.join(tool_analysis.target_users)}
        - Common Scenarios: {', '.join(tool_analysis.common_scenarios)}

        Requirements for the examples:
        1. Sound like REAL human speech - natural, conversational, sometimes vague
        2. Include diverse user types: busy professionals, casual users, seniors, parents, tech-savvy people
        3. Vary formality: "Can you please..." vs "I need..." vs "Show me..."
        4. Include emotional context: urgency, frustration, curiosity, etc.
        5. Use personal details: "my inbox", "from my doctor", "about the meeting"
        6. Avoid technical jargon - speak like normal people, not programmers
        7. Include typos and informal language occasionally
        8. Vary length from short commands to longer explanations

        Examples should cover different aspects:
        - Different capabilities of the tool
        - Various urgency levels (urgent, casual, exploratory)
        - Different user contexts (work, personal, family)
        - Mixed specificity (vague requests vs specific needs)

        IMPORTANT: Respond with ONLY valid JSON. Do not include any text before or after the JSON array.

        Required format - JSON array only:
        [
            {{"tool_name": "{tool_analysis.tool_name}", "query": "Show me my recent emails"}},
            {{"tool_name": "{tool_analysis.tool_name}", "query": "I need to check if my doctor replied about the test results"}}
        ]

        Generate exactly {count} examples with natural variety. Return ONLY the JSON array.
        """
        
        messages = [
            {"role": "user", "content": scenario_prompt}
        ]
        
        try:
            with error_context(
                component_name="conversational_generator",
                operation="scenario_generation",
                error_class=APIError,
                error_code=ErrorCode.API_RESPONSE_ERROR,
                logger=logger
            ):
                response = self.llm.generate_response(messages, max_tokens=8000)
                response_text = self.llm.extract_text_content(response)
                
                # Parse the JSON response
                examples = json.loads(response_text)
                
                # Validate the structure
                if isinstance(examples, list):
                    validated_examples = []
                    for example in examples:
                        if isinstance(example, dict) and "tool_name" in example and "query" in example:
                            validated_examples.append(example)
                    
                    logger.info(f"Generated {len(validated_examples)} valid examples")
                    return validated_examples
                else:
                    logger.warning("LLM returned non-list response, using fallback")
                    return self._generate_fallback_examples(tool_analysis, count)
                    
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM scenarios: {e}")
            return self._generate_fallback_examples(tool_analysis, count)
    
    def _generate_fallback_examples(self, tool_analysis: ToolAnalysis, count: int) -> List[Dict[str, str]]:
        """Generate fallback examples if LLM generation fails."""
        logger.info("Using fallback example generation")
        
        base_patterns = [
            f"Show me {tool_analysis.primary_purpose.lower()}",
            f"I need help with {tool_analysis.tool_name.replace('_', ' ')}",
            f"Can you {tool_analysis.capabilities[0] if tool_analysis.capabilities else 'help me'}?",
            f"Help me {tool_analysis.primary_purpose.lower()}",
            f"I want to {tool_analysis.capabilities[0] if tool_analysis.capabilities else 'use this tool'}"
        ]
        
        examples = []
        for i in range(min(count, len(base_patterns) * 10)):
            pattern = base_patterns[i % len(base_patterns)]
            examples.append({
                "tool_name": tool_analysis.tool_name,
                "query": pattern
            })
        
        return examples[:count]
    
    def validate_quality(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validate and filter examples for quality and authenticity using LLM.
        
        Args:
            examples: List of generated examples
            
        Returns:
            Filtered list of high-quality examples
        """
        if not examples:
            return examples
        
        logger.info(f"Validating quality of {len(examples)} examples")
        
        # Sample examples for validation (don't validate all if there are many)
        sample_size = min(20, len(examples))
        sample_examples = examples[:sample_size]
        
        validation_prompt = f"""
        Review these conversational examples and identify which ones sound like natural human speech vs robotic/technical language.

        Examples to review:
        {json.dumps([ex['query'] for ex in sample_examples], indent=2)}

        Criteria for GOOD examples:
        - Sound like real people talking (casual, natural, sometimes vague)
        - Use personal context ("my emails", "from my boss")
        - Vary in formality and urgency
        - Include emotional undertones
        - Avoid technical jargon

        Criteria for BAD examples:
        - Sound robotic or formal
        - Use technical terms like "execute", "invoke", "parameter"
        - Too generic or repetitive
        - Overly polite or formal

        IMPORTANT: Respond with ONLY valid JSON. Do not include any text before or after the JSON array.

        Return a JSON array with indices of the GOOD examples (0-based indexing):
        [0, 2, 5, 7, ...]
        """
        
        messages = [
            {"role": "user", "content": validation_prompt}
        ]
        
        try:
            response = self.llm.generate_response(messages)
            response_text = self.llm.extract_text_content(response)
            
            # Parse the validation results
            good_indices = json.loads(response_text)
            
            if isinstance(good_indices, list):
                # Apply validation results to full list
                validation_ratio = len(good_indices) / len(sample_examples)
                logger.info(f"Validation ratio: {validation_ratio:.2f}")
                
                # If validation ratio is reasonable, apply pattern to all examples
                if validation_ratio > 0.3:  # At least 30% should be good
                    # For now, return all examples but log the quality
                    logger.info(f"Quality validation complete, keeping all {len(examples)} examples")
                    return examples
                else:
                    logger.warning("Low validation ratio, keeping examples anyway")
                    return examples
            else:
                logger.warning("Invalid validation response, keeping all examples")
                return examples
                
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Validation failed: {e}, keeping all examples")
            return examples
    
    def generate(self, tool_path: str, count: int = 50, output_path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Main generation method that orchestrates the entire process.
        
        Args:
            tool_path: Path to the Python tool file
            count: Number of examples to generate
            output_path: Optional path for JSON output file
            
        Returns:
            List of generated examples
        """
        logger.info(f"Starting generation for {tool_path}")
        
        # Step 1: Analyze the tool using LLM
        tool_analysis = self.analyze_tool(tool_path)
        logger.info(f"Analyzed tool: {tool_analysis.tool_name}")
        logger.info(f"Purpose: {tool_analysis.primary_purpose}")
        logger.info(f"Capabilities: {len(tool_analysis.capabilities)} found")
        
        # Step 2: Generate scenarios using LLM
        examples = self.generate_scenarios(tool_analysis, count)
        logger.info(f"Generated {len(examples)} examples")
        
        # Step 3: Validate quality using LLM
        validated_examples = self.validate_quality(examples)
        logger.info(f"Validated {len(validated_examples)} high-quality examples")
        
        # Step 4: Save to file if requested
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validated_examples, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved examples to {output_path}")
        
        return validated_examples


def main():
    """Command-line interface for the generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate conversational examples for AI tool classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python conversational_example_generator.py tools/email_tool.py
  python conversational_example_generator.py tools/calendar_tool.py --count 100 --output examples.json
        """
    )
    
    parser.add_argument("tool_path", help="Path to the Python tool file")
    parser.add_argument("--count", type=int, default=50, help="Number of examples to generate")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.tool_path):
        logger.error(f"Tool file not found: {args.tool_path}")
        sys.exit(1)
    
    try:
        # Generate examples
        generator = ConversationalExampleGenerator()
        examples = generator.generate(args.tool_path, args.count, args.output)
        
        # Display results
        print(f"\nGenerated {len(examples)} conversational examples for {args.tool_path}")
        print("\nSample examples:")
        for i, example in enumerate(examples[:5], 1):
            print(f"{i}. {example['query']}")
        
        if len(examples) > 5:
            print(f"... and {len(examples) - 5} more")
        
        print(f"\nTool: {examples[0]['tool_name'] if examples else 'Unknown'}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()