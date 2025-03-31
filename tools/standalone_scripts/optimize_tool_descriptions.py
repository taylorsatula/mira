#!/usr/bin/env python3
"""
Tool description optimizer for Claude API interactions.

This script analyzes all tools in the application and generates optimized
descriptions following Anthropic's guidelines for effective tool definitions.
Run this script standalone to improve tool calling performance.
"""

import os
import sys
import json
import inspect
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import application components
from tools.repo import Tool, ToolRepository
from api.llm_bridge import LLMBridge
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("optimize_tools")

# Anthropic guidelines template for tool descriptions
TOOL_OPTIMIZATION_PROMPT = """
Create an optimized description for the tool named "{name}" following these EXACT requirements:

1. Start with an action verb that clearly indicates what the tool does
2. Be concise (under 500 characters) but comprehensive
3. MUST include the tool name "{name}" explicitly in the description
4. Specify what the tool does, when to use it, and any limitations
5. List the main parameters and their purpose
6. Mention what output the tool returns

REFERENCE INFORMATION:
- Original description: {description}
- Parameters: {parameters_json}
- Usage examples: {examples_json}

Provide ONLY the improved description text - no explanations, no formatting, no bullet points.
Example format: "Retrieves weather data for a location. Takes a location parameter (string) and optional units parameter. Returns current temperature, conditions, and forecast."
"""

class ToolDescriptionOptimizer:
    """Tool description optimizer for Claude API interactions."""
    
    def __init__(self, repo: ToolRepository, output_path: Optional[str] = None):
        """Initialize the optimizer with a tool repository."""
        self.repo = repo
        self.llm_bridge = LLMBridge()
        self.output_path = output_path or Path(config.paths.persistent_dir) / "optimized_tool_descriptions.json"
        self.optimized_descriptions = {}
        
    def optimize_all_tools(self, verify: bool = False) -> Dict[str, Dict[str, str]]:
        """Analyze and optimize all tools in the repository."""
        results = {}
        tools_to_optimize = list(self.repo.tools.items())
        
        logger.info(f"Optimizing {len(tools_to_optimize)} tools...")
        
        for name, tool in tools_to_optimize:
            try:
                logger.info(f"Optimizing tool: {name}")
                original_desc = self.get_tool_description(tool)
                optimized_desc = self.generate_optimized_description(tool)
                
                # Simple validation
                if self.validate_description(name, optimized_desc):
                    results[name] = {
                        "original": original_desc,
                        "optimized": optimized_desc
                    }
                    
                    # Store optimized description
                    self.optimized_descriptions[name] = optimized_desc
                    
                    # Show verification if requested
                    if verify and original_desc != optimized_desc:
                        print(f"\n--- Tool: {name} ---")
                        print(f"Original: {original_desc}")
                        print(f"Optimized: {optimized_desc}")
                        print("-" * 50)
                else:
                    logger.warning(f"Validation failed for {name}, keeping original description")
                    results[name] = {
                        "original": original_desc,
                        "optimized": original_desc,
                        "note": "Validation failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error optimizing tool {name}: {e}")
                results[name] = {
                    "original": self.get_tool_description(tool),
                    "error": str(e)
                }
        
        # Save the optimized descriptions
        self.save_optimized_descriptions()
        
        # Summary statistics
        improved_count = sum(1 for name, data in results.items() 
                          if "optimized" in data and data["original"] != data["optimized"])
        logger.info(f"Optimization complete. Improved {improved_count} of {len(results)} tools.")
        
        return results
    
    def get_tool_description(self, tool: Any) -> str:
        """Get the current description of a tool."""
        if hasattr(tool, 'description'):
            return tool.description
        return ""
    
    def get_tool_parameters(self, tool: Any) -> Dict[str, Any]:
        """Get parameter definitions for a tool."""
        if hasattr(tool, 'get_parameter_schema'):
            try:
                if inspect.isclass(tool):
                    return tool.get_parameter_schema()
                else:
                    return tool.get_parameter_schema()
            except Exception:
                pass
        return {}
    
    def get_tool_examples(self, tool: Any) -> List[Dict[str, Any]]:
        """Get usage examples for a tool."""
        return getattr(tool, 'usage_examples', [])
    
    def generate_optimized_description(self, tool: Any) -> str:
        """Generate an optimized description for a tool using LLM."""
        # Get tool information
        name = getattr(tool, 'name', "unknown")
        original_description = self.get_tool_description(tool)
        parameters = self.get_tool_parameters(tool)
        examples = self.get_tool_examples(tool)
        
        # Format the prompt with tool information
        prompt = TOOL_OPTIMIZATION_PROMPT.format(
            name=name,
            description=original_description,
            parameters_json=json.dumps(parameters, indent=2),
            examples_json=json.dumps(examples, indent=2)
        )
        
        # Use the LLM to generate optimized description
        try:
            messages = [{"role": "user", "content": prompt}]
            system_prompt = "You are a tool description optimizer for AI systems. Your job is to write clear, concise tool descriptions that help Claude use tools correctly. Follow the format requirements exactly. Always include the tool name in the description. Start with a verb. Be specific about parameters and return values."
            
            response = self.llm_bridge.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.2,  # Low temperature for consistent output
                max_tokens=300,
            )
            
            # Extract the optimized description
            optimized_description = self.llm_bridge.extract_text_content(response).strip()
            
            # If optimization fails, keep the original
            if not optimized_description:
                logger.warning(f"Failed to generate optimized description for {name}, keeping original")
                return original_description
                
            return optimized_description
            
        except Exception as e:
            logger.error(f"Error generating description for {name}: {e}")
            return original_description
    
    def validate_description(self, tool_name: str, description: str) -> bool:
        """Perform basic validation on the optimized description."""
        # Check for minimum length
        if len(description) < 10:
            logger.warning(f"Description too short for {tool_name}")
            return False
        
        # Check for maximum length (avoid excessive descriptions)
        if len(description) > 500:
            logger.warning(f"Description too long for {tool_name}")
            return False
            
        # Check that the description contains the tool name
        if tool_name.lower() not in description.lower():
            logger.warning(f"Description doesn't mention tool name: {tool_name}")
            return False
            
        return True
    
    def save_optimized_descriptions(self) -> None:
        """Save all optimized descriptions to a file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            with open(self.output_path, 'w') as f:
                json.dump(self.optimized_descriptions, f, indent=2)
                
            logger.info(f"Saved optimized descriptions to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving optimized descriptions: {e}")
    
    @classmethod
    def load_optimized_descriptions(cls, file_path: Optional[str] = None) -> Dict[str, str]:
        """Load previously optimized descriptions from a file."""
        path = file_path or Path(config.paths.persistent_dir) / "optimized_tool_descriptions.json"
        
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading optimized descriptions: {e}")
            
        return {}


def apply_optimized_descriptions(repo: ToolRepository, descriptions: Dict[str, str]) -> int:
    """Apply optimized descriptions to tool instances."""
    updated_count = 0
    
    for name, description in descriptions.items():
        tool = repo.get_tool(name)
        if tool:
            try:
                # Update the description at the class level
                if repo.tool_instances.get(name, False):
                    # It's an instance, get its class
                    tool_class = tool.__class__
                else:
                    # It's already a class
                    tool_class = tool
                    
                # Update the description
                tool_class.description = description
                updated_count += 1
                logger.info(f"Updated description for {name}")
                
            except Exception as e:
                logger.error(f"Error updating description for {name}: {e}")
    
    return updated_count


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Optimize tool descriptions for Claude API')
    parser.add_argument('--output', '-o', type=str, help='Path to save optimized descriptions')
    parser.add_argument('--apply', '-a', action='store_true', help='Apply optimized descriptions to tools')
    parser.add_argument('--verify', '-v', action='store_true', help='Show before/after comparisons')
    parser.add_argument('--load-only', '-l', action='store_true', help='Only load and apply existing optimized descriptions')
    args = parser.parse_args()
    
    # Initialize the tool repository
    repo = ToolRepository()
    
    if args.load_only:
        # Only load and apply existing optimized descriptions
        descriptions = ToolDescriptionOptimizer.load_optimized_descriptions(args.output)
        if descriptions:
            updated = apply_optimized_descriptions(repo, descriptions)
            logger.info(f"Loaded and applied {updated} optimized descriptions")
        else:
            logger.warning("No optimized descriptions found to load")
        return
    
    # Initialize the optimizer
    optimizer = ToolDescriptionOptimizer(repo, args.output)
    
    # Analyze and optimize tool descriptions
    results = optimizer.optimize_all_tools(verify=args.verify)
    
    # Apply optimized descriptions if requested
    if args.apply:
        updated = apply_optimized_descriptions(repo, optimizer.optimized_descriptions)
        logger.info(f"Applied {updated} optimized descriptions to tools")
    
    # Print summary
    improved_count = sum(1 for name, data in results.items() 
                        if "optimized" in data and data["original"] != data["optimized"])
    error_count = sum(1 for data in results.values() if "error" in data)
    
    print(f"\nSummary:")
    print(f"- Analyzed {len(results)} tools")
    print(f"- Improved {improved_count} descriptions")
    print(f"- Encountered {error_count} errors")
    print(f"- Saved to {optimizer.output_path}")


if __name__ == "__main__":
    main()