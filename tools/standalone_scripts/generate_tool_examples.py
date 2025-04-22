#!/usr/bin/env python3
"""
Utility script to generate synthetic examples for a specific tool.

This script loads the synthetic_data_generator and creates examples for a tool,
saving them to either classifier_examples.json or autogen_classifier_examples.json.
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.standalone_scripts.synthetic_data_generator import generate_examples_for_tool
from config.config_manager import config


def main():
    """Command line interface for generating tool examples."""
    parser = argparse.ArgumentParser(description="Generate synthetic training data for a specific tool")
    
    parser.add_argument("tool_name", type=str, help="Name of the tool (e.g., 'calendar_tool' without .py)")
    parser.add_argument("--autogen", action="store_true", 
                      help="Save as autogen_classifier_examples.json instead of classifier_examples.json")
    parser.add_argument("--examples", type=int, default=30, 
                      help="Number of examples to generate per temperature")
    parser.add_argument("--skip-review", action="store_true", 
                      help="Skip the LLM quality review step")
    parser.add_argument("--api-key", type=str, default=None,
                      help="Anthropic API key (or use config/env)")
    
    args = parser.parse_args()
    
    # Determine the tool file path
    tools_dir = getattr(config.paths, 'tools_dir', 'tools')
    tool_file_path = os.path.join(tools_dir, f"{args.tool_name}.py")
    
    # Check if the tool file exists
    if not os.path.exists(tool_file_path):
        print(f"Error: Tool file {tool_file_path} not found")
        return 1
    
    # Determine output path
    data_dir = config.paths.data_dir
    tool_data_dir = os.path.join(data_dir, "tools", args.tool_name)
    os.makedirs(tool_data_dir, exist_ok=True)
    
    if args.autogen:
        output_path = os.path.join(tool_data_dir, "autogen_classifier_examples.json")
    else:
        output_path = os.path.join(tool_data_dir, "classifier_examples.json")
    
    print(f"Generating examples for {args.tool_name}")
    print(f"Tool path: {tool_file_path}")
    print(f"Output path: {output_path}")
    print(f"Generating {args.examples} examples per temperature")
    print(f"Skip review: {args.skip_review}")
    
    # Generate examples
    try:
        examples = generate_examples_for_tool(
            tool_path=tool_file_path,
            output_path=output_path,
            examples_per_temp=args.examples,
            skip_llm_review=args.skip_review,
            api_key=args.api_key,
            model=None  # Use default model from config
        )
        
        print(f"Successfully generated {len(examples)} examples for {args.tool_name}")
        print(f"Examples saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error generating examples: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())