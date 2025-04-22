#!/usr/bin/env python3
"""
Test script for running the synthetic data generator on a specific tool.
Usage: python test_generator.py <tool_path>
Example: python test_generator.py ../../tools/sample_tool.py
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Add parent directories to path to ensure imports work
current_dir = Path(__file__).parent.absolute()
project_dir = current_dir.parent.parent
sys.path.insert(0, str(project_dir))

# Import the generator
from synthetic_data_generator import SyntheticDataGenerator

def setup_logging():
    """Set up logging configuration."""
    # Clear existing handlers from root logger to avoid duplication
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('generator_test.log')
        ]
    )
    
    # Create a specific logger for the test script
    logger = logging.getLogger("test_generator")
    
    # Disable propagation to avoid duplicate logs
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).propagate = False
    
    # But allow our test logger to propagate
    logger.propagate = True
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="Test the synthetic data generator on a specific tool")
    parser.add_argument("tool_path", help="Path to the tool Python file to analyze")
    parser.add_argument("--examples_per_temp", type=int, default=20, 
                      help="Base number of examples per temperature (default: 20)")
    parser.add_argument("--skip_llm_review", action="store_true", 
                      help="Skip LLM quality review (faster but lower quality)")
    parser.add_argument("--output_dir", type=str, default="./test_output",
                      help="Directory to save output (default: ./test_output)")
    parser.add_argument("--analysis_model", type=str, default="claude-3-7-sonnet-20250219",
                      help="Model for code analysis (default: claude-3-7-sonnet-20250219)")
    parser.add_argument("--generation_model", type=str, default="claude-3-haiku-20240307",
                      help="Model for example generation (default: claude-3-haiku-20240307)")
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"Testing generator on tool: {args.tool_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get absolute path to tool
    tool_path = str(Path(args.tool_path).absolute())
    
    try:
        # Initialize generator
        logger.info("Initializing generator...")
        generator = SyntheticDataGenerator(
            analysis_model=args.analysis_model,
            generation_model=args.generation_model,
            examples_per_temp=args.examples_per_temp,
            temperatures=[0.2, 0.8],  # Using optimized temperature settings
            similarity_threshold=0.85,  # More aggressive deduplication
            skip_llm_review=args.skip_llm_review,
            data_dir=args.output_dir
        )
        
        # Generate examples
        logger.info("Generating examples...")
        start_time = time.time()
        examples = generator.generate_for_tool(
            tool_path=tool_path,
            save_output=True
        )
        end_time = time.time()
        
        # Log results
        duration = end_time - start_time
        logger.info(f"Generation completed in {duration:.2f} seconds")
        logger.info(f"Generated {len(examples)} examples")
        
        # Save examples to JSON file
        tool_name = Path(tool_path).stem
        output_path = os.path.join(args.output_dir, f"{tool_name}_examples.json")
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print("\n" + "="*50)
        print(f"GENERATION SUMMARY FOR: {tool_name}")
        print("="*50)
        print(f"Total examples: {len(examples)}")
        print(f"Generation time: {duration:.2f} seconds")
        print(f"Output saved to: {output_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())