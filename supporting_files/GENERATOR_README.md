# Synthetic Data Generator

This directory contains tools for generating synthetic training data for tool classifiers.

## Quick Test Script

The `test_generator.py` script allows you to quickly test the synthetic data generator on a specific tool:

```bash
# Basic usage
python test_generator.py ../../tools/sample_tool.py

# With custom options
python test_generator.py ../../tools/sample_tool.py \
  --examples_per_temp 15 \
  --output_dir ./my_test_output \
  --skip_llm_review
```

### Command Line Options

- `tool_path` (required): Path to the tool Python file to analyze
- `--examples_per_temp`: Base number of examples per temperature (default: 20)
- `--skip_llm_review`: Skip LLM quality review for faster generation
- `--output_dir`: Directory to save output files (default: ./test_output)
- `--analysis_model`: Model for code analysis (default: claude-3-7-sonnet-20250219)
- `--generation_model`: Model for example generation (default: claude-3-haiku-20240307)

The script will:
1. Analyze the tool to understand its capabilities
2. Generate examples using a dual-model approach (Sonnet for analysis, Haiku for generation)
3. Save the results to the specified output directory
4. Print a summary of the generation process

## Full Generator

For more advanced usage, use the main `synthetic_data_generator.py` script directly:

```bash
# Generate for a specific tool
python synthetic_data_generator.py --tool_file ../../tools/sample_tool.py

# Generate for all tools in a directory
python synthetic_data_generator.py --tool_dir ../../tools

# Generate multi-tool examples
python synthetic_data_generator.py --tool_dir ../../tools --multi_tool
```

See `python synthetic_data_generator.py --help` for all available options.

## Implementation Details

The generator uses a multi-stage approach with enhanced performance:

1. **Basic Analysis Phase** (using Claude 3.7 Sonnet):
   - Analyzes the tool code to extract operations and parameters
   - Provides foundation for deep operation analysis
   - Efficiently extracts structure without using thinking budget

2. **Extended Thinking Analysis** (using Claude 3.7 Sonnet with thinking enabled):
   - Categorizes overall tool complexity as simple/standard/complex
   - Classifies operations as core/standard/fringe based on importance
   - Assigns relative weights (1-10) to prioritize operations
   - Allocates examples budget (30-60) based on tool complexity
   - Groups similar operations to avoid redundant examples
   - Uses Claude's extended thinking capability (4000 token budget)

3. **Semantic Clustering** (new):
   - Groups operations by functional similarity
   - Identifies overlapping functionality using name and description analysis
   - Reduces example count for operations with semantic overlap
   - Ensures better coverage with fewer examples

4. **Priority-Weighted Generation** (using Claude 3.5 Haiku):
   - Distributes examples by semantic importance
   - Guarantees minimum examples by importance level (4 for core, 2 for standard, 1 for fringe)
   - Enforces hard cap of 60 total examples
   - Generates diverse examples for each operation
   - Uses multiple temperature settings for better diversity
   - Performs semantic deduplication to remove similar examples
   - Includes enhanced review process for better quality

5. **Parallel Processing** (new):
   - Analyzes tools in parallel for CPU-intensive operations
   - Runs example generation with controlled parallelism for API efficiency
   - Configurable parallel tool processing to balance speed and API load

This approach ensures core functionality gets better coverage while optimizing for efficiency and capping total example count.