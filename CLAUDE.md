# Bot With Memory - Python Project Guide

## Commands
- **Setup**: `pip install -r requirements.txt`
- **Run**: `python main.py`
- **Tests**: `pytest` or `pytest tests/test_file.py::test_function`
- **Lint**: `flake8`
- **Type check**: `mypy .`
- **Format**: `black .`

## Code Style
- **Imports**: Group stdlib, third-party, local imports; sort alphabetically
- **Formatting**: Use Black with 88 char line length
- **Types**: Use type hints for functions and class attributes
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Docstrings**: Google style docstrings for all public functions/methods
- **Error handling**: Use specific exceptions, always document raised exceptions
- **Logging**: Use the logging module instead of print statements
- **Tests**: Write unit tests for all public functions with pytest
- **Memory handling**: Use appropriate data structures for memory management

## Code Editing Best Practices
- **Code Removal**: Delete code completely when removing it rather than commenting it out or replacing it with explanatory comments
- **Problem Diagnosis**: Before making changes, thoroughly investigate the root cause by examining related files and dependencies
- **Minimal Changes**: Prefer targeted, minimal edits over adding new code structures or abstractions
- **Existing Patterns**: Follow the established patterns in the codebase rather than introducing new approaches
- **Step-by-Step Testing**: Make incremental changes with validation at each step rather than large refactors
- **Context Gathering**: When debugging or adding features, review related files to understand the project's architecture and implementation details
- **Style Consistency**: Ensure new code precisely matches the style, complexity level, and design patterns of existing files in the project
- **Simple Solutions First**: Consider simpler approaches before adding complexity - often the issue can be solved with a small fix, but never sacrifice correctness for simplicity
- **Fix Upstream Issues**: When seeing incorrect data or input formats, address the root source of the problem rather than adapting downstream components to handle incorrect formats
- **Detailed Documentation**: Add comprehensive docstrings with parameter descriptions, return types, and raised exceptions to all public methods
- **Full Tool Reference**: For creating new tools, refer to `HOW_TO_BUILD_A_TOOL.md` for step-by-step guidance and best practices

## Todo lists
- If the user asks "What is on my todo list" reference TODO.md

## Dependency Management
- **Minimal Dependencies**: Prefer standard library solutions over adding new dependencies; only introduce external libraries when absolutely necessary
- **Dependency Justification**: Document the specific reason for each dependency in comments or documentation when adding new requirements

## Interface Guidelines
- **Interface Correctness**: Ensure interfaces are used as designed. When encountering incorrect usage patterns, correct the calling code rather than adapting interfaces to accommodate misuse
- **Interface Preservation**: When existing interface designs are correct and established, maintain them when refactoring internals to ensure backward compatibility
- **Tool Interface Consistency**: Ensure all tool implementations follow the same patterns for input/output handling and error management
- **Response Formatting**: Adhere to established response structures and formatting conventions when modifying or adding agent outputs
- **Type Enforcement**: Honor type annotations as contracts. If a parameter is defined as a specific type (e.g., List[str]), enforce that type rather than accepting alternative formats

## Tool Architecture
- **Tool Composition**: Compose existing tools rather than creating new ones. Example: Combine extraction and persistence tools for preference management instead of building a dedicated tool.
- **Single Responsibility**: Design tools with focused functionality. Extraction tools should extract, persistence tools should store - separating concerns improves flexibility and reuse.
- **Logic Placement**: Use system prompts for business logic (like detecting preference statements) rather than hardcoding it in tools. This keeps the codebase cleaner and more adaptable.
- **Tool Creation Test**: Before creating a new tool, ask: "Can existing tools solve this problem?" Only create new tools when composition is insufficient or a particular combination will be used repeatedly.
- **Reference Implementation**: Use `tools/sample_tool.py` as a blueprint when creating new tools. It demonstrates the proper structure, error handling, testing approach, and documentation style.
- **Data Management**: Store persistent tool data in `data/tools/{tool_name}/` directory to maintain consistency with project structure.
- **Error Handling**: Always use the `error_context` manager from `errors.py` and raise the appropriate `ToolError` with a specific `ErrorCode` to ensure consistent error reporting across tools.
- **Tool Documentation**: Write detailed tool descriptions (see `TOOL_DEF_BESTPRACTICE.md`) that clearly explain what the tool does, when it should be used, all parameters, and any limitations.
- **Comprehensive Testing**: For new tools, create corresponding test files in `tests/` that verify both success paths and error conditions, following patterns in existing test files.

## Continuous Improvement
- **Self-Reflection**: After completing tasks, reflect on what went well and what could be improved
- **Pattern Recognition**: Identify recurring success and failure patterns across different solutions
- **Feedback Integration**: Convert specific feedback into general principles that guide future work
- **Solution Alternatives**: Consider multiple approaches before implementation, evaluating tradeoffs
- **Knowledge Capture**: Proactively update this CLAUDE.md file when discovering significant insights; don't wait for explicit requests to document learnings
- **Solution Simplification**: Periodically review solutions to identify and eliminate unnecessary complexity
- **Anti-Patterns**: Document specific approaches to avoid and the contexts where they're problematic
- **Learning Transfer**: Apply principles across different parts of the codebase, even when contexts appear dissimilar
- **Guideline Evolution**: Refine guidelines with concrete examples as implementation experience grows
- **Test Before Commit**: Never commit code changes without verification that they solve the problem; enthusiasm to fix issues shouldn't override testing discipline

## Effective Problem-Solving
- **Root Cause Analysis**: Focus on understanding underlying issues rather than addressing surface symptoms
- **Leverage Built-in Capabilities**: Use language/framework introspection and reflection for automatic pattern detection
- **Lifecycle Management**: Separate object lifecycle phases (creation, initialization, usage) for cleaner architecture
- **Incremental Enhancement**: Build upon existing patterns rather than introducing completely new approaches
- **Minimal Design**: Add just enough abstraction to solve both immediate issues and support future changes
- **Generic Solutions**: Design solutions for the general case that can handle variations of the same problem
- **Dependency Management**: Use proper dependency management patterns to reduce coupling between components

## Implementation Strategy
- **Plan Architectural Integration**: Before coding, map out all integration points and data flows through the system
- **Configuration-First Design**: Define configuration parameters before implementing functionality to ensure flexibility
- **Progressive Implementation**: Build complex features in stages - starting with core functionality before adding optimizations
- **Bookmark Strategically**: Use clear bookmark comments for future implementation points in complex multi-step features
- **Staged Testing**: When implementing complex features, add detailed logging to verify behavior at each step
- **Observability-Driven Development**: Add performance metrics and detailed logging from the beginning, not as an afterthought
- **Cross-Component Analysis**: Regularly analyze interactions between components to identify inefficiencies
- **Iterative Refinement**: Start with a working implementation, then refine based on real-world performance observations
- **Low-to-High Risk Progression**: Implement lower-risk functionality first to establish foundation before higher-risk components
- **Deliberate Timing Measurement**: Include performance measurement instrumentation for critical paths from the outset
