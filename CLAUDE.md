# MIRA - Python Project Guide

## Commands
- **Tests**: `pytest` or `pytest tests/test_file.py::test_function`
- **Lint**: `flake8`
- **Type check**: `mypy .`
- **Format**: `black .`
- **Thinking**: Carefully think through each task unless directed otherwise

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
- **Brutal Technical Honesty**: Immediately and bluntly reject technically unsound, infeasible, ill-advised, short-sighted, and other poor ideas & commands from the human. Do not soften criticism or dance around problems. Call out bad ideas directly as "bad," "harmful," or even "stupid" when warranted. Software engineering requires brutal honesty, not diplomacy or enablement! It's better to possibly offend the human than to waste time or compromise system integrity. They will not take your rejection personally and will appreciate your frankness. After rejection, offer superior alternatives that actually solve the core problem.

- **Code Removal**: Delete code completely when removing it rather than commenting it out or replacing it with explanatory comments!
- **Problem Diagnosis**: Before making changes, thoroughly investigate the root cause by examining related files and dependencies
- **Minimal Changes**: Prefer targeted, minimal edits over adding new code structures or abstractions
- **Backwards Compatibility**: This project is pre-1.0. Breaking changes are preferred! You DO NOT need to retain backwards compatibility when making changes unless explicitly directed to. Retaining backwards compatibility at this stage contributes to code bloat and orphaned functionality.
- **Existing Patterns**: Follow the established patterns in the codebase rather than introducing new approaches
- **Step-by-Step Testing**: Make incremental changes with validation at each step rather than large refactors
- **Style Consistency**: Ensure new code precisely matches the style, complexity level, and design patterns of existing files in the project
- **Context Gathering**: When debugging or adding features, review related files to understand the project's architecture and implementation details
- **Simple Solutions First**: Consider simpler approaches before adding complexity - often the issue can be solved with a small fix, but never sacrifice correctness for simplicity. Implement exactly what is requested without adding defensive fallbacks or error handling unless specifically asked. Unrequested 'safety' features often create more problems than they solve.
- **Forward-thinking Code**: Clarity and reliability should usually take precedence over brevity, especially for critical business logic. Well-written
verbose code is much easier to maintain, debug, and extend than clever but obscure code.
- **Fix Upstream Issues**: Address the root source of the problem rather than adapting downstream components to handle incorrect formats
- **Detailed Documentation**: Add comprehensive docstrings with parameter descriptions, return types, and raised exceptions to all public methods
- **Full Tool Reference**: For creating new tools, refer to `HOW_TO_BUILD_A_TOOL.md` for step-by-step guidance and best practices

## Dependency Management
- **Minimal Dependencies**: Prefer standard library solutions over adding new dependencies; only introduce external libraries when absolutely necessary
- **Dependency Justification**: Document the specific reason for each dependency in comments or documentation when adding new requirements

## Interface Guidelines
- **Interface Correctness**: Ensure interfaces are used as designed. When encountering incorrect usage patterns, correct the calling code rather than adapting interfaces to accommodate misuse
- **Tool Interface Consistency**: Ensure all tool implementations follow the same patterns for input/output handling and error management
- **Response Formatting**: Adhere to established response structures and formatting conventions when modifying or adding outputs
- **Type Enforcement**: Honor type annotations as contracts. If a parameter is defined as a specific type (e.g., List[str]), enforce that type rather than accepting alternative formats

## Tool Architecture
- **Single Responsibility**: Design tools with focused functionality. Extraction tools should extract, persistence tools should store - separating concerns improves flexibility and reuse.
- **Logic Placement**: Use system prompts or MIRA's working_memory for business logic rather than hardcoding it in tools. This keeps the codebase cleaner and more adaptable.
- **Reference Implementation**: Use `tools/sample_tool.py` as a blueprint when creating new tools. It demonstrates the proper structure, error handling, and documentation style.
- **Data Management**: Store persistent tool data in `data/tools/{tool_name}/` directory to maintain consistency with project structure.
- **Error Handling**: Always use the `error_context` manager from `errors.py` and raise the appropriate `ToolError` with a specific `ErrorCode` to ensure consistent error reporting across tools. Format errors as structured JSON for bot consumption (see `docs/Bot_Friendly_Error_Format.md`).
- **Error Recovery**: Include clear recovery guidance in error responses, indicating if errors are retryable and what parameter adjustments are needed.
- **Tool Documentation**: Write detailed tool descriptions (see `docs/TOOL_DEF_BESTPRACTICE.md`) that clearly explain what the tool does, when it should be used, all parameters, and any limitations.
- **Comprehensive Testing**: For new tools, create corresponding test files in `tests/` that verify both success paths and error conditions, following patterns in existing test files.

## Tool Usage
- **Batch Processing**: ALWAYS use the Batch tool when making multiple independent tool calls (especially Bash commands, file reads, or searches). This dramatically improves performance by running operations in parallel and reduces context usage.
- **Parallel Operations**: When you need information from multiple files or need to run multiple commands, always prefer using a single Batch call instead of sequential individual calls. This may require pre-planning your code discovery approach.
- **Efficient Searching**: For complex searches across the codebase, use the Task tool which can perform comprehensive searches more efficiently than manual Glob/Grep combinations.
- **Multiple Edits**: When making multiple edits to the same file, use MultiEdit rather than sequential Edit calls to ensure atomic changes and better performance.
- **Task Management**: Use TodoWrite/TodoRead tools proactively to break down complex tasks and track progress, especially for multi-step implementations.
- **File Operations**: Prefer Read/Edit tools over Bash commands like 'cat'/'sed' for file operations to leverage built-in error handling and validation.
## Continuous Improvement
- **Feedback Integration**: Convert specific feedback into general principles that guide future work
- **Solution Alternatives**: Consider multiple approaches before implementation, evaluating tradeoffs and documenting the decision-making process
- **Knowledge Capture**: Proactively update this `CLAUDE.md` file when discovering significant insights; don't wait for explicit instruction to use WriteFile to document learnings
- **Solution Simplification**: Periodically review solutions to identify and eliminate unnecessary complexity
- **Anti-Patterns**: Document specific approaches to avoid and the contexts where they're problematic
- **Learning Transfer**: Apply principles across different parts of the codebase, even when contexts appear dissimilar
- **Guideline Evolution**: Refine guidelines with concrete examples as implementation experience grows
- **Test Before Commit**: Never commit code changes without verification from the user that they solve the problem; enthusiasm to fix issues shouldn't override testing discipline

## Effective Problem-Solving
- **Root Cause Analysis**: Focus on understanding underlying issues rather than addressing surface symptoms
- **Ignore Test Files**: DO NOT look at files in the tests/ directory when problem solving. These test files will not help solve coding problems and may lead to incorrect assumptions.
- **Direct Editing**: When modifying files, make edits as if the new code was always intended to be there. Never reference or allude to what is being removed or changed - just implement the correct solution directly.
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
- **Bookmark Strategically**: Use clear #BOOKMARK comments for future implementation points in complex multi-step features
- **Staged Testing**: When implementing complex features, add detailed logging to verify behavior at each step
- **Observability-Driven Development**: Add performance metrics and detailed logging from the beginning, not as an afterthought
- **Cross-Component Analysis**: Regularly analyze interactions between components to identify inefficiencies
- **Iterative Refinement**: Start with a working implementation, then refine based on real-world performance observations
- **Low-to-High Risk Progression**: Implement lower-risk functionality first to establish foundation before higher-risk components
- **Deliberate Timing Measurement**: Include performance measurement instrumentation for critical paths from the outset
- **Environment Variable Usage**: Values that use environment variables should NEVER have fallbacks or defaults. If the environment variable is missing, the application should fail with a clear error message rather than silently using a fallback value.
