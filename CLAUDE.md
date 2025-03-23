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
- **Simple Solutions First**: Consider simpler approaches before adding complexity - often the issue can be solved with a small fix

ø## Dependency Management
- **Minimal Dependencies**: Prefer standard library solutions over adding new dependencies; only introduce external libraries when absolutely necessary
- **Dependency Justification**: Document the specific reason for each dependency in comments or documentation when adding new requirements

## Interface Guidelines
- **Interface Preservation**: Maintain existing public interfaces when refactoring internals to ensure backward compatibility
- **Tool Interface Consistency**: Ensure all tool implementations follow the same patterns for input/output handling and error management
- **Response Formatting**: Adhere to established response structures and formatting conventions when modifying or adding agent outputs

## Tool Architecture
- **Tool Composition**: Favor composing existing tools together over creating new tools; let the LLM handle coordination between tools
- **Single Responsibility**: Each tool should have a single, well-defined purpose; avoid creating monolithic tools that combine multiple responsibilities
- **Logic Placement**: Keep business logic in the LLM prompts where possible rather than hardcoding it in tools
- **Tool Creation Criteria**: Only create new tools when functionality cannot be achieved by combining existing tools or when a composition would be used repeatedly

## Continuous Improvement
- **Self-Reflection**: After completing tasks, reflect on what went well and what could be improved
- **Pattern Recognition**: Identify recurring success patterns and failure patterns in your solutions
- **Feedback Integration**: When receiving feedback, document the specific insights and how they apply to future tasks
- **Solution Alternatives**: For each problem, consider multiple approaches before implementing a solution
- **Knowledge Persistence**: Update this CLAUDE.md file with new insights and guidelines based on feedback
- **Solution Simplification**: Periodically review completed solutions to identify unnecessary complexity
- **Anti-Patterns**: Document approaches to avoid based on past experiences
- **Learning Transfer**: Apply lessons from one part of the codebase to similar situations elsewhere
