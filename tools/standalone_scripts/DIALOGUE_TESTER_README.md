# Dialogue Testing Tool

A tool for rapid testing of multi-turn dialogues with the system. This tool enables you to create, save, and run predefined conversation sequences to test tool activation, classification, and system responses. Results from test runs are automatically saved for analysis and comparison.

## Features

- **Reuse Test Dialogues**: Save and replay conversations to ensure consistent testing
- **Fast Iteration**: Quickly run through multi-turn conversations without manual input
- **Tool Activation Analysis**: Track which tools are activated during conversations
- **Performance Metrics**: Measure response times and system performance
- **Expected Response Validation**: Verify responses against expected content
- **Automatic Result Saving**: Save detailed test results to JSON files for analysis
- **Reproducible Tests**: Ensure consistent testing across system changes

## Usage

### List Available Test Dialogues

```bash
./dialogue_tester.py list
```

This command displays all saved dialogue test files with their descriptions and turn counts.

### Create a New Test Dialogue

```bash
./dialogue_tester.py create
```

This launches an interactive creation process that lets you define:
- Dialogue name and description
- User inputs for each turn
- Expected responses (optional)
- Additional notes (optional)

The dialogue is saved to the `persistent/test_dialogues` directory.

### Run a Test Dialogue

```bash
./dialogue_tester.py run /path/to/dialogue.json
```

This executes the specified dialogue file, displaying:
- User inputs and system responses for each turn
- Response times for each turn
- Active tools for each turn
- Validation against expected responses (if specified)
- Summary of tool activations and performance metrics

### Adding Pauses Between Turns

To add pauses between turns (useful for watching tool activations):

```bash
./dialogue_tester.py run /path/to/dialogue.json --pause 1.5
```

This adds a 1.5-second pause between each turn.

### Disabling Result Saving

By default, all test run results are saved to JSON files in the `persistent/test_results` directory. To disable this behavior:

```bash
./dialogue_tester.py run /path/to/dialogue.json --no-save
```

### Test Results

Test results are saved in JSON format in the `persistent/test_results` directory. The results contain detailed information about each test run:

```json
{
  "dialogue_name": "Test Name",
  "run_timestamp": "ISO date string",
  "total_time": 12.34,
  "avg_response_time": 3.45,
  "tool_activations": {
    "tool1": 2,
    "tool2": 1
  },
  "turns": [
    {
      "turn_number": 1,
      "user_input": "User input text",
      "response": "System response",
      "response_time": 2.45,
      "active_tools": ["tool1"],
      "expected_response": "Expected text",
      "contains_expected": true,
      "notes": "Notes about this turn"
    },
    // Additional turns...
  ]
}
```

You can use these result files to:
- Compare performance across system changes
- Track tool activation patterns
- Validate expected responses
- Monitor conversation stability over time
- Analyze failure cases

## Dialogue File Format

Test dialogues are stored as JSON files with the following structure:

```json
{
  "name": "Test Name",
  "description": "Test description",
  "created_at": "ISO date string",
  "turns": [
    {
      "user": "User input text",
      "expected_response": "Optional text that should appear in response",
      "notes": "Optional notes about this turn"
    },
    // Additional turns...
  ]
}
```

## Example Test Cases

Several example test dialogues are included:

1. **Email and Customer Database Mixed Test**:
   - Tests interactions that require both email and customer database tools
   - Verifies context preservation between related queries
   - Tests tool switching based on query content

2. **Complex Multi-Tool Workflow Test**:
   - Tests complex workflows requiring multiple tools
   - Verifies information sharing between tools
   - Tests memory persistence across turns with related context

## Tips for Effective Testing

1. **Start Simple**: Begin with single-tool tests before testing complex interactions
2. **Consistent Test Cases**: Use the same test cases before and after system changes
3. **Realistic Scenarios**: Design tests based on real user workflows
4. **Targeted Testing**: Create focused tests for specific failure modes or edge cases
5. **Progressive Complexity**: Add turns that progressively build on previous context

## Tool Feedback Integration

Use the `/toolfeedback` command as the final turn in test dialogues to capture:
- Analysis of tool classification performance
- Insights into semantic matching patterns
- Suggestions for training data improvements