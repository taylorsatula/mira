# Tool Definition Best Practices

To get the best performance out of Claude when using tools, follow these comprehensive guidelines for tool descriptions.

## Structure Your Tool Description

For optimal results, structure your tool descriptions with these clear sections:

1. **Introduction (1-2 sentences)**: Concisely explain what the tool does and its primary purpose

2. **Operations Section (most important)**:
   - List each operation the tool supports with a brief description
   - For each operation, clearly document:
     - All required parameters (marked as required)
     - All optional parameters with their default values
     - What each parameter does and its expected format
     - What the operation returns

3. **Conceptual Categories/Features**: 
   - Explain any categorization or grouping the tool might do
   - Describe special features and how they work

4. **Usage Notes**:
   - Provide practical guidance on when and how to use the tool
   - Include common workflows and usage patterns
   - Highlight efficient usage strategies

5. **Limitations**:
   - Clearly state what the tool cannot do
   - Note any restrictions or potential issues

## Formatting Best Practices

- Use clear section headers (ALL CAPS or markdown) to organize the description
- Maintain consistent indentation for parameters under each operation
- Use concise parameter descriptions in format: `parameter_name (required/optional): Description`
- Group related operations together
- Keep explanations brief but complete - be thorough without being verbose

## Content Guidelines

- **Comprehensive Coverage**: Ensure every operation and parameter is documented
- **Practical Context**: Include when to use (and when not to use) each operation
- **Parameter Details**: Explain format requirements (e.g., "YYYY-MM-DD format")
- **Return Information**: Describe what data is returned for each operation
- **Error Cases**: Note common error conditions and their meaning when relevant

## Examples

While the description is most important, well-structured examples showing different operations and parameter combinations are extremely helpful. Place examples in the `usage_examples` list after completing your description.

## Sample Description Format

```
API management tool that provides access to an external data service. Use this tool to search, retrieve, update, and manage records in the external system.

OPERATIONS:
- get_records: Retrieve records from the system based on specified criteria
  Parameters:
    query (optional): Search query string to filter results
    category (optional, default="all"): Category of records to search in
    limit (optional, default=10): Maximum number of records to return
    include_archived (optional, default=False): Whether to include archived records

- get_record_by_id: Retrieve a specific record by its unique identifier
  Parameters:
    record_id (required): Unique identifier of the record to retrieve
    include_details (optional, default=True): Whether to include full record details

- update_record: Update an existing record in the system
  Parameters:
    record_id (required): Unique identifier of the record to update
    fields (required): Dictionary of field names and values to update
    notify_users (optional, default=False): Whether to notify associated users

- create_record: Create a new record in the system
  Parameters:
    record_type (required): Type of record to create
    fields (required): Dictionary of field names and values for the new record
    tags (optional): List of tags to associate with the record
    priority (optional, default="normal"): Priority level of the record

DATA CATEGORIZATION:
The tool organizes records into different types:
- active: Currently active records
- pending: Records waiting for processing
- archived: Historical records no longer in active use
- flagged: Records marked for special attention

IMPORTANT USAGE NOTES:
- Always check if a record exists before attempting to update it
- For bulk operations, use the appropriate bulk_ prefixed operations
- When creating records, ensure all required fields for that record_type are provided
- Record IDs are persistent identifiers for referencing records
- For complex searches, use the advanced_search operation with custom filters

LIMITATIONS:
- Cannot access records marked as restricted
- Bulk operations limited to 100 records at a time
- Updates may take up to 60 seconds to propagate through the system
- Historical data older than 5 years may not be available
```

## Checklist Before Finalizing

- Does the description cover ALL operations?
- Are ALL parameters documented with clear explanations?
- Is the format consistent and organized with clear sections?
- Would a user know exactly how to use each operation from your description?
- Are important limitations and requirements clearly stated?
- Does the description distinguish between similar operations?

The more structured and detailed your tool descriptions, the more effectively Claude will be able to select and use your tools.