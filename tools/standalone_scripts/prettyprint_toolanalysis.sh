#!/bin/bash

# Prettify JSON script
# Usage: ./prettify_json.sh path/to/file.json

if [ $# -eq 0 ]; then
    echo "Error: No file specified"
    echo "Usage: $0 path/to/file.json"
    exit 1
fi

FILE_PATH="$1"

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' not found"
    exit 1
fi

# Check if file is readable
if [ ! -r "$FILE_PATH" ]; then
    echo "Error: Cannot read file '$FILE_PATH'"
    exit 1
fi

# Check if file is writable
if [ ! -w "$FILE_PATH" ]; then
    echo "Error: Cannot write to file '$FILE_PATH'"
    exit 1
fi

# Check if file appears to be JSON
if ! grep -q '{' "$FILE_PATH"; then
    echo "Warning: File may not be valid JSON (no '{' found)"
    # Continue anyway, as this is just a warning
fi

# Create a temporary file
TEMP_FILE=$(mktemp)

# Prettify the JSON and store in temporary file
if command -v python3 >/dev/null 2>&1; then
    # Use Python if available (handles large files better)
    python3 -m json.tool "$FILE_PATH" > "$TEMP_FILE"
    RESULT=$?
elif command -v jq >/dev/null 2>&1; then
    # Use jq if available
    jq '.' "$FILE_PATH" > "$TEMP_FILE"
    RESULT=$?
elif command -v node >/dev/null 2>&1; then
    # Use Node.js if available
    node -e "const fs=require('fs');const file=fs.readFileSync('$FILE_PATH','utf8');console.log(JSON.stringify(JSON.parse(file),null,2))" > "$TEMP_FILE"
    RESULT=$?
else
    echo "Error: Requires either Python 3, jq, or Node.js to prettify JSON"
    rm -f "$TEMP_FILE"
    exit 1
fi

# Check if prettification was successful
if [ $RESULT -ne 0 ]; then
    echo "Error: Failed to prettify JSON (invalid JSON format?)"
    rm -f "$TEMP_FILE"
    exit 1
fi

# Check if temp file has content
if [ ! -s "$TEMP_FILE" ]; then
    echo "Error: Prettification produced empty output"
    rm -f "$TEMP_FILE"
    exit 1
fi

# Move the temporary file to the original file
mv "$TEMP_FILE" "$FILE_PATH"

echo "Successfully prettified: $FILE_PATH"
exit 0
