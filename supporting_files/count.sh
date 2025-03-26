#!/bin/bash

# Script to count files, lines, and characters in the current directory
# while respecting .gitignore rules

echo "Counting files, lines, and characters (respecting .gitignore)..."
echo "===================================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. This script requires git to parse .gitignore rules."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Warning: Not in a git repository. Creating temporary git repo to use .gitignore..."
    # Create a temporary git repository
    TEMP_GIT=true
    git init -q
else
    TEMP_GIT=false
fi

# Function to clean up temporary git repo if needed
cleanup() {
    if [ "$TEMP_GIT" = true ]; then
        echo "Cleaning up temporary git repository..."
        rm -rf .git
    fi
}

# Set up trap to ensure cleanup happens on exit
trap cleanup EXIT

# Get list of all files that are not ignored by git
files=$(git ls-files --others --exclude-standard --cached)

# Initialize counters
file_count=0
line_count=0
char_count=0

# Process each file
while IFS= read -r file; do
    # Skip if it's a directory
    if [ -d "$file" ]; then
        continue
    fi
    
    # Count this file
    ((file_count++))
    
    # Count lines and characters in this file
    if [ -f "$file" ]; then
        file_lines=$(wc -l < "$file")
        file_chars=$(wc -m < "$file")
        
        ((line_count += file_lines))
        ((char_count += file_chars))
        
        # Optional: Show per-file statistics
        # echo "File: $file - Lines: $file_lines, Chars: $file_chars"
    fi
done <<< "$files"

# Print summary
echo "Summary:"
echo "Total files: $file_count"
echo "Total lines: $line_count"
echo "Total characters: $char_count"
echo "===================================================="

# Optional: Show top 5 largest files by line count
echo "Top 5 files by line count:"
for file in $files; do
    if [ -f "$file" ]; then
        wc -l "$file"
    fi
done | sort -rn | head -5

echo "===================================================="
# Optional: Show statistics by file extension
echo "Statistics by file extension:"
for file in $files; do
    if [ -f "$file" ]; then
        extension="${file##*.}"
        echo "$extension"
    fi
done | sort | uniq -c | sort -rn | head -10
