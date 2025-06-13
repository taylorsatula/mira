#!/bin/bash

# Initialize counters
total_files=0
total_lines=0
total_chars=0

# Check if we're in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    # Use git ls-files to respect .gitignore
    echo "Using git to respect .gitignore..."
    
    # Get list of Python files tracked by git or not ignored
    files=$(git ls-files '*.py' 2>/dev/null)
    
    # Also include untracked Python files that aren't ignored
    untracked=$(git ls-files --others --exclude-standard '*.py' 2>/dev/null)
    
    # Combine both lists
    all_files=$(echo -e "$files\n$untracked" | sort -u | grep -v '^$')
else
    # Not a git repository, use find
    echo "Not a git repository. Using find (won't respect .gitignore)..."
    all_files=$(find . -type f -name "*.py" 2>/dev/null | sed 's|^\./||')
fi

# Check if any Python files were found
if [ -z "$all_files" ]; then
    echo "No Python files found."
    exit 0
fi

echo "Analyzing Python files..."
echo "------------------------"

# Process each file
while IFS= read -r file; do
    if [ -f "$file" ]; then
        # Count lines and characters for this file
        file_stats=$(wc -l -m < "$file" 2>/dev/null)
        file_lines=$(echo "$file_stats" | awk '{print $1}')
        file_chars=$(echo "$file_stats" | awk '{print $2}')
        
        # Update totals
        total_files=$((total_files + 1))
        total_lines=$((total_lines + file_lines))
        total_chars=$((total_chars + file_chars))
        
        # Display file info
        printf "%-50s %6d lines, %8d chars\n" "$file" "$file_lines" "$file_chars"
    fi
done <<< "$all_files"

# Display summary
echo "------------------------"
echo "Summary:"
echo "  Total files: $total_files"
echo "  Total lines: $total_lines"
echo "  Total chars: $total_chars"

# Calculate averages if there are files
if [ $total_files -gt 0 ]; then
    avg_lines=$((total_lines / total_files))
    avg_chars=$((total_chars / total_files))
    echo "  Average lines per file: $avg_lines"
    echo "  Average chars per file: $avg_chars"
fi
