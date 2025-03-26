#!/bin/bash

# Define the directories to check
dirs=("." "api" "config" "persistent" "tools")

# Output file
output_file="flake8_report.txt"

# Current directory
current_dir=$(pwd)

# Clear previous output
echo "Flake8 Report - $(date)" > "$output_file"

# Iterate over each directory
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then  # Check if directory exists
        echo "Checking directory: $dir" >> "$output_file"
        echo "------------------------" >> "$output_file"
        
        # For root directory, only check Python files directly in it (not in subdirectories)
        if [ "$dir" = "." ]; then
            for file in $(find "$dir" -maxdepth 1 -type f -name "*.py"); do
                echo "Checking $file..."
                echo -e "\n--- $file ---" >> "$output_file"
                flake8 "$file" >> "$output_file" 2>&1 || echo "No issues found" >> "$output_file"
                echo "" >> "$output_file"  # Add a blank line for readability
            done
        else
            # For other directories, check all Python files recursively
            for file in $(find "$dir" -type f -name "*.py"); do
                echo "Checking $file..."
                echo -e "\n--- $file ---" >> "$output_file"
                flake8 "$file" >> "$output_file" 2>&1 || echo "No issues found" >> "$output_file"
                echo "" >> "$output_file"  # Add a blank line for readability
            done
        fi
    else
        echo "Directory $dir not found, skipping" >> "$output_file"
    fi
done

echo "Flake8 check complete. Results saved in $output_file"

