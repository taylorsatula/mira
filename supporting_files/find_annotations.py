#!/usr/bin/env python3
import os
import json
import re
import argparse
from pathlib import Path


def find_annotations(root_dir, output_file):
    """Find all #ANNOTATION or # ANNOTATION comments in code files.
    
    Args:
        root_dir: Root directory to start search from
        output_file: Path to output JSON file
    """
    annotations = []
    pattern = re.compile(r'#\s*ANNOTATION')
    
    # Extensions to search - add more if needed
    code_extensions = {'.py', '.js', '.ts', '.html', '.css', '.md', '.txt'}
    
    # Directories to skip
    skip_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.vscode'}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip directories we want to exclude
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            # Check if file extension is in our list
            ext = os.path.splitext(file)[1]
            if ext not in code_extensions:
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if pattern.search(line):
                            annotations.append({
                                "filename": os.path.relpath(file_path, root_dir),
                                "linenumber": i,
                                "annotation": line.strip()
                            })
            except UnicodeDecodeError:
                # Skip binary files
                continue
    
    # Write results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2)
    
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find code annotations.')
    parser.add_argument('--root', '-r', default='.', 
                        help='Root directory to start search (default: current directory)')
    parser.add_argument('--output', '-o', default='annotations.json',
                        help='Output JSON file (default: annotations.json)')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    output_file = args.output
    
    annotations = find_annotations(root_dir, output_file)
    
    print(f"Found {len(annotations)} annotations")
    print(f"Results saved to {output_file}")