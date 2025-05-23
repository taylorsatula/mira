#!/usr/bin/env python3
"""
Claude Web Search Tool

This script uses Anthropic's Claude API with web search capabilities
to perform internet searches and return structured results.

Usage:
  python claude_search.py QUERY [options]

Arguments:
  QUERY                   The search query to execute

Options:
  -h, --help               Show this help message and exit
  -n, --num-results NUM    Maximum number of results to return (default: 3)
  -o, --output FILE        Write output to FILE instead of stdout
  -a, --allowed DOMAINS    Comma-separated list of allowed domains
  -b, --blocked DOMAINS    Comma-separated list of blocked domains
  -f, --format FORMAT      Output format: text|json|markdown (default: text)
  --raw                    Show raw API response (for debugging)
  --verbose                Enable verbose logging output
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import anthropic

# Add the parent directory to sys.path to import project modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from config import config


def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("claude_search")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search the web using Claude's web search API."
    )
    
    parser.add_argument(
        "query",
        help="The search query to execute"
    )
    
    parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=3,
        help="Maximum number of results to return (default: 3)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Write output to FILE instead of stdout"
    )
    
    parser.add_argument(
        "-a", "--allowed",
        help="Comma-separated list of allowed domains"
    )
    
    parser.add_argument(
        "-b", "--blocked",
        help="Comma-separated list of blocked domains"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw API response (for debugging)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    return parser.parse_args()


def configure_web_search_tool(max_uses, allowed_domains=None, blocked_domains=None):
    """
    Configure the web search tool for Claude.
    
    Args:
        max_uses: Maximum number of searches to perform
        allowed_domains: List of allowed domains (optional)
        blocked_domains: List of blocked domains (optional)
        
    Returns:
        Dictionary with web search tool configuration
    """
    # Build the web search tool definition
    web_search_tool = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }
    
    # Add domain filters if provided
    if allowed_domains:
        web_search_tool["allowed_domains"] = allowed_domains
    
    if blocked_domains:
        web_search_tool["blocked_domains"] = blocked_domains
    
    return web_search_tool


def execute_search(client, query, web_search_tool, output_format, logger):
    """
    Execute the search using Claude's API.
    
    Args:
        client: Anthropic API client
        query: The search query
        web_search_tool: Configured web search tool
        output_format: Desired output format
        logger: Logger instance
        
    Returns:
        Search results as formatted text
    """
    # Create a system prompt that instructs Claude on how to format the results
    format_instruction = ""
    if output_format == "json":
        format_instruction = """
        Return the results as a JSON array of objects with these properties:
        - title: A descriptive title for the search result
        - url: The source URL
        - snippet: A brief snippet or summary of the content
        - source: The domain name of the source

        Wrap your response in ```json and ``` tags.
        """
    elif output_format == "markdown":
        format_instruction = """
        Format your response as Markdown with:
        - Each result as a section with the title as a heading
        - The URL as a proper markdown link
        - Content as paragraphs of text
        - Bullet points for any lists of information
        - Use proper markdown formatting for any quotes or code
        """
    else:  # text format (default)
        format_instruction = """
        Format your response as plain text with:
        - Each result clearly separated with titles and URLs
        - Clear, concise summaries of the key information
        - Proper attribution for any direct quotes or specific facts
        """
    
    system_prompt = f"""
    You are a helpful research assistant. Your task is to search the web for information 
    on the provided query and summarize the results. Please follow these guidelines:
    
    - Use the web_search tool to find relevant information
    - Provide accurate, up-to-date information from reputable sources
    - Focus on factual information, not opinions
    - Cite sources clearly for each piece of information
    - Organize the information in a logical, easy-to-read format
    
    {format_instruction}
    """
    
    # Prepare the user message with the search query
    user_message = f"Search for: {query}"
    
    logger.info(f"Executing search for: {query}")
    
    try:
        # Make the API call
        start_time = time.time()
        
        response = client.messages.create(
            model=config.api.model,
            max_tokens=1024,
            temperature=0.2,  # Lower temperature for more factual responses
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=[web_search_tool]
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f} seconds")
        
        # Extract the text content from the response
        result_content = ""
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == "text":
                result_content += content_block.text
        
        return result_content, response
    
    except Exception as e:
        logger.error(f"Error executing search: {str(e)}")
        raise


def format_raw_response(response):
    """Format the raw API response for debugging."""
    # Create a simplified version of the response for display
    simplified = {
        "id": response.id,
        "model": response.model,
        "role": response.role,
        "content": []
    }
    
    # Extract and simplify content blocks
    for block in response.content:
        if hasattr(block, 'type'):
            if block.type == "text":
                simplified["content"].append({
                    "type": "text",
                    "text": block.text[:200] + ("..." if len(block.text) > 200 else "")
                })
            elif block.type == "tool_use":
                simplified["content"].append({
                    "type": "tool_use",
                    "name": block.name,
                    "id": block.id,
                    "input": block.input
                })
            elif block.type == "tool_result":
                simplified["content"].append({
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content[:200] + ("..." if len(block.content) > 200 else "")
                })
    
    # Add usage information
    simplified["usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }
    
    return json.dumps(simplified, indent=2)


def main():
    """Run the Claude search script."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    # Parse domain lists if provided
    allowed_domains = args.allowed.split(",") if args.allowed else None
    blocked_domains = args.blocked.split(",") if args.blocked else None
    
    try:
        # Initialize Anthropic client
        api_key = config.api_key
        if not api_key:
            logger.error("API key not found in configuration")
            print("Error: API key not found. Please set it in your config.")
            sys.exit(1)
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Configure the web search tool
        web_search_tool = configure_web_search_tool(
            max_uses=args.num_results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains
        )
        
        # Execute the search
        result_content, raw_response = execute_search(
            client=client,
            query=args.query,
            web_search_tool=web_search_tool,
            output_format=args.format,
            logger=logger
        )
        
        # Process the output
        if args.raw:
            output = format_raw_response(raw_response)
        else:
            output = result_content
        
        # Output the result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info(f"Output written to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()