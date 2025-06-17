#!/usr/bin/env python
"""
Embeddings Benchmark Tool

This script allows benchmarking of the ToolRelevanceEngine embedding quality
in isolation from the conversation system. It provides a simple interactive
interface to test queries against the engine and see detailed matching results.
"""
import os
import sys
import json
import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to sys.path to allow importing from parent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tool_relevance_engine import ToolRelevanceEngine, MultiLabelClassifier
from tools.repo import ToolRepository
from api.embeddings_provider import EmbeddingsProvider
from config import config


class EmbeddingsBenchmarkTool:
    """
    Tool for benchmarking embedding quality of the ToolRelevanceEngine.
    
    This tool provides an interactive interface for testing queries against
    the ToolRelevanceEngine and analyzing the results.
    """
    
    def __init__(self):
        """Initialize the benchmark tool with a ToolRelevanceEngine instance."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("embeddings_benchmark")
        
        # Create necessary directories
        os.makedirs(os.path.join(config.paths.data_dir, "tools"), exist_ok=True)
        os.makedirs(os.path.join(config.paths.data_dir, "classifier"), exist_ok=True)
        
        # Initialize repository and engine
        self.tool_repo = ToolRepository()
        self.tool_repo.discover_tools()
        
        # Initialize embeddings provider for tool classification
        self.embeddings_provider = EmbeddingsProvider(
            provider_type="local",
            enable_reranker=False
        )
        
        # Initialize engine - it will use existing cache if available
        self.engine = ToolRelevanceEngine(self.tool_repo, self.embeddings_provider)
        
        # Initialize test history
        self.test_history: List[Dict[str, Any]] = []
    
    def analyze_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Analyze a query using the ToolRelevanceEngine.
        
        Args:
            query: The query to analyze
            
        Returns:
            List of (tool_name, confidence_score) tuples
        """
        return self.engine.analyze_message(query)
    
    def get_tool_details(self, tool_name: str) -> Dict[str, Any]:
        """
        Get details about a specific tool.
        
        Args:
            tool_name: Name of the tool to get details for
            
        Returns:
            Dictionary with tool details
        """
        try:
            tool = self.tool_repo.get_tool(tool_name)
            metadata = tool.get_metadata()
            
            # Additional info from examples
            example_info = self._get_tool_example_info(tool_name)
            
            return {
                "metadata": metadata,
                "examples": example_info
            }
        except Exception as e:
            self.logger.error(f"Error getting tool details: {e}")
            return {"error": str(e)}
    
    def _get_tool_example_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get information about the examples used to train the classifier for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with example information
        """
        example_info = {
            "count": 0,
            "examples": [],
            "is_autogen": None
        }
        
        try:
            # Check if tool has examples in the engine
            if tool_name in self.engine.tool_examples:
                tool_data = self.engine.tool_examples[tool_name]
                examples = tool_data.get("examples", [])
                
                example_info["count"] = len(examples)
                example_info["examples"] = examples
                example_info["is_autogen"] = tool_data.get("is_autogen", False)
            
            return example_info
        except Exception as e:
            self.logger.error(f"Error getting example info: {e}")
            return example_info
    
    def get_classifier_threshold(self, tool_name: str) -> Optional[float]:
        """
        Get the classification threshold for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            The threshold value or None if not found
        """
        try:
            if hasattr(self.engine, 'classifier') and hasattr(self.engine.classifier, 'classifiers'):
                classifiers = self.engine.classifier.classifiers
                if tool_name in classifiers:
                    return classifiers[tool_name].get("threshold")
            return None
        except Exception:
            return None
    
    def get_comparison_details(self, query: str, results: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Get detailed comparison information for query results.
        
        Args:
            query: The original query
            results: List of (tool_name, confidence_score) tuples from analyze_query
            
        Returns:
            Dictionary with detailed comparison information
        """
        comparison = {
            "query": query,
            "matches": [],
            "similarity_details": {}
        }
        
        try:
            # Get embedding for the query
            query_embedding = None
            if hasattr(self.engine.classifier, '_compute_embedding'):
                query_embedding = self.engine.classifier._compute_embedding(query)
            
            # For each matched tool
            for tool_name, confidence in results:
                tool_info = self.get_tool_details(tool_name)
                threshold = self.get_classifier_threshold(tool_name)
                
                match_info = {
                    "tool_name": tool_name,
                    "confidence": confidence,
                    "threshold": threshold,
                    "examples_count": tool_info.get("examples", {}).get("count", 0),
                    "is_autogen": tool_info.get("examples", {}).get("is_autogen", False)
                }
                
                comparison["matches"].append(match_info)
                
                # If we have a query embedding, calculate similarity with examples
                if query_embedding and hasattr(self.engine.classifier, 'calculate_text_similarity'):
                    examples = []
                    for example in tool_info.get("examples", {}).get("examples", []):
                        if "query" in example:
                            ex_query = example["query"]
                            similarity = self.engine.classifier.calculate_text_similarity(query, ex_query)
                            examples.append({
                                "query": ex_query,
                                "similarity": similarity
                            })
                    
                    # Sort examples by similarity (highest first)
                    examples.sort(key=lambda x: x["similarity"], reverse=True)
                    
                    # Add to comparison details
                    comparison["similarity_details"][tool_name] = {
                        "examples": examples[:5]  # Top 5 most similar examples
                    }
            
            return comparison
        except Exception as e:
            self.logger.error(f"Error getting comparison details: {e}")
            return {"error": str(e), **comparison}
    
    def run_interactive(self):
        """Run an interactive session for testing queries."""
        print("=" * 80)
        print("Embeddings Benchmark Tool")
        print("=" * 80)
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for a list of commands.")
        print()
        
        while True:
            try:
                command = input("\nEnter query or command: ").strip()
                
                if command.lower() in ["exit", "quit"]:
                    break
                elif command.lower() == "help":
                    self._show_help()
                elif command.lower() == "history":
                    self._show_history()
                elif command.lower().startswith("compare "):
                    self._run_comparison(command[8:].strip())
                elif command.lower().startswith("save "):
                    self._save_results(command[5:].strip())
                elif command.lower() == "tools":
                    self._list_tools()
                elif command.lower().startswith("tool "):
                    self._show_tool_details(command[5:].strip())
                else:
                    # Treat as a query
                    self._run_query(command)
            
            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\nAvailable commands:")
        print("  query             - Analyze a query (default when no command is specified)")
        print("  help              - Show this help information")
        print("  exit/quit         - Exit the benchmark tool")
        print("  history           - Show test history")
        print("  compare <query>   - Get detailed comparison for a query")
        print("  save <filename>   - Save test history to a file")
        print("  tools             - List all available tools")
        print("  tool <tool_name>  - Show details for a specific tool")
    
    def _run_query(self, query: str):
        """Run a query and display results."""
        print(f"\nAnalyzing query: '{query}'")
        
        # Analyze the query
        results = self.analyze_query(query)
        
        # Format and display results
        print("\nResults:")
        if not results:
            print("  No relevant tools found")
        else:
            for i, (tool_name, confidence) in enumerate(results, 1):
                threshold = self.get_classifier_threshold(tool_name)
                threshold_str = f" (threshold: {threshold:.4f})" if threshold is not None else ""
                print(f"  {i}. {tool_name}: {confidence:.4f}{threshold_str}")
        
        # Add to history
        self.test_history.append({
            "query": query,
            "results": results,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def _run_comparison(self, query: str):
        """Run a comparison query with detailed results."""
        print(f"\nRunning detailed comparison for: '{query}'")
        
        # Analyze the query
        results = self.analyze_query(query)
        
        # Get detailed comparison
        comparison = self.get_comparison_details(query, results)
        
        # Format and display results
        print("\nMatched Tools:")
        if not comparison["matches"]:
            print("  No relevant tools found")
        else:
            for i, match in enumerate(comparison["matches"], 1):
                tool_name = match["tool_name"]
                confidence = match["confidence"]
                threshold = match["threshold"]
                examples_count = match["examples_count"]
                is_autogen = match["is_autogen"]
                
                print(f"  {i}. {tool_name}: {confidence:.4f} (threshold: {threshold:.4f})")
                print(f"     Examples: {examples_count} {'(auto-generated)' if is_autogen else ''}")
                
                # Show similarity with top examples
                if tool_name in comparison["similarity_details"]:
                    sim_details = comparison["similarity_details"][tool_name]
                    print("     Most similar examples:")
                    for j, ex in enumerate(sim_details["examples"][:3], 1):
                        ex_query = ex["query"]
                        similarity = ex["similarity"]
                        # Truncate long example queries
                        if len(ex_query) > 80:
                            ex_query = ex_query[:77] + "..."
                        print(f"       {j}. [{similarity:.4f}] {ex_query}")
        
        # Add to history
        self.test_history.append({
            "query": query,
            "results": results,
            "comparison": comparison,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def _show_history(self):
        """Show test history."""
        print("\nTest History:")
        if not self.test_history:
            print("  No tests run yet")
        else:
            for i, test in enumerate(self.test_history, 1):
                query = test["query"]
                results = test["results"]
                result_str = f"{len(results)} matches" if results else "No matches"
                print(f"  {i}. '{query}' - {result_str}")
    
    def _save_results(self, filename: str):
        """Save test history to a file."""
        if not filename:
            filename = f"embedding_benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Ensure it has a .json extension
            if not filename.endswith(".json"):
                filename += ".json"
            
            # Create data directory if needed
            data_dir = os.path.join(config.paths.data_dir, "benchmark")
            os.makedirs(data_dir, exist_ok=True)
            
            # Full path to the output file
            output_path = os.path.join(data_dir, filename)
            
            with open(output_path, "w") as f:
                json.dump({
                    "tests": self.test_history,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tools_count": len(self.tool_repo.list_all_tools())
                }, f, indent=2)
            
            print(f"\nSaved test history to {output_path}")
        
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _list_tools(self):
        """List all available tools."""
        all_tools = self.tool_repo.list_all_tools()
        
        print(f"\nAvailable Tools ({len(all_tools)}):")
        for i, tool_name in enumerate(sorted(all_tools), 1):
            # Get example count
            example_info = self._get_tool_example_info(tool_name)
            examples_count = example_info.get("count", 0)
            is_autogen = example_info.get("is_autogen", False)
            
            autogen_str = " (auto-gen)" if is_autogen else ""
            print(f"  {i}. {tool_name}: {examples_count} examples{autogen_str}")
    
    def _show_tool_details(self, tool_name: str):
        """Show details for a specific tool."""
        if not tool_name:
            print("Error: Tool name is required")
            return
        
        try:
            # Get tool details
            tool_details = self.get_tool_details(tool_name)
            
            if "error" in tool_details:
                print(f"Error: {tool_details['error']}")
                return
            
            metadata = tool_details["metadata"]
            examples_info = tool_details["examples"]
            
            print(f"\nTool: {tool_name}")
            print(f"Description: {metadata['description']}")
            print(f"Examples: {examples_info['count']} {'(auto-generated)' if examples_info['is_autogen'] else ''}")
            
            # Get threshold
            threshold = self.get_classifier_threshold(tool_name)
            if threshold is not None:
                print(f"Threshold: {threshold:.4f}")
            
            # Parameters
            if metadata["parameters"]:
                print("\nParameters:")
                for param_name, param_info in metadata["parameters"].items():
                    required = " (required)" if param_name in metadata["required_parameters"] else ""
                    param_type = param_info.get("type", "string")
                    print(f"  - {param_name}{required}: {param_type}")
            
            # Show some examples
            if examples_info["examples"]:
                print("\nExample Queries:")
                for i, example in enumerate(examples_info["examples"][:5], 1):
                    if "query" in example:
                        print(f"  {i}. {example['query']}")
            
        except Exception as e:
            print(f"Error showing tool details: {e}")


def main():
    """Run the embeddings benchmark tool."""
    tool = EmbeddingsBenchmarkTool()
    tool.run_interactive()


if __name__ == "__main__":
    main()