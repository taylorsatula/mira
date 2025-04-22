"""
Tool finder module for the botwithmemory system (optimized version).

This module provides a specialized tool that can analyze user messages,
identify potentially relevant tools, and enable them for use.
This version uses abbreviated tool descriptions for better performance.
"""
import logging
import re
from typing import Dict, List, Any, Set, Optional

from tools.repo import Tool
from tools.repo import ToolRepository
from errors import ErrorCode, error_context, ToolError
from api.llm_bridge import LLMBridge


class ToolFinder2(Tool):
    """
    A meta-tool that analyzes user queries to find and enable relevant tools.
    
    This tool examines the user's message, compares it against the descriptions
    and capabilities of all available tools (but not yet enabled), and enables
    the most relevant tools that might help with the user's request.
    
    This is an optimized version that uses abbreviated tool descriptions
    for better performance and lower token usage.
    """
    
    name = "tool_finder"
    description = """Intelligently analyzes user messages to discover and enable the most appropriate tools for specific requests.

This meta-tool functions as an automated tool selector with sophisticated capabilities:

1. Tool Analysis and Selection:
   - Examines user messages to understand their intent and requirements
   - Evaluates both currently enabled tools and available-but-disabled tools
   - Recommends optimal tools based on relevance to the specific request
   - Prioritizes already-enabled tools before suggesting new ones

2. Smart Tool Management:
   - Automatically enables the most relevant tools (up to specified limit)
   - Provides detailed explanations for why specific tools were selected
   - Avoids unnecessary tool activation when existing tools can handle the request
   - Allows for adjustable relevance thresholds via 'force_enable' parameter

3. Configuration Options:
   - 'message': The user message to analyze (required)
   - 'max_tools_to_enable': Maximum number of new tools to enable (default: 1)
   - 'force_enable': Consider somewhat relevant tools, not just highly relevant ones
   - 'specific_tools': Restrict analysis to only these named tools

Use this tool when you need to dynamically discover and activate specialized capabilities based on evolving user requests."""
    usage_examples = [
        {
            "input": {
                "message": "What's the weather like in New York?",
                "max_tools_to_enable": 1
            },
            "output": {
                "tools_enabled": ["weather_tool"],
                "reason": "The message asks about weather information, which the weather_tool can provide."
            }
        },
        {
            "input": {
                "message": "Can you help me create a questionnaire about food preferences?",
                "max_tools_to_enable": 1
            },
            "output": {
                "tools_enabled": ["questionnaire_tool"],
                "reason": "The message requests creating a questionnaire, which the questionnaire_tool is designed for."
            }
        }
    ]
    
    # Abbreviated tool descriptions for faster LLM processing
    tool_summaries = {
        "tool_finder": "Analyzes user messages and enables appropriate tools for specific requests.",
        "tool_finder": "Analyzes user messages and enables appropriate tools for specific requests (optimized version).",
        "sample_tool": "Example tool demonstrating the proper structure and implementation of a bot tool.",
        "questionnaire_tool": "Creates, manages, and processes user surveys and questionnaires.",
        "http_tool": "Makes HTTP requests to external web services and APIs.",
        "calendar_tool": "Manages calendar events, appointments, and schedules.",
        "email_tool": "Sends, receives, and manages email messages.",
        "maps_tool": "Provides location, navigation, and map-related services.",
        "customer_tool": "Manages customer information, profiles, and relationships.",
        "quote_tool": "Generates quotes and pricing information for services.",
        "square_tool": "Processes payments and manages business operations using Square APIs.",
        "kasa_tool": "Controls smart home devices on the TP-Link Kasa platform."
    }
    
    def __init__(self, llm_bridge: LLMBridge, tool_repo: ToolRepository):
        """
        Initialize the tool finder tool.
        
        Args:
            llm_bridge: LLMBridge instance for generating suggestions
            tool_repo: ToolRepository instance to access and enable tools
        """
        super().__init__()
        self.llm_bridge = llm_bridge
        self.tool_repo = tool_repo
        
    def run(
        self,
        message: str,
        max_tools_to_enable: int = 1,
        force_enable: bool = False,
        specific_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the user message and enable relevant tools.
        
        Args:
            message: The user message to analyze
            max_tools_to_enable: Maximum number of tools to enable (default: 1)
            force_enable: If True, enables tools even if they seem only somewhat relevant
            specific_tools: Optional list of specific tool names to consider
            
        Returns:
            Dictionary containing the enabled tools and the reason
            
        Raises:
            ToolError: If tool analysis or enabling fails
        """
        self.logger.info(f"Analyzing message to find relevant tools: {message[:50]}...")
        
        with error_context(
            component_name=self.name,
            operation="analyzing message for tools",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Get all tools
            all_tools = set(self.tool_repo.list_all_tools())
            enabled_tools = set(self.tool_repo.list_enabled_tools())
            
            # Remove the tool_finder itself from enabled_tools for analysis
            enabled_tools_for_analysis = {t for t in enabled_tools if t != self.name and t != "tool_finder"}
            
            # Get available tools (not yet enabled)
            available_tools = all_tools - enabled_tools
            
            # If specific tools are provided, only consider those from available tools
            if specific_tools:
                if isinstance(specific_tools, str):
                    # Handle case where a single tool name is passed as string
                    specific_tools = [specific_tools]
                available_tools = {t for t in available_tools if t in specific_tools}
            
            # If no tools to analyze, return early
            if not enabled_tools_for_analysis and not available_tools:
                return {
                    "tools_enabled": [],
                    "reason": "No tools available to analyze for this request."
                }
            
            # Get abbreviated descriptions for enabled tools
            enabled_tool_descriptions = ""
            for tool_name in enabled_tools_for_analysis:
                summary = self.tool_summaries.get(tool_name, "No description available")
                enabled_tool_descriptions += f"Tool Name: {tool_name}\n"
                enabled_tool_descriptions += f"Description: {summary}\n\n"
            
            # Get abbreviated descriptions for available tools
            available_tool_descriptions = ""
            for tool_name in available_tools:
                summary = self.tool_summaries.get(tool_name, "No description available")
                available_tool_descriptions += f"Tool Name: {tool_name}\n"
                available_tool_descriptions += f"Description: {summary}\n\n"
            
            # Prepare prompt for the LLM
            relevance_threshold = "highly" if not force_enable else "somewhat"
            
            # Create a single prompt that first checks enabled tools, then considers available tools
            prompt = f"""
            User message: "{message}"
            
            First, determine if any ALREADY ENABLED tools can handle this request:
            
            ALREADY ENABLED TOOLS:
            {enabled_tool_descriptions or "No tools currently enabled."}
            
            ONLY IF none of the already enabled tools can handle the request well, consider these AVAILABLE TOOLS that could be enabled:
            {available_tool_descriptions or "No additional tools available."}
            
            Follow these steps in your analysis:
            1. First determine if ANY already enabled tools can handle the request well
            2. If an enabled tool works well, recommend ONLY that tool - don't enable new tools unnecessarily
            3. Only if NO enabled tools work well, recommend up to {max_tools_to_enable} available tools to enable
            4. Tools must be {relevance_threshold} relevant to the request to be considered
            
            Return your response in this EXACT format:
            
            Use existing tools: [Yes/No]
            Recommended enabled tools:
            - tool_name_1: Clear explanation why this tool works for the request
            
            Enable new tools: [Yes/No]
            Tools to enable:
            - tool_name_2: Clear explanation why this tool should be enabled
            
            If no tools are relevant in either category, use empty lists like:
            Recommended enabled tools: []
            Tools to enable: []
            """
            
            # Call the LLM to analyze the message and tools
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_bridge.generate_response(messages)
                content = self.llm_bridge.extract_text_content(response)
                
                # Parse the response
                use_existing = "use existing tools: yes" in content.lower()
                enable_new = "enable new tools: yes" in content.lower()
                
                # Extract recommended enabled tools
                enabled_tools_section = re.search(r'Recommended enabled tools:(.*?)(?=Enable new tools:|$)', 
                                               content, re.DOTALL)
                
                # Extract tools to enable
                tools_to_enable_section = re.search(r'Tools to enable:(.*?)$', 
                                                 content, re.DOTALL)
                
                recommended_enabled_tools = []
                tools_to_enable = []
                enabled_reasons = []
                enable_reasons = []
                
                # Process recommended enabled tools
                if enabled_tools_section and use_existing:
                    section_text = enabled_tools_section.group(1).strip()
                    if section_text and "[]" not in section_text:
                        tool_matches = re.finditer(r'-\s+([a-zA-Z0-9_]+):\s*(.*?)(?=\n-|\n\n|$)', section_text, re.DOTALL)
                        for match in tool_matches:
                            tool_name = match.group(1).strip()
                            reason = match.group(2).strip()
                            if tool_name in enabled_tools_for_analysis:
                                recommended_enabled_tools.append(tool_name)
                                enabled_reasons.append(reason)
                
                # Process tools to enable
                if tools_to_enable_section and enable_new and not recommended_enabled_tools:
                    section_text = tools_to_enable_section.group(1).strip()
                    if section_text and "[]" not in section_text:
                        tool_matches = re.finditer(r'-\s+([a-zA-Z0-9_]+):\s*(.*?)(?=\n-|\n\n|$)', section_text, re.DOTALL)
                        for match in tool_matches:
                            tool_name = match.group(1).strip()
                            reason = match.group(2).strip()
                            if tool_name in available_tools:
                                tools_to_enable.append(tool_name)
                                enable_reasons.append(reason)
                
                # If we found recommended already-enabled tools, return those
                if recommended_enabled_tools:
                    tool_recommendations = []
                    for i, tool_name in enumerate(recommended_enabled_tools):
                        if i < len(enabled_reasons):
                            tool_recommendations.append(f"{tool_name}: {enabled_reasons[i]}")
                        else:
                            tool_recommendations.append(tool_name)
                    
                    return {
                        "tools_enabled": [],
                        "already_suitable_tools": recommended_enabled_tools,
                        "reason": f"These already enabled tools can handle your request:\n" + "\n".join(tool_recommendations)
                    }
                
                # If we need to enable tools
                if tools_to_enable:
                    successfully_enabled = []
                    tool_recommendations = []
                    
                    for i, tool_name in enumerate(tools_to_enable):
                        try:
                            self.tool_repo.enable_tool(tool_name)
                            successfully_enabled.append(tool_name)
                            
                            # Add reason if available
                            if i < len(enable_reasons):
                                tool_recommendations.append(f"{tool_name}: {enable_reasons[i]}")
                            else:
                                tool_recommendations.append(tool_name)
                            
                            self.logger.info(f"Enabled tool: {tool_name}")
                        except Exception as e:
                            self.logger.error(f"Failed to enable tool {tool_name}: {e}")
                    
                    if successfully_enabled:
                        return {
                            "tools_enabled": successfully_enabled,
                            "reason": "Enabled tools based on your request:\n" + "\n".join(tool_recommendations)
                        }
                
                # If we got here, no suitable tools were found or enabled
                if enabled_tools_for_analysis:
                    enabled_list = ", ".join(enabled_tools_for_analysis)
                    return {
                        "tools_enabled": [],
                        "reason": f"No suitable tools found for this request. Already enabled tools ({enabled_list}) are not appropriate, and no new tools need to be enabled."
                    }
                else:
                    return {
                        "tools_enabled": [],
                        "reason": "No suitable tools found for this request."
                    }
                
            except Exception as e:
                self.logger.error(f"Error analyzing tools with LLM: {e}")
                return {
                    "tools_enabled": [],
                    "reason": f"Failed to analyze tools: {str(e)}"
                }
    
    def _parse_relevant_tools(self, content: str) -> List[str]:
        """
        Parse the LLM response to extract relevant tool names.
        
        Args:
            content: The LLM response content
            
        Returns:
            List of relevant tool names
        """
        # Check for empty list response
        if "relevant tools: []" in content.lower():
            return []
        
        # Use regex to extract tool names (assuming format "- tool_name: reason")
        tool_pattern = r'-\s+([a-zA-Z0-9_]+):'
        matches = re.findall(tool_pattern, content)
        
        relevant_tools = []
        for match in matches:
            # Verify the match is actually a tool name (not some other text)
            if match.strip() and not match.startswith("-"):
                relevant_tools.append(match.strip())
        
        return relevant_tools