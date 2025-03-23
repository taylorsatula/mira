"""
Extraction tool for extracting information from user messages using LLM.

This tool uses the LLM Bridge to extract specific information from user 
messages based on configurable prompt templates.
"""
import logging
from typing import Dict, Any, Optional, List

from tools.repo import Tool
from api.llm_bridge import LLMBridge
from errors import ToolError, ErrorCode


# Extraction prompt templates
EXTRACTION_TEMPLATES = {
    "general": "Extract the following information from the message: {target}. Return ONLY the extracted information, nothing else.",
    
    "personal_info": """Extract any personal information from the message such as:
- Name
- Age
- Location
- Preferences
- Goals
Return only the extracted information in JSON format with appropriate keys.""",
    
    "keywords": "Extract the main keywords from the message. Return only a list of keywords, separated by commas.",
    
    "question": "Is there a question in this message? If so, extract it. Return only the question itself or 'No question found'.",
    
    "sentiment": "Analyze the sentiment of this message. Return only one word: positive, negative, or neutral.",
    
    "entities": "Extract named entities (people, places, organizations, products) from the message. Return only the entities in JSON format with appropriate type labels."
}


class ExtractionTool(Tool):
    """
    Tool for extracting specific information from user messages using LLM.
    
    Uses prompt templates to guide the extraction process for different
    types of information.
    """
    
    name = "extract"
    description = "Extract specific information from user messages using LLM"
    usage_examples = [
        {
            "input": {"message": "I'm John, I live in New York and I like jazz.", "template": "personal_info"},
            "output": {"extracted": {"name": "John", "location": "New York", "preferences": "jazz"}}
        },
        {
            "input": {"message": "What's the weather like in Paris?", "template": "question"},
            "output": {"extracted": "What's the weather like in Paris?"}
        }
    ]
    
    def __init__(self):
        """Initialize the extraction tool with LLM bridge."""
        super().__init__()
        self.llm_bridge = LLMBridge()
    
    def run(
        self, 
        message: str,
        template: str = "general",
        target: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Extract information from a message using the specified template.
        
        Args:
            message: The message to extract information from
            template: Name of the extraction template to use
            target: Target information to extract (used with 'general' template)
            temperature: LLM temperature (lower for more precise extractions)
            
        Returns:
            Dictionary with extracted information
            
        Raises:
            ToolError: If extraction fails or template is invalid
        """
        self.logger.info(f"Extracting information using template: {template}")
        
        # Validate template
        if template not in EXTRACTION_TEMPLATES and template != "custom":
            templates = list(EXTRACTION_TEMPLATES.keys())
            raise ToolError(
                f"Invalid template: {template}. Must be one of {templates} or 'custom'",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            # Get prompt template
            prompt = self._get_prompt(template, target)
            
            # Prepare messages for LLM
            messages = [
                {"role": "user", "content": f"Message: {message}\n\nInstruction: {prompt}"}
            ]
            
            # Set system prompt for extraction
            system_prompt = "You are an information extraction assistant. Your task is to extract specific information from messages accurately. Only return the requested information with no additional text, explanations, or acknowledgments."
            
            # Generate response using LLM Bridge
            response = self.llm_bridge.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=500
            )
            
            # Extract text content from response
            extracted = self.llm_bridge.extract_text_content(response)
            
            return {"extracted": extracted.strip()}
            
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            
            raise ToolError(
                f"Error extracting information: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _get_prompt(self, template: str, target: Optional[str]) -> str:
        """
        Get the prompt template for extraction.
        
        Args:
            template: Name of the template to use
            target: Target information for general template
            
        Returns:
            Prompt template string
            
        Raises:
            ToolError: If template requires target but none provided
        """
        if template == "general" and not target:
            raise ToolError(
                "Target parameter is required for 'general' template",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        if template == "custom":
            if not target:
                raise ToolError(
                    "Target parameter is required for 'custom' template and should contain the full prompt",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            return target
        
        # Get template string and format with target if needed
        template_str = EXTRACTION_TEMPLATES[template]
        if "{target}" in template_str and target:
            return template_str.format(target=target)
        
        return template_str
    
    def list_templates(self) -> List[str]:
        """
        Get a list of available extraction templates.
        
        Returns:
            List of template names
        """
        return list(EXTRACTION_TEMPLATES.keys()) + ["custom"]