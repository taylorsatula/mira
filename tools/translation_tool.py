import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from api.llm_bridge import LLMBridge
from config.registry import registry

# Define configuration class for TranslationTool
class TranslationToolConfig(BaseModel):
    """Configuration for the translation_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    # Add configuration fields specific to Translation
    default_model: str = Field(default="claude-3-7-sonnet-20250219", description="Default model for translations")
    temperature: float = Field(default=0.3, description="Temperature setting for translation generation")

# Register with registry
registry.register("translation_tool", TranslationToolConfig)


class TranslationTool(Tool):
    """
    Translates communicative intent between languages with context awareness.
    
    This tool conveys semantic meaning and intention in a culturally appropriate way,
    taking into account contextual information that might affect the translation.
    It generates natural-sounding expressions that native speakers would use in the
    given context and situation.
    
    Uses Claude 3.5 Sonnet model specifically for optimal translation quality.
    """
    
    name = "translation_tool"
    simple_description = """
    Translates communicative intent into different languages in a natural, culturally 
    appropriate way, considering provided context. Use this tool when users want to know 
    how to express something in another language with proper cultural and situational awareness."""
    
    implementation_details = """
    
    This tool does not perform word-for-word translation, but rather understands what 
    the user wants to communicate and produces natural phrasing that a native speaker 
    would use in the specific context. It adjusts for different formality levels, 
    social situations, and can provide pronunciation guides or cultural usage notes.
    
    Parameters:
    - intent: The core message to translate (what you want to say/ask)
    - target_language: The language to translate the intent into
    - context: Additional background information that affects how the intent should be expressed
    - formality_level: Level of formality (casual, neutral, formal, honorific) - defaults to neutral
    - include_pronunciation: Whether to include a pronunciation guide - defaults to false
    - include_cultural_notes: Whether to include cultural usage notes - defaults to false
    
    This tool is particularly useful for:
    1. Preparing to communicate in foreign countries with proper social awareness
    2. Learning culturally appropriate expressions for specific situations
    3. Generating natural-sounding phrases beyond what simple translation tools provide
    
    The tool uses Claude 3.7 Sonnet specifically for optimal translation quality.
    """
    
    description = simple_description + implementation_details
    
    usage_examples = [
        {
            "input": {
                "intent": "where is the nearest grocery store",
                "target_language": "Japanese",
                "context": "I am a tourist and appear obviously foreign",
                "formality_level": "polite"
            },
            "output": {
                "translation": "一番近いスーパーはどこですか？",
                "pronunciation": "Ichiban chikai suupaa wa doko desu ka?",
                "cultural_notes": "Using polite form suitable for asking locals for directions as a foreigner",
                "source_intent": "where is the nearest grocery store",
                "target_language": "Japanese"
            }
        }
    ]
    
    def __init__(self, llm_bridge: LLMBridge):
        """
        Initialize the translation tool.
        
        Args:
            llm_bridge: LLMBridge instance for communicating with Claude
        """
        super().__init__()
        self.llm_bridge = llm_bridge
        self.logger.info("Translation tool initialized")
        
        # Supported formality levels
        self.formality_levels = ["casual", "neutral", "formal", "honorific"]
        
    def run(
        self,
        intent: str,
        target_language: str,
        context: Optional[str] = None,
        formality_level: str = "neutral",
        include_pronunciation: bool = False,
        include_cultural_notes: bool = False
    ) -> Dict[str, Any]:
        """
        Translate semantic intent into the target language with context awareness.
        
        Args:
            intent: The core message to translate (what you want to say/ask)
            target_language: The language to translate into
            context: Additional background information that affects how the intent should be expressed
            formality_level: Level of formality (casual, neutral, formal, honorific)
            include_pronunciation: Whether to include a pronunciation guide
            include_cultural_notes: Whether to include cultural usage notes
            
        Returns:
            Dictionary containing translation and optional additional information
            
        Raises:
            ToolError: If inputs are invalid or translation fails
        """
        # Import config when needed (avoids circular imports)
        from config import config
        self.logger.info(f"Translating intent to {target_language} with context using {config.translation_tool.default_model}")
        
        # Input validation using error_context
        with error_context(
            component_name=self.name,
            operation="validating inputs",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_INVALID_INPUT,
            logger=self.logger
        ):
            # Validate intent
            if not intent or not isinstance(intent, str):
                raise ToolError(
                    "Intent must be a non-empty string",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_intent": str(intent)}
                )
                
            # Validate target language
            if not target_language or not isinstance(target_language, str):
                raise ToolError(
                    "Target language must be a non-empty string",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_language": str(target_language)}
                )
                
            # Validate formality level
            if formality_level not in self.formality_levels:
                raise ToolError(
                    f"Invalid formality level: {formality_level}. Must be one of: {', '.join(self.formality_levels)}",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_formality": formality_level, "valid_levels": self.formality_levels}
                )
        
        # Perform translation
        with error_context(
            component_name=self.name,
            operation="translating content",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Generate system prompt
            system_prompt = self._create_system_prompt(
                target_language, 
                formality_level,
                include_pronunciation,
                include_cultural_notes
            )
            
            # Create user message combining intent and context
            user_message = self._format_user_message(intent, context)
            
            # Create message for Claude
            messages = [
                {"role": "user", "content": user_message}
            ]
            
            # Make LLM call with explicit Sonnet model
            try:
                # Import config when needed (avoids circular imports)
                from config import config
                
                response = self.llm_bridge.generate_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    # Explicitly override model to use Sonnet
                    model=config.translation_tool.default_model,
                    # Use temperature from configuration
                    temperature=config.translation_tool.temperature
                )
                
                # Extract translation
                translation_text = self.llm_bridge.extract_text_content(response)
                
                # Parse the response to separate translation from pronunciation/notes
                result = self._parse_translation_response(
                    translation_text, 
                    include_pronunciation,
                    include_cultural_notes
                )
                
                # Return results
                return {
                    "translation": result.get("translation", ""),
                    "pronunciation": result.get("pronunciation", "") if include_pronunciation else "",
                    "cultural_notes": result.get("cultural_notes", "") if include_cultural_notes else "",
                    "source_intent": intent,
                    "context": context or "",
                    "target_language": target_language,
                    "model_used": config.translation_tool.default_model
                }
            except Exception as e:
                # If Sonnet fails, log the error and re-raise
                self.logger.error(f"Translation failed with model {config.translation_tool.default_model}: {str(e)}")
                raise ToolError(
                    f"Translation failed: {str(e)}",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                    {"intent": intent, "target_language": target_language}
                )
    
    def _format_user_message(self, intent: str, context: Optional[str] = None) -> str:
        """
        Format the user message with intent and optional context.
        
        Args:
            intent: The core message to translate
            context: Optional contextual information
            
        Returns:
            Formatted message for the LLM
        """
        if context:
            return f"INTENT: {intent}\n\nCONTEXT: {context}"
        else:
            return f"INTENT: {intent}"
    
    def _create_system_prompt(
        self, 
        target_language: str, 
        formality_level: str,
        include_pronunciation: bool,
        include_cultural_notes: bool
    ) -> str:
        """
        Create a system prompt for the translation request.
        
        Args:
            target_language: Target language
            formality_level: Formality level
            include_pronunciation: Whether to include pronunciation
            include_cultural_notes: Whether to include cultural notes
            
        Returns:
            Formatted system prompt
        """
        
        
        
        # This was the original one that worked okay in the limited examples I tried it on. New one is from Anthopic Console.
        # You are a cultural and linguistic translator skilled in conveying communicative intent across languages.
        # 
        # Your task is to translate the user's INTENT into {target_language} at a {formality_level} level of formality.
        # The user may also provide CONTEXT information that should inform how you translate the intent.
        # 
        # Guidelines:
        # 1. Only translate the INTENT, not the CONTEXT
        # 2. Use the CONTEXT to determine appropriate phrasing, formality adjustments, and cultural adaptations
        # 3. Create a natural phrase that a native {target_language} speaker would use in the given situation
        # 4. Consider social dynamics, speaker/listener relationship, and situational appropriateness
        # 5. Do not translate word-for-word; focus on conveying the communicative goal naturally
        # 
        # For example, translating "where is the bathroom" would differ if the context is:
        # - "I am at a formal business dinner" (more polite/formal expression)
        # - "I am at a friend's house" (more casual expression)
        # - "I am a tourist asking a stranger" (might use simpler terms or include "excuse me")
        # 
        # Format your response as follows:
        prompt = f"""
        You are an advanced AI language model specialized in cultural and linguistic translation. Your primary function is to translate communicative intent across languages in a natural and culturally appropriate manner. 
        
        Your task is to translate the user's INTENT into {target_language} at the specified level of formality ({formality_level}). The CONTEXT, if provided, should inform how you translate the intent.
        
        Guidelines:
        1. Focus on translating the INTENT, not the CONTEXT.
        2. Use the CONTEXT to determine appropriate phrasing, formality adjustments, and cultural adaptations.
        3. Create a natural phrase that a native speaker of the target language would use in the given situation.
        4. Consider social dynamics, speaker/listener relationships, and situational appropriateness.
        5. Do not translate word-for-word; instead, focus on conveying the communicative goal naturally and effectively.
        6. Adjust your translation to match the specified formality level.
        7. Be aware of cultural nuances and adapt the translation accordingly.
        
        Process:
        1. Analyze the provided INTENT and CONTEXT (if available).
        2. Consider the target language, formality level, and cultural aspects.
        3. Formulate a translation that captures the communicative intent while adhering to the guidelines above.
        4. Review your translation for naturalness and cultural appropriateness.
        
        Before providing your final translation, wrap your analysis inside <cultural_linguistic_analysis> tags. In this analysis:
        1. List key words or phrases from the INTENT that require special attention or direct translation.
        2. Explicitly consider cultural nuances, idioms, and potential misunderstandings.
        3. Propose multiple translation options, considering their pros and cons.
        4. Explain your reasoning for choosing the final translation.
        
        It's OK for this section to be quite long.
        
        Then, present your final translation within <translation> tags.
        
        Example structure (do not use this content, it's just to illustrate the format):
        
        <cultural_linguistic_analysis>
        1. Key phrases: "Where is the bathroom?"
        2. Cultural considerations: In many [target language] cultures, asking about toilets directly can be considered impolite in formal settings.
        3. Translation options:
           a. [Formal, indirect phrase in target language] (Literal: May I ask where the restroom is located?)
           b. [Less formal, more direct phrase in target language] (Literal: Where is the toilet?)
           c. [Highly formal, very indirect phrase in target language] (Literal: Could you kindly direct me to the facilities?)
        4. Chosen translation: Option (a) because it balances formality with clarity, avoiding potential embarrassment while still being understood.
        </cultural_linguistic_analysis>
        
        <translation>
        [Your culturally appropriate, formal translation of "Where is the bathroom?" in the target language]
        </translation>
        
        Please proceed with your analysis and translation of the provided INTENT, considering the CONTEXT if available.
        """
        
        prompt += f"\nTRANSLATION:\n[The translated intent in {target_language}]"
        
        if include_pronunciation:
            prompt += "\n\nPRONUNCIATION:\n[Simple pronunciation guide for non-native speakers]"
            
        if include_cultural_notes:
            prompt += "\n\nCULTURAL NOTES:\n[Brief cultural context or usage notes relevant to this phrase]"
            
        return prompt
        
    def _parse_translation_response(
        self, 
        response_text: str,
        include_pronunciation: bool,
        include_cultural_notes: bool
    ) -> Dict[str, str]:
        """
        Parse the response from Claude into structured data.
        
        Args:
            response_text: Raw text response from Claude
            include_pronunciation: Whether pronunciation was requested
            include_cultural_notes: Whether cultural notes were requested
            
        Returns:
            Dictionary containing parsed translation components
        """
        result = {
            "translation": "",
            "pronunciation": "",
            "cultural_notes": ""
        }
        
        # Simple parsing logic - handles section headers followed by content
        lines = response_text.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if "TRANSLATION:" in line.upper():
                current_section = "translation"
                # Skip this header line
                continue
            elif "PRONUNCIATION:" in line.upper():
                current_section = "pronunciation"
                continue
            elif "CULTURAL NOTES:" in line.upper():
                current_section = "cultural_notes"
                continue
            
            # If we have a current section and the line is not empty, add to that section
            if current_section and line:
                if result[current_section]:
                    result[current_section] += "\n" + line
                else:
                    result[current_section] = line
        
        # If parsing fails, just use the whole response as the translation
        if not result["translation"]:
            result["translation"] = response_text.strip()
            
        return result