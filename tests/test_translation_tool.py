import unittest
from unittest.mock import MagicMock, patch

from tools.translation_tool import TranslationTool
from api.llm_bridge import LLMBridge
from errors import ToolError


class TestTranslationTool(unittest.TestCase):
    """Tests for the Translation Tool."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock LLMBridge
        self.mock_llm_bridge = MagicMock(spec=LLMBridge)
        
        # Create instance of tool with mock bridge
        self.tool = TranslationTool(llm_bridge=self.mock_llm_bridge)
        
    def test_init(self):
        """Test tool initialization."""
        self.assertEqual(self.tool.name, "translation_tool")
        self.assertEqual(self.tool.sonnet_model, "claude-3-5-sonnet-20240620")
        self.assertEqual(len(self.tool.formality_levels), 4)
        
    def test_format_user_message_with_context(self):
        """Test formatting of user messages with context."""
        result = self.tool._format_user_message(
            intent="where is the bathroom",
            context="I'm at a formal business dinner"
        )
        self.assertIn("INTENT: where is the bathroom", result)
        self.assertIn("CONTEXT: I'm at a formal business dinner", result)
        
    def test_format_user_message_without_context(self):
        """Test formatting of user messages without context."""
        result = self.tool._format_user_message(
            intent="where is the bathroom"
        )
        self.assertEqual(result, "INTENT: where is the bathroom")
        self.assertNotIn("CONTEXT", result)
        
    def test_create_system_prompt(self):
        """Test system prompt creation with various options."""
        # Basic prompt
        basic_prompt = self.tool._create_system_prompt(
            target_language="Spanish",
            formality_level="formal",
            include_pronunciation=False,
            include_cultural_notes=False
        )
        self.assertIn("translate the user's INTENT into Spanish", basic_prompt)
        self.assertIn("at a formal level of formality", basic_prompt)
        self.assertIn("TRANSLATION:", basic_prompt)
        self.assertNotIn("PRONUNCIATION:", basic_prompt)
        self.assertNotIn("CULTURAL NOTES:", basic_prompt)
        
        # Prompt with all options
        full_prompt = self.tool._create_system_prompt(
            target_language="Spanish",
            formality_level="formal",
            include_pronunciation=True,
            include_cultural_notes=True
        )
        self.assertIn("TRANSLATION:", full_prompt)
        self.assertIn("PRONUNCIATION:", full_prompt)
        self.assertIn("CULTURAL NOTES:", full_prompt)
        
    def test_parse_translation_response_full(self):
        """Test parsing a complete response with all sections."""
        mock_response = """
        TRANSLATION:
        ¿Dónde está el baño?
        
        PRONUNCIATION:
        DON-day es-TA el BAN-yo
        
        CULTURAL NOTES:
        This is a formal way to ask. In casual settings with friends, you might hear "¿Dónde está el baño?" or simply "¿El baño?"
        """
        
        result = self.tool._parse_translation_response(
            mock_response,
            include_pronunciation=True,
            include_cultural_notes=True
        )
        
        self.assertEqual(result["translation"], "¿Dónde está el baño?")
        self.assertIn("DON-day es-TA el BAN-yo", result["pronunciation"])
        self.assertIn("formal way to ask", result["cultural_notes"])
        
    def test_parse_translation_response_translation_only(self):
        """Test parsing a response with only translation."""
        mock_response = """
        TRANSLATION:
        ¿Dónde está el baño?
        """
        
        result = self.tool._parse_translation_response(
            mock_response,
            include_pronunciation=False,
            include_cultural_notes=False
        )
        
        self.assertEqual(result["translation"], "¿Dónde está el baño?")
        self.assertEqual(result["pronunciation"], "")
        self.assertEqual(result["cultural_notes"], "")
        
    def test_parse_translation_response_unformatted(self):
        """Test parsing an unformatted response."""
        mock_response = "¿Dónde está el baño?"
        
        result = self.tool._parse_translation_response(
            mock_response,
            include_pronunciation=False,
            include_cultural_notes=False
        )
        
        self.assertEqual(result["translation"], "¿Dónde está el baño?")
    
    @patch.object(TranslationTool, '_parse_translation_response')
    def test_run_successful(self, mock_parse):
        """Test successful execution of the tool."""
        # Configure mocks
        mock_response = MagicMock()
        self.mock_llm_bridge.generate_response.return_value = mock_response
        self.mock_llm_bridge.extract_text_content.return_value = "TRANSLATION: ¿Dónde está el baño?"
        
        mock_parse.return_value = {
            "translation": "¿Dónde está el baño?",
            "pronunciation": "DON-day es-TA el BAN-yo",
            "cultural_notes": "This is a formal way to ask."
        }
        
        # Run the tool
        result = self.tool.run(
            intent="where is the bathroom",
            target_language="Spanish",
            context="I am at a formal business dinner",
            formality_level="formal",
            include_pronunciation=True,
            include_cultural_notes=True
        )
        
        # Verify the LLM bridge was called with correct parameters
        self.mock_llm_bridge.generate_response.assert_called_once()
        call_args = self.mock_llm_bridge.generate_response.call_args[1]
        
        self.assertEqual(call_args["model"], "claude-3-5-sonnet-20240620")
        self.assertEqual(call_args["temperature"], 0.3)
        self.assertIn("system_prompt", call_args)
        
        # Check the messages include intent and context
        messages = call_args["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIn("INTENT: where is the bathroom", messages[0]["content"])
        self.assertIn("CONTEXT: I am at a formal business dinner", messages[0]["content"])
        
        # Check the result
        self.assertEqual(result["translation"], "¿Dónde está el baño?")
        self.assertEqual(result["pronunciation"], "DON-day es-TA el BAN-yo")
        self.assertEqual(result["cultural_notes"], "This is a formal way to ask.")
        self.assertEqual(result["source_intent"], "where is the bathroom")
        self.assertEqual(result["context"], "I am at a formal business dinner")
        self.assertEqual(result["target_language"], "Spanish")
        self.assertEqual(result["model_used"], "claude-3-5-sonnet-20240620")
        
    def test_run_invalid_intent(self):
        """Test validation of intent parameter."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                intent="",  # Empty intent
                target_language="Spanish"
            )
        self.assertEqual(context.exception.code.name, "TOOL_INVALID_INPUT")
        
    def test_run_invalid_target_language(self):
        """Test validation of target_language parameter."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                intent="where is the bathroom",
                target_language=""  # Empty target language
            )
        self.assertEqual(context.exception.code.name, "TOOL_INVALID_INPUT")
        
    def test_run_invalid_formality(self):
        """Test validation of formality_level parameter."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                intent="where is the bathroom",
                target_language="Spanish",
                formality_level="super-formal"  # Invalid formality level
            )
        self.assertEqual(context.exception.code.name, "TOOL_INVALID_INPUT")
        

if __name__ == '__main__':
    unittest.main()