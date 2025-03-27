import os
import json
import unittest
from unittest.mock import patch, mock_open, MagicMock

from tools.questionnaire_tool import QuestionnaireTool
from errors import ToolError


class TestQuestionnaireTool(unittest.TestCase):
    """Tests for the QuestionnaireTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool = QuestionnaireTool()
        
        # Sample questionnaire data
        self.sample_questions = [
            {
                "id": "q1",
                "text": "Test question 1?",
                "options": ["Option A", "Option B"],
                "preference_key": "Question One"
            },
            {
                "id": "q2",
                "text": "Test question 2?",
                "allow_free_text": True,
            }
        ]
        
        # Sample questionnaire file content
        self.sample_json = json.dumps(self.sample_questions)

    @patch('builtins.print')
    @patch('builtins.input')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_with_predefined_questionnaire(self, mock_file, mock_exists, mock_input, mock_print):
        """Test running a questionnaire from a predefined file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.return_value.__enter__.return_value.read.return_value = self.sample_json
        mock_input.side_effect = ["1", "Free text answer"]
        
        # Run tool
        result = self.tool.run("test_questionnaire")
        
        # Verify results
        self.assertEqual(result["questionnaire_id"], "test_questionnaire")
        self.assertTrue(result["completed"])
        self.assertEqual(result["responses"]["q1"], "Option A")
        self.assertEqual(result["responses"]["q2"], "Free text answer")
        
        # Verify the file was opened with the correct path
        mock_file.assert_called_once()
        file_path = mock_file.call_args[0][0]
        self.assertTrue(file_path.endswith("test_questionnaire.json"))

    @patch('builtins.print')
    @patch('builtins.input')
    def test_run_with_custom_questions(self, mock_input, mock_print):
        """Test running a questionnaire with custom questions."""
        # Setup mock inputs
        mock_input.side_effect = ["Option B", "Custom answer"]
        
        # Run tool with custom questions
        result = self.tool.run(
            questionnaire_id="custom", 
            custom_questions=self.sample_questions
        )
        
        # Verify results
        self.assertEqual(result["questionnaire_id"], "custom")
        self.assertTrue(result["completed"])
        self.assertEqual(result["responses"]["q1"], "Option B")
        self.assertEqual(result["responses"]["q2"], "Custom answer")

    @patch('os.path.exists')
    def test_nonexistent_questionnaire(self, mock_exists):
        """Test handling of a nonexistent questionnaire file."""
        # Setup mock
        mock_exists.return_value = False
        
        # Verify exception is raised
        with self.assertRaises(ToolError) as context:
            self.tool.run("nonexistent")
            
        # Check error details
        self.assertEqual(context.exception.error_code.name, "TOOL_INVALID_INPUT")
        self.assertIn("not found", str(context.exception))

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_invalid_json_format(self, mock_file, mock_exists):
        """Test handling of invalid JSON in questionnaire file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.return_value.__enter__.return_value.read.return_value = "{invalid json"
        
        # Verify exception is raised
        with self.assertRaises(ToolError) as context:
            self.tool.run("invalid_json")
            
        # Check error details
        self.assertEqual(context.exception.error_code.name, "TOOL_INVALID_INPUT")
        self.assertIn("Invalid JSON", str(context.exception))

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_invalid_questionnaire_format(self, mock_file, mock_exists):
        """Test handling of valid JSON but invalid questionnaire format."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.return_value.__enter__.return_value.read.return_value = '{"not_a_list": true}'
        
        # Verify exception is raised
        with self.assertRaises(ToolError) as context:
            self.tool.run("invalid_format")
            
        # Check error details
        self.assertEqual(context.exception.error_code.name, "TOOL_INVALID_INPUT")
        self.assertIn("Invalid questionnaire format", str(context.exception))

    @patch('builtins.print')
    @patch('builtins.input')
    def test_preference_key_mapping(self, mock_input, mock_print):
        """Test that preference keys are correctly mapped in the response."""
        # Setup mock inputs
        mock_input.side_effect = ["1", "Some text"]
        
        # Create questions with preference_key mapping
        questions = [
            {
                "id": "q1",
                "text": "Question 1?",
                "options": ["Option A", "Option B"],
                "preference_key": "Mapped Key 1"
            },
            {
                "id": "q2",
                "text": "Question 2?",
                "allow_free_text": True,
                "preference_key": "Mapped Key 2"
            }
        ]
        
        # Run tool
        result = self.tool.run("custom_mapping", custom_questions=questions)
        
        # Verify mapped keys in response
        self.assertEqual(result["responses"]["Mapped Key 1"], "Option A")
        self.assertEqual(result["responses"]["Mapped Key 2"], "Some text")
        self.assertNotIn("q1", result["responses"])
        self.assertNotIn("q2", result["responses"])

    @patch('builtins.print')
    @patch('builtins.input')
    def test_numeric_option_selection(self, mock_input, mock_print):
        """Test selecting options by number."""
        # Setup mock inputs
        mock_input.side_effect = ["2"]  # Select option 2
        
        # Questions with only one multiple-choice question
        questions = [
            {
                "id": "test",
                "text": "Test question?",
                "options": ["Option A", "Option B", "Option C"]
            }
        ]
        
        # Run tool
        result = self.tool.run("numeric_test", custom_questions=questions)
        
        # Verify the second option was selected
        self.assertEqual(result["responses"]["test"], "Option B")

    @patch('builtins.print')
    @patch('builtins.input')
    def test_text_option_selection(self, mock_input, mock_print):
        """Test selecting options by text."""
        # Setup mock inputs
        mock_input.side_effect = ["Option C"]  # Select by name
        
        # Questions with only one multiple-choice question
        questions = [
            {
                "id": "test",
                "text": "Test question?",
                "options": ["Option A", "Option B", "Option C"]
            }
        ]
        
        # Run tool
        result = self.tool.run("text_test", custom_questions=questions)
        
        # Verify the option was selected by name
        self.assertEqual(result["responses"]["test"], "Option C")


if __name__ == '__main__':
    unittest.main()