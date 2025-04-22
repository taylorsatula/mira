"""
Tests for the synthetic example generation in the tool relevance engine.
"""
import os
import shutil
import unittest
import tempfile
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tool_relevance_engine import ToolRelevanceEngine


class TestSyntheticExamples(unittest.TestCase):
    """Test cases for synthetic example generation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock tool repository
        self.mock_tool_repo = MagicMock()
        self.mock_tool_repo.list_all_tools.return_value = ["sample_tool", "test_tool"]
        
        # Create a mock config
        self.mock_config = {
            "paths": {
                "data_dir": os.path.join(self.temp_dir, "data"),
                "tools_dir": os.path.join(self.temp_dir, "tools"),
            },
            "tool_relevance": {
                "thread_limit": 2,
                "context_window_size": 3,
                "topic_coherence_threshold": 0.7,
                "tool_persistence_messages": 5,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        }
        
        # Create necessary directories
        os.makedirs(os.path.join(self.temp_dir, "data", "tools", "sample_tool"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "tools"), exist_ok=True)
        
        # Create a sample tool file
        with open(os.path.join(self.temp_dir, "tools", "sample_tool.py"), "w") as f:
            f.write('"""Sample tool for testing."""\n')
            f.write('class SampleTool:\n')
            f.write('    """A sample tool for testing the synthetic example generator."""\n')
            f.write('    name = "sample_tool"\n')
            f.write('    description = "A tool for testing"\n')
            f.write('    \n')
            f.write('    def run(self, param1: str):\n')
            f.write('        """Run the tool."""\n')
            f.write('        return {"result": param1}\n')
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    @patch("tool_relevance_engine.config")
    @patch("tool_relevance_engine.MultiLabelClassifier")
    def test_load_tool_examples_with_existing_examples(self, mock_classifier, mock_config):
        """Test loading examples when classifier_examples.json exists."""
        # Configure mocks
        mock_config.configure_mock(**self.mock_config)
        mock_classifier_instance = MagicMock()
        mock_classifier.return_value = mock_classifier_instance
        
        # Create example file
        examples_dir = os.path.join(self.temp_dir, "data", "tools", "sample_tool")
        os.makedirs(examples_dir, exist_ok=True)
        
        with open(os.path.join(examples_dir, "classifier_examples.json"), "w") as f:
            f.write('''[
                {"tool_name": "sample_tool", "query": "Test query 1"},
                {"tool_name": "sample_tool", "query": "Test query 2"}
            ]''')
        
        # Create the tool relevance engine
        engine = ToolRelevanceEngine(self.mock_tool_repo)
        
        # Check that examples were loaded
        self.assertIn("sample_tool", engine.tool_examples)
        self.assertEqual(len(engine.tool_examples["sample_tool"]["examples"]), 2)
        self.assertFalse(engine.tool_examples["sample_tool"]["is_autogen"])
    
    @patch("tool_relevance_engine.config")
    @patch("tool_relevance_engine.MultiLabelClassifier")
    def test_load_tool_examples_with_autogen_examples(self, mock_classifier, mock_config):
        """Test loading examples when only autogen_classifier_examples.json exists."""
        # Configure mocks
        mock_config.configure_mock(**self.mock_config)
        mock_classifier_instance = MagicMock()
        mock_classifier.return_value = mock_classifier_instance
        
        # Create example file
        examples_dir = os.path.join(self.temp_dir, "data", "tools", "sample_tool")
        os.makedirs(examples_dir, exist_ok=True)
        
        with open(os.path.join(examples_dir, "autogen_classifier_examples.json"), "w") as f:
            f.write('''[
                {"tool_name": "sample_tool", "query": "Auto test query 1"},
                {"tool_name": "sample_tool", "query": "Auto test query 2"}
            ]''')
        
        # Create the tool relevance engine
        engine = ToolRelevanceEngine(self.mock_tool_repo)
        
        # Check that examples were loaded
        self.assertIn("sample_tool", engine.tool_examples)
        self.assertEqual(len(engine.tool_examples["sample_tool"]["examples"]), 2)
        self.assertTrue(engine.tool_examples["sample_tool"]["is_autogen"])
    
    @patch("tool_relevance_engine.config")
    @patch("tool_relevance_engine.MultiLabelClassifier")
    @patch("tool_relevance_engine.ToolRelevanceEngine._generate_synthetic_examples")
    def test_load_tool_examples_with_no_examples(self, mock_generate, mock_classifier, mock_config):
        """Test that synthetic examples are generated when no examples exist."""
        # Configure mocks
        mock_config.configure_mock(**self.mock_config)
        mock_classifier_instance = MagicMock()
        mock_classifier.return_value = mock_classifier_instance
        
        def side_effect(tool_names):
            """Create synthetic examples when _generate_synthetic_examples is called."""
            self.assertEqual(tool_names, ["sample_tool"])
            examples_dir = os.path.join(self.temp_dir, "data", "tools", "sample_tool")
            os.makedirs(examples_dir, exist_ok=True)
            with open(os.path.join(examples_dir, "autogen_classifier_examples.json"), "w") as f:
                f.write('''[
                    {"tool_name": "sample_tool", "query": "Generated query 1"},
                    {"tool_name": "sample_tool", "query": "Generated query 2"}
                ]''')
        
        mock_generate.side_effect = side_effect
        
        # Create the tool relevance engine
        engine = ToolRelevanceEngine(self.mock_tool_repo)
        
        # Check that examples were generated and loaded
        mock_generate.assert_called_once_with(["sample_tool"])
        self.assertIn("sample_tool", engine.tool_examples)
        self.assertEqual(len(engine.tool_examples["sample_tool"]["examples"]), 2)
        self.assertTrue(engine.tool_examples["sample_tool"]["is_autogen"])


if __name__ == "__main__":
    unittest.main()