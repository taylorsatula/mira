"""
Workflow manager for handling multi-step workflows.

This module provides a workflow management system that can detect when user
input matches a workflow, track progress through workflow steps, and integrate
with the conversation system.
"""

import datetime
import hashlib
import json
import logging
import os
import re
from glob import glob
from typing import Dict, List, Any, Optional, Tuple, Set, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from tools.repo import ToolRepository
from errors import ErrorCode, error_context, ToolError
from config import config
from serialization import to_json, from_json
from utils.tag_parser import parser as tag_parser


class WorkflowManager:
    """
    Manages detection and execution of predefined workflows.
    
    This class handles loading workflow definitions, detecting when user input
    matches a workflow, tracking progress through workflow steps, and integrating
    with the conversation system via the system prompt.
    """
    
    def __init__(
        self, 
        tool_repo: ToolRepository,
        model,
        workflows_dir: Optional[str] = None
    ):
        """
        Initialize the workflow manager.
        
        Args:
            tool_repo: Repository of available tools
            workflows_dir: Directory containing workflow definition files
            model: Pre-loaded SentenceTransformer model to use for embedding computations
        """
        self.logger = logging.getLogger("workflow_manager")
        self.tool_repo = tool_repo
        
        # Set workflows directory
        self.workflows_dir = workflows_dir
        if self.workflows_dir is None:
            # Use default directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.workflows_dir = current_dir
        
        # Initialize empty workflows dictionary
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
        # Initialize file hashes for change detection
        self.file_hashes: Dict[str, str] = {}
        
        # Get configuration values
        self.embedding_model = config.tool_relevance.embedding_model
        self.match_threshold = 0.65  # Default, can be overridden from config #ANNOTATION <- This should be a config only value
        
        # State for the current workflow (if any)
        self.active_workflow_id: Optional[str] = None
        self.active_step_index: int = 0
        
        # Cache for workflow example embeddings
        self.workflow_embeddings: Dict[str, Dict[str, Any]] = {}
        
        # Use the provided model
        self.model = model
        self.logger.info("Using provided SentenceTransformer model")
        
        # Load workflow definitions
        self.load_workflows()
        
        # Compute embeddings for all workflows
        self._compute_workflow_embeddings()
    
    def load_workflows(self) -> None:
        """
        Load workflow definitions from JSON files.
        
        This method discovers and loads all workflow definition files in the
        workflows directory, checking file hashes to avoid unnecessarily
        reloading unchanged files. 
        """
        self.logger.info(f"Loading workflows from {self.workflows_dir}")
        
        # Find all JSON files in the workflows directory
        workflow_files = glob(os.path.join(self.workflows_dir, "*.json"))
        
        # Check which files have changed
        current_hashes = {}
        changed_files = []
        
        for file_path in workflow_files:
            try:
                file_hash = self._calculate_file_hash(file_path)
                current_hashes[file_path] = file_hash
                
                # Check if file has changed
                if file_path not in self.file_hashes or self.file_hashes[file_path] != file_hash:
                    changed_files.append(file_path)
                    self.logger.debug(f"Workflow file changed or new: {file_path}")
            except Exception as e:
                self.logger.error(f"Error checking workflow file hash {file_path}: {e}")
        
        # Check for deleted files
        for file_path in list(self.file_hashes.keys()):
            if file_path not in current_hashes:
                workflow_id = os.path.splitext(os.path.basename(file_path))[0]
                if workflow_id in self.workflows:
                    del self.workflows[workflow_id]
                    self.logger.info(f"Removed deleted workflow: {workflow_id}")
        
        # Update file hashes
        self.file_hashes = current_hashes
        
        # Load changed files
        for file_path in changed_files:
            try:
                with open(file_path, 'r') as f:
                    workflow = json.load(f)
                
                # Validate the workflow definition
                if self._validate_workflow(workflow):
                    workflow_id = workflow["id"]
                    self.workflows[workflow_id] = workflow
                    self.logger.info(f"Loaded workflow: {workflow_id} ({workflow['name']})")
            except Exception as e:
                self.logger.error(f"Error loading workflow file {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(self.workflows)} workflows")
    
    def _calculate_file_hash(self, file_path: str) -> str: #ANNOTATION We should create one global file hash checker function and use it throughout the codebase instead of each python file handling its own hashing function.
        """
        Calculate a hash of the file contents.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            SHA-256 hash of the file contents
        """
        hash_obj = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        """
        Validate a workflow definition.
        
        Args:
            workflow: Workflow definition to validate
            
        Returns:
            True if the workflow is valid, False otherwise
        """
        # Check required fields
        required_fields = ["id", "name", "description", "trigger_examples", "steps"]
        for field in required_fields:
            if field not in workflow:
                self.logger.error(f"Workflow missing required field: {field}")
                return False
        
        # Check that steps is a non-empty list
        if not isinstance(workflow["steps"], list) or not workflow["steps"]:
            self.logger.error("Workflow steps must be a non-empty list")
            return False
        
        # Check that trigger_examples is a non-empty list
        if not isinstance(workflow["trigger_examples"], list) or not workflow["trigger_examples"]:
            self.logger.error("Workflow trigger_examples must be a non-empty list")
            return False
        
        # Check that each step has required fields
        for i, step in enumerate(workflow["steps"]):
            if not isinstance(step, dict):
                self.logger.error(f"Step {i} is not a dictionary")
                return False
            
            for field in ["id", "description", "tools", "guidance"]:
                if field not in step:
                    self.logger.error(f"Step {i} missing required field: {field}")
                    return False
            
            # Check that tools is a list
            if not isinstance(step["tools"], list):
                self.logger.error(f"Step {i} tools must be a list")
                return False
        
        return True
    
    def _compute_workflow_embeddings(self) -> None:
        """
        Compute embeddings for all workflow trigger examples.
        
        This precomputes embeddings for all workflow trigger examples for more
        efficient matching.
        """
        if not self.model:
            self.logger.error("Cannot compute embeddings: model not loaded")
            return
        
        self.logger.info("Computing embeddings for workflow trigger examples")
        
        for workflow_id, workflow in self.workflows.items():
            examples = workflow["trigger_examples"]
            
            try:
                # Compute embeddings for all examples
                embeddings = self.model.encode(examples)
                
                # Store embeddings
                self.workflow_embeddings[workflow_id] = {
                    "examples": examples,
                    "embeddings": embeddings
                }
                
                self.logger.debug(f"Computed embeddings for workflow {workflow_id}: {len(examples)} examples")
            except Exception as e:
                self.logger.error(f"Error computing embeddings for workflow {workflow_id}: {e}")
        
        self.logger.info(f"Computed embeddings for {len(self.workflow_embeddings)} workflows")
    
    def detect_workflow(self, message: str) -> Tuple[Optional[str], float]:
        """
        Detect if a message matches a workflow.
        
        Args:
            message: User message to analyze
            
        Returns:
            Tuple of (workflow_id, confidence) if a match is found, (None, 0.0) otherwise
        """
        if not self.model or not self.workflows or not self.workflow_embeddings:
            return None, 0.0
        
        with error_context(
            component_name="WorkflowManager",
            operation="detecting workflow",
            error_class=ToolError,
            logger=self.logger
        ):
            # Convert message to lowercase for better matching
            message_lower = message.lower()
            
            # Get embedding for the message
            message_embedding = self.model.encode(message_lower)
            
            best_match_id = None
            best_match_score = 0.0
            
            # Check each workflow
            for workflow_id, embedding_data in self.workflow_embeddings.items():
                example_embeddings = embedding_data["embeddings"]
                
                # Calculate similarity scores
                # Ensure embeddings are normalized for proper cosine similarity
                message_norm = np.linalg.norm(message_embedding) #ANNOTATION explain this normalizing process to me and tell me why it is necessary.
                if message_norm > 0:
                    message_embedding_norm = message_embedding / message_norm
                else:
                    message_embedding_norm = message_embedding
                
                # Calculate similarity with each example
                scores = []
                for example_embedding in example_embeddings:
                    example_norm = np.linalg.norm(example_embedding)
                    if example_norm > 0:
                        example_embedding_norm = example_embedding / example_norm
                    else:
                        example_embedding_norm = example_embedding
                    
                    similarity = np.dot(message_embedding_norm, example_embedding_norm)
                    scores.append(float(similarity))
                
                # Get best score for this workflow
                if scores:
                    best_score = max(scores)
                    
                    # Update best match if this is better
                    if best_score > best_match_score:
                        best_match_score = best_score
                        best_match_id = workflow_id
            
            # Check if the best match exceeds the threshold
            if best_match_score >= self.match_threshold:
                self.logger.info(f"Detected workflow: {best_match_id} (confidence: {best_match_score:.4f})")
                return best_match_id, best_match_score
            else:
                self.logger.debug(f"No workflow detected (best confidence: {best_match_score:.4f})")
                return None, 0.0
    
    def start_workflow(self, workflow_id: str) -> Dict[str, Any]: #ANNOTATION Recently I have experienced an issue where MIRA (incorrectly) guesses on what the name of the workflow. We should either improve the prompt that gives direction on how to pick a workflow ID ~or~ build in a programmatic matcher. I think option 1 would be better.
        """
        Start a workflow.
        
        Args:
            workflow_id: ID of the workflow to start
            
        Returns:
            Dictionary containing workflow information
            
        Raises:
            ValueError: If the workflow doesn't exist
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow with ID '{workflow_id}' doesn't exist")
        
        workflow = self.workflows[workflow_id]
        
        # Disable all currently enabled tools to start fresh
        currently_enabled_tools = self.tool_repo.get_enabled_tools()
        for tool_name in currently_enabled_tools:
            try:
                self.tool_repo.disable_tool(tool_name)
                self.logger.info(f"Disabled previously enabled tool: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error disabling tool {tool_name}: {e}")
        
        # Activate the workflow
        self.active_workflow_id = workflow_id
        self.active_step_index = 0
        
        # Enable the tools for the first step
        self._enable_tools_for_current_step()
        
        self.logger.info(f"Started workflow: {workflow_id}")
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "active_step": workflow["steps"][self.active_step_index],
        }
    
    def advance_workflow(self) -> Dict[str, Any]:
        """
        Advance the workflow to the next step.
        
        Returns:
            Dictionary containing workflow information
            
        Raises:
            ValueError: If no workflow is active
        """
        if not self.active_workflow_id:
            raise ValueError("No active workflow! How the h e c k did we get here?")
        
        workflow = self.workflows[self.active_workflow_id]
        
        # Advance to the next step
        self.active_step_index += 1
        
        # Check if this was the last step
        if self.active_step_index >= len(workflow["steps"]):
            # Workflow is complete
            result = self.complete_workflow()
            return result
        
        # Enable the tools for the next step
        self._enable_tools_for_current_step()
        
        self.logger.info(f"Advanced workflow {self.active_workflow_id} to step {self.active_step_index}")
        
        return {
            "workflow_id": self.active_workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "active_step": workflow["steps"][self.active_step_index],
        }
    
    def complete_workflow(self) -> Dict[str, Any]:
        """
        Complete the active workflow.
        
        Returns:
            Dictionary containing workflow information
            
        Raises:
            ValueError: If no workflow is active
        """
        if not self.active_workflow_id:
            raise ValueError("No active workflow")
        
        workflow = self.workflows[self.active_workflow_id]
        
        # Store the completed workflow info before clearing active state
        completed_workflow = {
            "workflow_id": self.active_workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "status": "completed"
        }
        
        # Disable all workflow tools before clearing workflow state #ANNOTATION 
        currently_enabled_tools = self.tool_repo.get_enabled_tools()
        for tool_name in currently_enabled_tools:
            try:
                self.tool_repo.disable_tool(tool_name)
                self.logger.info(f"Disabled workflow tool on completion: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error disabling tool {tool_name}: {e}")
        
        # Clear the active workflow
        self.active_workflow_id = None
        self.active_step_index = 0
        
        self.logger.info(f"Completed workflow: {completed_workflow['workflow_id']}")
        
        return completed_workflow
    
    def cancel_workflow(self) -> Dict[str, Any]:
        """
        Cancel the active workflow.
        
        Returns:
            Dictionary containing workflow information
            
        Raises:
            ValueError: If no workflow is active
        """
        if not self.active_workflow_id:
            raise ValueError("No active workflow")
        
        workflow = self.workflows[self.active_workflow_id]
        
        # Store the cancelled workflow info before clearing active state
        cancelled_workflow = {
            "workflow_id": self.active_workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "status": "cancelled"
        }
        
        # Disable all workflow tools before clearing workflow state #ANNOTATION I worry that this could cause the next message after a canceled workflow to have access to no tools. Please investigate if the workflow tool wipe happens BEFORE or AFTER the message analysis that enables tools in normal conversation.
        currently_enabled_tools = self.tool_repo.get_enabled_tools()
        for tool_name in currently_enabled_tools:
            try:
                self.tool_repo.disable_tool(tool_name)
                self.logger.info(f"Disabled workflow tool on cancellation: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error disabling tool {tool_name}: {e}")
        
        # Clear the active workflow
        self.active_workflow_id = None
        self.active_step_index = 0
        
        self.logger.info(f"Cancelled workflow: {cancelled_workflow['workflow_id']}")
        
        return cancelled_workflow
    
    def get_active_workflow(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the active workflow.
        
        Returns:
            Dictionary containing workflow information, or None if no workflow is active
        """
        if not self.active_workflow_id:
            return None
        
        workflow = self.workflows[self.active_workflow_id]
        
        return {
            "workflow_id": self.active_workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "active_step_index": self.active_step_index,
            "active_step": workflow["steps"][self.active_step_index],
            "total_steps": len(workflow["steps"])
        }
    
    def _enable_tools_for_current_step(self) -> None:
        """
        Enable the tools required for the current workflow step and disable tools
        that are not needed for this step.
        """
        if not self.active_workflow_id:
            return
        
        workflow = self.workflows[self.active_workflow_id]
        
        # Check if the step index is valid
        if self.active_step_index >= len(workflow["steps"]):
            self.logger.error(f"Invalid step index: {self.active_step_index}")
            return
        
        # Get the current step
        step = workflow["steps"][self.active_step_index]
        
        # Get the tools for this step
        tools_for_current_step = step.get("tools", [])
        
        # Get list of currently enabled tools
        currently_enabled_tools = self.tool_repo.get_enabled_tools()
        
        # First, disable tools that are not needed for this step
        for tool_name in currently_enabled_tools:
            if tool_name not in tools_for_current_step:
                try:
                    self.tool_repo.disable_tool(tool_name)
                    self.logger.info(f"Disabled tool not needed for current workflow step: {tool_name}")
                except Exception as e:
                    self.logger.error(f"Error disabling tool {tool_name}: {e}")
        
        # Then, enable each tool needed for this step #ANNOTATION does this attempt to reload tools that are already in tools_for_current_step?
        for tool_name in tools_for_current_step:
            try:
                if not self.tool_repo.is_tool_enabled(tool_name):
                    self.tool_repo.enable_tool(tool_name)
                    self.logger.info(f"Enabled tool for workflow step: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error enabling tool {tool_name}: {e}")
    
    def get_system_prompt_extension(self) -> str:
        """
        Get a system prompt extension for an active workflow.
        
        This method generates text to be appended to the system prompt that
        informs the LLM about the active workflow and provides guidance for
        the current step, including a visual checklist.
        
        Returns:
            Text to append to the system prompt, or empty string if no workflow is active
        """
        if not self.active_workflow_id:
            return ""
        
        workflow = self.workflows[self.active_workflow_id]
        
        # Check if the step index is valid
        if self.active_step_index >= len(workflow["steps"]):
            return ""
        
        # Get the current step
        current_step = workflow["steps"][self.active_step_index]
        
        # Format the checklist
        checklist_items = []
        for i, step in enumerate(workflow["steps"]):
            if i < self.active_step_index:
                # Completed step
                status_marker = "[✅]"
            elif i == self.active_step_index:
                # Current step
                status_marker = "[🔄]"
            else:
                # Future step
                status_marker = "[ ]"
            
            checklist_items.append(f"{status_marker} {step['description']}")
        
        checklist_text = "\n".join(checklist_items)
        
        # Build the prompt
        prompt = [
            "\n\n# ACTIVE WORKFLOW GUIDANCE",
            f"You are currently helping the user with: **{workflow['name']}**",
            f"Description: {workflow['description']}",
            "",
            f"## Current Step: {current_step['description']}",
            f"Guidance: {current_step['guidance']}",
            "",
            "As you assist the user:",
            "1. Focus on completing the current step before moving to the next",
            "2. When you believe the current step is complete, explicitly mark it as complete by including this exact text in your response:",
            "<workflow_complete />",
            "3. If the user gets sidetracked, gently guide them back to the workflow when appropriate",
            "3b. To cancel the workflow at any time, include this exact text:",
            "<workflow_cancel />",
            "",
            "## Workflow Checklist:",
            checklist_text
        ]
        
        return "\n".join(prompt)
    
    def check_for_workflow_commands(self, message: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check an assistant message for workflow commands.
        
        Args:
            message: Assistant message to check
            
        Returns:
            Tuple of (command_found, command_type, command_params)
            command_found: True if a command was found, False otherwise
            command_type: "start", "complete", or "cancel" if a command was found, None otherwise
            command_params: Additional parameters for the command (e.g., workflow_id for "start"), None if not applicable
        """
        # Use the centralized tag parser
        workflow_action = tag_parser.get_workflow_action(message)
        
        if workflow_action and workflow_action["action"]:
            action = workflow_action["action"]
            
            if action == "start":
                workflow_id = workflow_action["id"]
                if workflow_id:
                    return True, "start", workflow_id
            elif action == "complete":
                return True, "complete", None
            elif action == "cancel":
                return True, "cancel", None
        
        return False, None, None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workflow manager state to a dictionary for serialization.
        
        Returns:
            Dictionary containing workflow manager state
        """
        return {
            "active_workflow_id": self.active_workflow_id,
            "active_step_index": self.active_step_index
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load workflow manager state from a dictionary.
        
        Args:
            data: Dictionary containing workflow manager state
        """
        self.active_workflow_id = data.get("active_workflow_id")
        self.active_step_index = data.get("active_step_index", 0)
        
        self.logger.info(f"Loaded workflow manager state: active workflow: {self.active_workflow_id}")
        
        # If there's an active workflow, enable the tools for the current step
        if self.active_workflow_id:
            self._enable_tools_for_current_step()
    
    def to_json(self) -> str:
        """
        Convert workflow manager state to a JSON string.
        
        Returns:
            JSON string containing workflow manager state
        """
        return to_json(self.to_dict())
    
    def from_json(self, json_str: str) -> None:
        """
        Load workflow manager state from a JSON string.
        
        Args:
            json_str: JSON string containing workflow manager state
        """
        data = from_json(json_str)
        self.from_dict(data)