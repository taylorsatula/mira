"""
Workflow manager for handling flexible, data-driven workflows.

This module provides a workflow management system that can detect when user
input matches a workflow, track progress through workflow steps in a non-linear fashion,
and integrate with the conversation system.
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

from tools.repo import ToolRepository
from errors import ErrorCode, error_context, ToolError
from config import config
from serialization import to_json, from_json
from utils.tag_parser import parser as tag_parser


class WorkflowManager:
    """
    Manages detection and execution of flexible, data-driven workflows.
    
    This class handles loading workflow definitions, detecting when user input
    matches a workflow, tracking state through workflow steps, and integrating
    with the conversation system via the system prompt.
    """
    
    def __init__(
        self,
        tool_repo: ToolRepository,
        model,
        workflows_dir: Optional[str] = None,
        llm_bridge = None,
        working_memory = None
    ):
        """
        Initialize the workflow manager.

        Args:
            tool_repo: Repository of available tools
            model: Pre-loaded ONNX embedding model to use for embedding computations
            workflows_dir: Directory containing workflow definition files
            llm_bridge: LLM bridge instance for dynamic operations
            working_memory: Working memory instance for storing workflow content
        """
        self.logger = logging.getLogger("workflow_manager")
        self.tool_repo = tool_repo

        # Store working memory reference
        self.working_memory = working_memory

        # Track detected but not yet active workflow ID
        self._detected_workflow_id = None

        # Track workflow hint ID for memory management
        self._workflow_hint_id = None

        # Content cache to avoid unnecessary updates to working memory
        self._content_cache = {
            "current_workflow": None,
            "header": None,
            "steps": {},  # step_id -> content
            "data": {},    # field_name -> content
            "checklist": None,
            "navigation": None
        }
        
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
        self.match_threshold = 0.65  # Default, can be overridden from config
        
        # State for the current workflow (if any)
        self.active_workflow_id: Optional[str] = None
        
        # New state tracking for flexible workflows
        self.completed_steps: Set[str] = set()
        self.available_steps: Set[str] = set()
        self.workflow_data: Dict[str, Any] = {}
        
        # Cache for workflow example embeddings
        self.workflow_embeddings: Dict[str, Dict[str, Any]] = {}
        
        # Use the provided model
        self.model = model
        self.logger.info("Using provided ONNX embedding model")
        
        # Store LLM bridge for dynamic operations
        self.llm_bridge = llm_bridge
        
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
    
    def _calculate_file_hash(self, file_path: str) -> str:
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
        Validate a workflow definition according to the new schema.
        
        Args:
            workflow: Workflow definition to validate
            
        Returns:
            True if the workflow is valid, False otherwise
        """
        # Check required fields
        required_fields = ["id", "name", "description", "trigger_examples", "steps", "completion_requirements"]
        for field in required_fields:
            if field not in workflow:
                self.logger.error(f"Workflow missing required field: {field}")
                return False
        
        # Check that steps is a dictionary
        if not isinstance(workflow["steps"], dict) or not workflow["steps"]:
            self.logger.error("Workflow steps must be a non-empty dictionary")
            return False
        
        # Check that trigger_examples is a non-empty list
        if not isinstance(workflow["trigger_examples"], list) or not workflow["trigger_examples"]:
            self.logger.error("Workflow trigger_examples must be a non-empty list")
            return False
        
        # Check completion requirements
        completion_requirements = workflow.get("completion_requirements", {})
        if not isinstance(completion_requirements, dict):
            self.logger.error("Workflow completion_requirements must be a dictionary")
            return False
        
        # Check that at least one of required_steps or required_data is present
        if "required_steps" not in completion_requirements and "required_data" not in completion_requirements:
            self.logger.error("Workflow completion_requirements must include required_steps or required_data")
            return False
        
        # Check that each step has required fields
        for step_id, step in workflow["steps"].items():
            if not isinstance(step, dict):
                self.logger.error(f"Step {step_id} is not a dictionary")
                return False
            
            for field in ["id", "description", "tools", "guidance", "prerequisites"]:
                if field not in step:
                    self.logger.error(f"Step {step_id} missing required field: {field}")
                    return False
            
            # Check that tools is a list
            if not isinstance(step["tools"], list):
                self.logger.error(f"Step {step_id} tools must be a list")
                return False
            
            # Check that prerequisites is a list
            if not isinstance(step["prerequisites"], list):
                self.logger.error(f"Step {step_id} prerequisites must be a list")
                return False
            
            # Check optional flag is boolean if present
            if "optional" in step and not isinstance(step["optional"], bool):
                self.logger.error(f"Step {step_id} optional flag must be a boolean")
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
    
    def update_workflow_hint(self, detected_workflow_id: Optional[str] = None) -> None:
        """
        Update the workflow hint in working memory.

        This method adds a hint to the working memory when a workflow is detected
        but not yet activated, to guide the assistant to start the workflow.

        Args:
            detected_workflow_id: ID of the detected workflow, or None if no workflow is detected
        """
        # Store the detected workflow ID for future reference
        if detected_workflow_id:
            self._detected_workflow_id = detected_workflow_id
        else:
            detected_workflow_id = self._detected_workflow_id

        # Remove existing workflow hint if present
        if self._workflow_hint_id and self.working_memory:
            self.working_memory.remove(self._workflow_hint_id)
            self._workflow_hint_id = None

        # If no workflow is detected or the working memory is not provided, return
        if not detected_workflow_id or not self.working_memory:
            return

        # If a workflow is already active, don't add a hint
        if self.get_active_workflow():
            return

        # Get the workflow information
        workflow = self.workflows.get(detected_workflow_id)
        if not workflow:
            return

        # Create the workflow hint
        workflow_hint = f"# Detected Workflow\n"
        workflow_hint += f"I've detected that the user might want help with: {workflow['name']}.\n"
        workflow_hint += "If this seems correct, you can confirm and start this workflow process by including this exact text in your response:\n"
        workflow_hint += f"<workflow_start id=\"{detected_workflow_id}\" />"

        # Add to working memory
        self._workflow_hint_id = self.working_memory.add(
            content=workflow_hint,
            category="workflow_hint"
        )

        self.logger.debug(f"Added workflow hint for '{workflow['name']}' to working memory (item ID: {self._workflow_hint_id})")

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
                message_norm = np.linalg.norm(message_embedding)
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
    
    def update_working_memory(self) -> None:
        """
        Update workflow content in working memory.

        This is the main method called by WorkingMemory to refresh workflow content.
        It updates both active workflow content and workflow hints for detection.
        """
        # Update hints for detected workflows if no active workflow
        if not self.get_active_workflow() and hasattr(self, '_detected_workflow_id') and getattr(self, '_detected_workflow_id', None):
            self.update_workflow_hint(self._detected_workflow_id)

        # Update content for active workflow
        self._update_workflow_content()

    def _update_workflow_content(self) -> None:
        """
        Update workflow content in working memory.

        Uses the category system built into WorkingMemory to efficiently
        manage workflow content, only updating content that has changed.
        """
        if not self.working_memory or not self.active_workflow_id:
            return

        workflow = self.workflows[self.active_workflow_id]

        # If we've switched workflows, clear previous workflow content
        if self._content_cache["current_workflow"] != self.active_workflow_id:
            self._clear_workflow_content()
            self._content_cache["current_workflow"] = self.active_workflow_id

        # Update workflow header (workflow name and description)
        header_content = self._generate_header_content(workflow)
        if header_content != self._content_cache["header"]:
            # Remove existing header content if any
            self.working_memory.remove_by_category("workflow_header")

            # Add new header content
            self.working_memory.add(
                content=header_content,
                category="workflow_header"
            )

            # Update cache
            self._content_cache["header"] = header_content
            self.logger.debug("Updated workflow header in working memory")

        # Update available steps
        current_step_ids = set(self.available_steps)
        cached_step_ids = set(self._content_cache["steps"].keys())

        # Handle steps that are no longer available
        steps_to_remove = cached_step_ids - current_step_ids
        for step_id in steps_to_remove:
            # Remove from working memory by category
            self.working_memory.remove_by_category(f"workflow_step_{step_id}")

            # Remove from cache
            del self._content_cache["steps"][step_id]
            self.logger.debug(f"Removed step {step_id} from working memory")

        # Add or update available steps
        for step_id in current_step_ids:
            step = workflow["steps"].get(step_id)
            if not step:
                continue

            step_content = self._generate_step_content(step_id, step)

            # Only update if content has changed
            if step_content != self._content_cache["steps"].get(step_id):
                # Remove existing step content if any
                self.working_memory.remove_by_category(f"workflow_step_{step_id}")

                # Add new step content
                self.working_memory.add(
                    content=step_content,
                    category=f"workflow_step_{step_id}"
                )

                # Update cache
                self._content_cache["steps"][step_id] = step_content
                self.logger.debug(f"Updated step {step_id} in working memory")

        # Update workflow data
        current_data_fields = set(self.workflow_data.keys())
        cached_data_fields = set(self._content_cache["data"].keys())

        # Handle data fields no longer present
        fields_to_remove = cached_data_fields - current_data_fields
        for field in fields_to_remove:
            # Remove from working memory
            self.working_memory.remove_by_category(f"workflow_data_{field}")

            # Remove from cache
            del self._content_cache["data"][field]
            self.logger.debug(f"Removed data field {field} from working memory")

        # Add or update data fields
        for field, value in self.workflow_data.items():
            data_content = self._generate_data_field_content(field, value, workflow)

            # Only update if content has changed
            if data_content != self._content_cache["data"].get(field):
                # Remove existing data content if any
                self.working_memory.remove_by_category(f"workflow_data_{field}")

                # Add new data content
                self.working_memory.add(
                    content=data_content,
                    category=f"workflow_data_{field}"
                )

                # Update cache
                self._content_cache["data"][field] = data_content
                self.logger.debug(f"Updated data field {field} in working memory")

        # Update checklist
        checklist_content = self._generate_checklist_content(workflow)
        if checklist_content != self._content_cache["checklist"]:
            # Remove existing checklist content if any
            self.working_memory.remove_by_category("workflow_checklist")

            # Add new checklist content
            self.working_memory.add(
                content=checklist_content,
                category="workflow_checklist"
            )

            # Update cache
            self._content_cache["checklist"] = checklist_content
            self.logger.debug("Updated workflow checklist in working memory")

        # Update navigation
        navigation_content = self._generate_navigation_content()
        if navigation_content != self._content_cache["navigation"]:
            # Remove existing navigation content if any
            self.working_memory.remove_by_category("workflow_navigation")

            # Add new navigation content
            self.working_memory.add(
                content=navigation_content,
                category="workflow_navigation"
            )

            # Update cache
            self._content_cache["navigation"] = navigation_content
            self.logger.debug("Updated workflow navigation in working memory")

        self.logger.info(f"Updated workflow content in working memory for workflow: {self.active_workflow_id}")

    def _clear_workflow_content(self) -> None:
        """
        Clear all workflow content from working memory.

        Uses category-based removal for efficient cleanup.
        """
        if not self.working_memory:
            return

        # Remove workflow header
        self.working_memory.remove_by_category("workflow_header")

        # Remove workflow steps - need to handle each step individually
        for step_id in list(self._content_cache["steps"].keys()):
            self.working_memory.remove_by_category(f"workflow_step_{step_id}")

        # Remove workflow data fields - need to handle each field individually
        for field in list(self._content_cache["data"].keys()):
            self.working_memory.remove_by_category(f"workflow_data_{field}")

        # Remove workflow checklist
        self.working_memory.remove_by_category("workflow_checklist")

        # Remove workflow navigation
        self.working_memory.remove_by_category("workflow_navigation")

        # Reset content cache
        self._content_cache = {
            "current_workflow": None,
            "header": None,
            "steps": {},
            "data": {},
            "checklist": None,
            "navigation": None
        }

        self.logger.debug("Cleared all workflow content from working memory")

    def _generate_header_content(self, workflow: Dict[str, Any]) -> str:
        """Generate the workflow header content."""
        content = "\n\n# ACTIVE WORKFLOW GUIDANCE\n"
        content += f"You are currently helping the user with: **{workflow['name']}**\n"
        content += f"Description: {workflow['description']}\n"
        return content

    def _generate_step_content(self, step_id: str, step: Dict[str, Any]) -> str:
        """Generate content for a workflow step."""
        # Mark optional steps
        optional_text = " (Optional)" if step.get("optional", False) else ""

        # Build step content
        content = f"\n## Available Step: {step['description']}{optional_text} (ID: {step_id})\n"
        content += f"{step['guidance']}\n"

        # Add input requirements if defined
        if "inputs" in step and step["inputs"]:
            content += "\n### Required inputs for this step:\n"
            for input_item in step["inputs"]:
                req_marker = "(Required)" if input_item.get("required", False) else "(Optional)"
                input_desc = input_item.get("description", input_item["name"])

                # Add format guidance if available
                format_info = ""
                if input_item.get("type") == "select" and "options" in input_item:
                    options = input_item["options"]
                    if isinstance(options[0], dict):
                        option_values = ", ".join([f"{o.get('label', o['value'])}" for o in options])
                    else:
                        option_values = ", ".join([str(o) for o in options])
                    format_info = f" - Options: {option_values}"
                elif input_item.get("example"):
                    format_info = f" - Example: {input_item['example']}"

                content += f"- **{input_item['name']}** {req_marker}: {input_desc}{format_info}\n"

        return content

    def _generate_data_field_content(self, field: str, value: Any, workflow: Dict[str, Any]) -> str:
        """Generate content for a workflow data field."""
        # Format the value for display
        if isinstance(value, (dict, list)):
            import json
            display_value = json.dumps(value, indent=2)
        else:
            display_value = str(value)

        # Get field description if available
        field_description = ""
        if "data_schema" in workflow and field in workflow["data_schema"]:
            field_description = f" - {workflow['data_schema'][field].get('description', '')}"

        # Create data item content
        return f"\n## Data: {field}{field_description}\n{display_value}"

    def _generate_checklist_content(self, workflow: Dict[str, Any]) -> str:
        """Generate the workflow checklist content."""
        content = "\n## Workflow Checklist\n"

        # Add each step with status indicator
        for step_id, step in sorted(workflow["steps"].items()):
            if step_id in self.completed_steps:
                status_marker = "[✅]"
            elif step_id in self.available_steps:
                status_marker = "[🔄]"
            else:
                status_marker = "[ ]"

            # Mark optional steps
            optional_text = " (Optional)" if step.get("optional", False) else ""

            # Include the step ID explicitly in the checklist
            content += f"{status_marker} {step['description']}{optional_text} `(ID: {step_id})`\n"

        return content

    def _generate_navigation_content(self) -> str:
        """Generate the workflow navigation guidance content."""
        return """
## Navigation Commands
You can navigate the workflow using these commands:
- <workflow_complete_step id="step_id" /> - Mark a step as complete (replace step_id with the actual ID)
- <workflow_skip_step id="step_id" /> - Skip an optional step (replace step_id with the actual ID)
- <workflow_revisit_step id="step_id" /> - Go back to a previously completed step (replace step_id with the actual ID)
- <workflow_complete /> - Complete the entire workflow
- <workflow_cancel /> - Cancel the workflow

## IMPORTANT: Using Step IDs Correctly
- When using workflow commands, you MUST use the exact step ID as shown in parentheses
- Only mark a step as complete when you have collected ALL required inputs for that step
- After completing a step, available steps will automatically update based on the workflow structure
- Complete each step fully before moving to the next step
"""

    def start_workflow(self, workflow_id: str, triggering_message: str = None, llm_bridge = None) -> Dict[str, Any]:
        """
        Start a workflow.

        Args:
            workflow_id: ID of the workflow to start
            triggering_message: Optional message that triggered the workflow, used for initial data extraction
            llm_bridge: Optional LLMBridge instance for extracting data from the triggering message

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

        # Initialize workflow state
        self.completed_steps = set()
        self.workflow_data = {}

        # Attempt to extract initial data from the triggering message if provided
        if triggering_message and llm_bridge:
            extracted_data = self._extract_initial_data(workflow_id, triggering_message, llm_bridge)
            if extracted_data:
                self.workflow_data.update(extracted_data)
                self.logger.info(f"Extracted initial data from triggering message: {extracted_data}")

        # Determine which steps are available at the start
        self.available_steps = set()

        # Check if the workflow has explicitly defined entry points
        entry_points = []
        if "entry_points" in workflow and workflow["entry_points"]:
            entry_points = [
                step_id for step_id in workflow["entry_points"]
                if step_id in workflow["steps"]
            ]
        else:
            # Fall back to finding steps with no prerequisites
            entry_points = [
                step_id for step_id, step in workflow["steps"].items()
                if not step["prerequisites"]
            ]

        # Process entry points - auto-complete steps if we already have their data
        for step_id in entry_points:
            step = workflow["steps"].get(step_id)
            if not step:
                continue

            # Check if this entry point step provides data that we already have
            if "provides_data" in step and step["provides_data"]:
                # If we have all the data this step would provide, mark it completed
                if all(field in self.workflow_data for field in step["provides_data"]):
                    self.completed_steps.add(step_id)
                    self.logger.info(f"Auto-completed step {step_id} based on extracted data")
                else:
                    # Otherwise make it available
                    self._check_and_add_available_step(step_id)
            else:
                # No data requirements, just make it available
                self._check_and_add_available_step(step_id)

        # After handling entry points, calculate all available steps
        # This finds steps that become available due to auto-completed entry points
        for step_id in workflow["steps"]:
            if step_id not in self.completed_steps and step_id not in self.available_steps:
                self._check_and_add_available_step(step_id)

        # Enable tools for all available steps
        self._update_tool_access()

        # Update working memory with workflow content
        if self.working_memory:
            self._update_workflow_content()

        self.logger.info(f"Started workflow: {workflow_id}")

        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "available_steps": list(self.available_steps),
            "completed_steps": list(self.completed_steps),
            "workflow_data": self.workflow_data
        }
        
    def _extract_initial_data(self, workflow_id: str, message: str, llm_bridge) -> Dict[str, Any]:
        """
        Extract initial data for a workflow from a triggering message using the LLM.
        
        Uses the workflow's data schema to guide extraction.
        
        Args:
            workflow_id: ID of the workflow
            message: The message to extract data from
            llm_bridge: LLMBridge instance for LLM access
            
        Returns:
            Dictionary of extracted data that matches the workflow's data schema
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow or not message or not llm_bridge:
            return {}
            
        # Get the data schema for this workflow
        data_schema = workflow.get("data_schema", {})
        if not data_schema:
            return {}
            
        try:
            # Create a system prompt for the LLM to extract structured data
            system_prompt = f"""
            You are a data extraction assistant that extracts structured information from natural language requests.
            
            For the workflow: "{workflow['name']}" ({workflow['description']}), extract any relevant data from the user's message.
            
            Only extract data that is explicitly mentioned or can be clearly inferred. DO NOT make up or assume information not in the message.
            
            The data schema for this workflow contains these possible fields:
            """
            
            # Add information about each potential field
            for field_name, field_info in data_schema.items():
                field_type = field_info.get("type", "string")
                field_description = field_info.get("description", "")
                system_prompt += f"\n- {field_name} ({field_type}): {field_description}"
            
            # Add instructions for the output format
            system_prompt += """
            
            IMPORTANT OUTPUT FORMATTING INSTRUCTIONS:
            1. Return a JSON object with field names as keys and extracted values as values
            2. ONLY include fields that are explicitly mentioned or clearly implied in the message
            3. DO NOT include fields where no information is provided
            4. DO NOT add explanations, comments, or markdown formatting
            5. Return ONLY valid, parseable JSON
            """
            
            # Create the LLM query
            user_message = f"Extract data from this message: {message}"
            
            # Use a small, non-streaming model for this query
            response = llm_bridge.generate_response(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=system_prompt,
                temperature=0.0,  # Use zero temperature for deterministic extraction
                stream=False
            )
            
            # Extract the text content from the response
            response_text = llm_bridge.extract_text_content(response)
            
            # Parse the JSON response
            import json
            
            # Clean up common formatting issues
            # Remove possible markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            # Try to parse the JSON
            try:
                extracted_data = json.loads(response_text)
                
                # Validate the extracted data against the schema
                validated_data = {}
                for field, value in extracted_data.items():
                    if field in data_schema:
                        # Basic type validation
                        field_type = data_schema[field]["type"]
                        
                        # Add the field if it passes basic type validation
                        if (field_type == "string" and isinstance(value, str)) or \
                           (field_type == "number" and isinstance(value, (int, float))) or \
                           (field_type == "integer" and isinstance(value, int)) or \
                           (field_type == "boolean" and isinstance(value, bool)) or \
                           (field_type == "array" and isinstance(value, list)) or \
                           (field_type == "object" and isinstance(value, dict)):
                            validated_data[field] = value
                        else:
                            self.logger.warning(f"Field {field} has incorrect type: expected {field_type}")
                
                return validated_data
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from LLM response: {e}")
                self.logger.debug(f"Response text: {response_text}")
                return {}
            
        except Exception as e:
            self.logger.error(f"Error extracting initial data: {e}")
            return {}
    
    def _check_and_add_available_step(self, step_id: str) -> None:
        """
        Check if a step should be available and add it to available_steps if so.
        
        Args:
            step_id: ID of the step to check
        """
        if not self.active_workflow_id:
            return
        
        workflow = self.workflows[self.active_workflow_id]
        step = workflow["steps"].get(step_id)
        
        if not step:
            return
        
        # Skip if already completed or already available
        if step_id in self.completed_steps or step_id in self.available_steps:
            return
        
        # Check prerequisites
        for prereq in step["prerequisites"]:
            if prereq not in self.completed_steps:
                return  # Prerequisite not met
        
        # Check condition if present
        if "condition" in step:
            # Simple condition evaluation
            condition = step["condition"]
            if condition.startswith("!workflow_data."):
                # Check if data field doesn't exist or is falsy
                data_field = condition.split(".", 1)[1]
                if data_field in self.workflow_data and self.workflow_data[data_field]:
                    return  # Condition not met
            elif condition.startswith("workflow_data."):
                # Check if data field exists and is truthy
                data_field = condition.split(".", 1)[1]
                if data_field not in self.workflow_data or not self.workflow_data[data_field]:
                    return  # Condition not met
        
        # Check if we have required data
        if "requires_data" in step:
            for data_field in step["requires_data"]:
                if data_field not in self.workflow_data:
                    return  # Required data not available
        
        # All checks passed, add to available steps
        self.available_steps.add(step_id)
    
    def complete_step(self, step_id: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete a workflow step and update workflow state.

        Args:
            step_id: ID of the step to complete
            data: Data collected during this step

        Returns:
            Dictionary containing updated workflow information

        Raises:
            ValueError: If the step doesn't exist or isn't available
        """
        if not self.active_workflow_id:
            raise ValueError("No active workflow")

        workflow = self.workflows[self.active_workflow_id]

        if step_id not in workflow["steps"]:
            raise ValueError(f"Step '{step_id}' doesn't exist in workflow '{self.active_workflow_id}'")

        if step_id not in self.available_steps:
            raise ValueError(f"Step '{step_id}' is not currently available")

        # Mark step as completed
        self.completed_steps.add(step_id)
        self.available_steps.remove(step_id)

        # Update workflow data
        if data:
            self.workflow_data.update(data)

        # If the step provides data, record that
        step = workflow["steps"][step_id]
        if "provides_data" in step and not data:
            self.logger.warning(f"Step {step_id} is marked as providing data but no data was provided")

        # Recalculate available steps
        for potential_step_id in workflow["steps"]:
            self._check_and_add_available_step(potential_step_id)

        # Update tool access
        self._update_tool_access()

        # Update working memory
        if self.working_memory:
            self._update_workflow_content()

        self.logger.info(f"Completed workflow step: {step_id}")

        # Check if the workflow is now complete
        is_complete = self._check_workflow_completion()

        if is_complete:
            return self.complete_workflow()

        return {
            "workflow_id": self.active_workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "available_steps": list(self.available_steps),
            "completed_steps": list(self.completed_steps),
            "workflow_data": self.workflow_data
        }
    
    def skip_step(self, step_id: str) -> Dict[str, Any]:
        """
        Skip an optional workflow step.

        Args:
            step_id: ID of the step to skip

        Returns:
            Dictionary containing updated workflow information

        Raises:
            ValueError: If the step doesn't exist, isn't available, or isn't optional
        """
        if not self.active_workflow_id:
            raise ValueError("No active workflow")

        workflow = self.workflows[self.active_workflow_id]

        if step_id not in workflow["steps"]:
            raise ValueError(f"Step '{step_id}' doesn't exist in workflow '{self.active_workflow_id}'")

        if step_id not in self.available_steps:
            raise ValueError(f"Step '{step_id}' is not currently available")

        step = workflow["steps"][step_id]
        if not step.get("optional", False):
            raise ValueError(f"Step '{step_id}' is not optional and cannot be skipped")

        # Mark step as completed without updating data
        self.completed_steps.add(step_id)
        self.available_steps.remove(step_id)

        # Recalculate available steps
        for potential_step_id in workflow["steps"]:
            self._check_and_add_available_step(potential_step_id)

        # Update tool access
        self._update_tool_access()

        # Update working memory
        if self.working_memory:
            self._update_workflow_content()

        self.logger.info(f"Skipped workflow step: {step_id}")

        # Check if the workflow is now complete
        is_complete = self._check_workflow_completion()

        if is_complete:
            return self.complete_workflow()

        return {
            "workflow_id": self.active_workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "available_steps": list(self.available_steps),
            "completed_steps": list(self.completed_steps),
            "workflow_data": self.workflow_data
        }
    
    def revisit_step(self, step_id: str) -> Dict[str, Any]:
        """
        Revisit a previously completed step.

        Args:
            step_id: ID of the step to revisit

        Returns:
            Dictionary containing updated workflow information

        Raises:
            ValueError: If the step doesn't exist or wasn't previously completed
        """
        if not self.active_workflow_id:
            raise ValueError("No active workflow")

        workflow = self.workflows[self.active_workflow_id]

        if step_id not in workflow["steps"]:
            raise ValueError(f"Step '{step_id}' doesn't exist in workflow '{self.active_workflow_id}'")

        if step_id not in self.completed_steps:
            raise ValueError(f"Step '{step_id}' was not previously completed")

        # Move step from completed to available
        self.completed_steps.remove(step_id)
        self.available_steps.add(step_id)

        # Update tool access
        self._update_tool_access()

        # Update working memory
        if self.working_memory:
            self._update_workflow_content()

        self.logger.info(f"Revisiting workflow step: {step_id}")

        return {
            "workflow_id": self.active_workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "available_steps": list(self.available_steps),
            "completed_steps": list(self.completed_steps),
            "workflow_data": self.workflow_data
        }
    
    def _check_workflow_completion(self) -> bool:
        """
        Check if the current workflow is complete based on completion requirements.
        
        Returns:
            True if the workflow is complete, False otherwise
        """
        if not self.active_workflow_id:
            return False
        
        workflow = self.workflows[self.active_workflow_id]
        completion_requirements = workflow.get("completion_requirements", {})
        
        # Check required steps
        required_steps = completion_requirements.get("required_steps", [])
        for step_id in required_steps:
            if step_id not in self.completed_steps:
                return False
        
        # Check required data
        required_data = completion_requirements.get("required_data", [])
        for data_field in required_data:
            if data_field not in self.workflow_data:
                return False
        
        return True
    
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
            "status": "completed",
            "completed_steps": list(self.completed_steps),
            "workflow_data": self.workflow_data
        }

        # Disable all workflow tools before clearing workflow state
        currently_enabled_tools = self.tool_repo.get_enabled_tools()
        for tool_name in currently_enabled_tools:
            try:
                self.tool_repo.disable_tool(tool_name)
                self.logger.info(f"Disabled workflow tool on completion: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error disabling tool {tool_name}: {e}")

        # Clear workflow content from working memory
        if self.working_memory:
            self._clear_workflow_content()

        # Clear the active workflow
        self.active_workflow_id = None
        self.completed_steps = set()
        self.available_steps = set()
        self.workflow_data = {}

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
            "status": "cancelled",
            "completed_steps": list(self.completed_steps),
            "workflow_data": self.workflow_data
        }

        # Disable all workflow tools before clearing workflow state
        currently_enabled_tools = self.tool_repo.get_enabled_tools()
        for tool_name in currently_enabled_tools:
            try:
                self.tool_repo.disable_tool(tool_name)
                self.logger.info(f"Disabled workflow tool on cancellation: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error disabling tool {tool_name}: {e}")

        # Clear workflow content from working memory
        if self.working_memory:
            self._clear_workflow_content()

        # Clear the active workflow
        self.active_workflow_id = None
        self.completed_steps = set()
        self.available_steps = set()
        self.workflow_data = {}

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
            "available_steps": list(self.available_steps),
            "completed_steps": list(self.completed_steps),
            "workflow_data": self.workflow_data,
            "total_steps": len(workflow["steps"])
        }
    
    def _update_tool_access(self) -> None:
        """
        Update tool access based on the current workflow state.
        Enable tools for all available steps and disable tools that aren't needed.
        """
        if not self.active_workflow_id:
            return
        
        workflow = self.workflows[self.active_workflow_id]
        
        # Get all tools needed for current available steps
        needed_tools = set()
        for step_id in self.available_steps:
            step = workflow["steps"].get(step_id)
            if step:
                # Process tools for this step
                if "tools" in step and step["tools"]:
                    # First add all explicitly defined tools (except the special marker)
                    for tool_name in step.get("tools", []):
                        # Skip the special marker and empty tool names
                        if tool_name and tool_name != "specialcase__decideviallm":
                            needed_tools.add(tool_name)
                    
                    # Check for special case of dynamic tool selection
                    if "specialcase__decideviallm" in step["tools"]:
                        # Get tools via LLM reasoning if LLM bridge is available
                        if self.llm_bridge:
                            self.logger.info(f"Determining required tools dynamically for step {step_id}")
                            dynamic_tools = self.determine_required_tools(self.active_workflow_id, step_id)
                            if dynamic_tools:
                                for tool_name in dynamic_tools:
                                    if tool_name:  # Skip empty tool names
                                        needed_tools.add(tool_name)
                                self.logger.info(f"Dynamically enabled tools: {dynamic_tools}")
                else:
                    # No tools defined for this step
                    pass
        
        # Get list of currently enabled tools
        currently_enabled_tools = self.tool_repo.get_enabled_tools()
        
        # First, disable tools that are not needed
        for tool_name in currently_enabled_tools:
            if tool_name not in needed_tools:
                try:
                    self.tool_repo.disable_tool(tool_name)
                    self.logger.info(f"Disabled tool not needed for current workflow state: {tool_name}")
                except Exception as e:
                    self.logger.error(f"Error disabling tool {tool_name}: {e}")
        
        # Then, enable each needed tool
        for tool_name in needed_tools:
            # Skip empty tool names
            if not tool_name:
                self.logger.info("Skipping empty tool name in workflow step")
                continue
                
            try:
                if not self.tool_repo.is_tool_enabled(tool_name):
                    self.tool_repo.enable_tool(tool_name)
                    self.logger.info(f"Enabled tool for workflow state: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error enabling tool {tool_name}: {e}")
    
    def get_system_prompt_extension(self) -> str:
        """
        Get a system prompt extension for an active workflow.
        
        This method generates text to be appended to the system prompt that
        informs the LLM about the active workflow and provides guidance for
        the available steps, including a visual progress tracking.
        
        Returns:
            Text to append to the system prompt, or empty string if no workflow is active
        """
        if not self.active_workflow_id:
            return ""
        
        workflow = self.workflows[self.active_workflow_id]
        
        # Format available steps with their guidance and input requirements
        available_step_info = []
        for step_id in sorted(self.available_steps):
            step = workflow["steps"].get(step_id)
            if step:
                # Mark optional steps
                optional_text = " (Optional)" if step.get("optional", False) else ""
                
                # Add step ID explicitly to help Mira reference it correctly
                step_header = f"### {step['description']}{optional_text} (ID: {step_id})"
                
                # Add guidance
                guidance_text = step['guidance']
                
                # Add input requirements section if inputs are defined
                input_reqs = []
                if "inputs" in step and step["inputs"]:
                    input_reqs.append("\n#### Required inputs for this step:")
                    for input_item in step["inputs"]:
                        req_marker = "(Required)" if input_item.get("required", False) else "(Optional)"
                        input_desc = input_item.get("description", input_item["name"])
                        
                        # Add format guidance if available
                        format_info = ""
                        if input_item.get("type") == "select" and "options" in input_item:
                            options = input_item["options"]
                            if isinstance(options[0], dict):
                                option_values = ", ".join([f"{o.get('label', o['value'])}" for o in options])
                            else:
                                option_values = ", ".join([str(o) for o in options])
                            format_info = f" - Options: {option_values}"
                        elif input_item.get("example"):
                            format_info = f" - Example: {input_item['example']}"
                        
                        input_reqs.append(f"- **{input_item['name']}** {req_marker}: {input_desc}{format_info}")
                
                full_guidance = f"{step_header}\n{guidance_text}{''.join(input_reqs)}"
                available_step_info.append(full_guidance)
        
        available_steps_text = "\n\n".join(available_step_info) if available_step_info else "No steps currently available."
        
        # Format the checklist of all steps
        checklist_items = []
        for step_id, step in workflow["steps"].items():
            if step_id in self.completed_steps:
                # Completed step
                status_marker = "[✅]"
            elif step_id in self.available_steps:
                # Available step
                status_marker = "[🔄]"
            else:
                # Future step
                status_marker = "[ ]"
            
            # Mark optional steps
            optional_text = " (Optional)" if step.get("optional", False) else ""
            # Include the step ID explicitly in the checklist
            checklist_items.append(f"{status_marker} {step['description']}{optional_text} `(ID: {step_id})`")
        
        checklist_text = "\n".join(checklist_items)
        
        # Format collected data
        collected_data_items = []
        for field, value in self.workflow_data.items():
            # Format the value for display
            if isinstance(value, (dict, list)):
                display_value = json.dumps(value, indent=2)
            else:
                display_value = str(value)
            
            # Get field description if available
            field_description = ""
            if "data_schema" in workflow and field in workflow["data_schema"]:
                field_description = f" - {workflow['data_schema'][field].get('description', '')}"
            
            collected_data_items.append(f"**{field}**{field_description}: {display_value}")
        
        collected_data_text = "\n".join(collected_data_items) if collected_data_items else "No data collected yet."
        
        # Build the prompt
        prompt = [
            "\n\n# ACTIVE WORKFLOW GUIDANCE",
            f"You are currently helping the user with: **{workflow['name']}**",
            f"Description: {workflow['description']}",
            "",
            # Remove Current Progress section as it's redundant with the checklist
            # and can be misleading for branching workflows
            "",

            "## Available Steps",
            available_steps_text,
            "",
            "## Workflow Checklist",
            checklist_text,
            "",
            "## Collected Information",
            collected_data_text,
            "",
            "## Navigation Commands",
            "You can navigate the workflow using these commands:",
            "- <workflow_complete_step id=\"step_id\" /> - Mark a step as complete (replace step_id with the actual ID shown in parentheses)",
            "- <workflow_skip_step id=\"step_id\" /> - Skip an optional step (replace step_id with the actual ID shown in parentheses)",
            "- <workflow_revisit_step id=\"step_id\" /> - Go back to a previously completed step (replace step_id with the actual ID shown in parentheses)",
            "- <workflow_complete /> - Complete the entire workflow",
            "- <workflow_cancel /> - Cancel the workflow",
            "",
            "## IMPORTANT: Using Step IDs Correctly",
            "- When using workflow commands, you MUST use the exact step ID as shown in parentheses (e.g., step1, step2, etc.)",
            "- DO NOT create your own step IDs or use descriptions as IDs",
            "- Example: To complete the 'Select frequency' step with ID step2, use: <workflow_complete_step id=\"step2\" />",
            "- Only mark a step as complete when you have collected ALL required inputs for that step",
            "- For steps with conditional paths, ensure you collect appropriate information before proceeding",
            "- After completing a step, available steps will automatically update based on the workflow structure",
            "- Complete the task in step-by-step order; do not skip ahead or try to complete multiple steps at once",
            "",
            "As you assist the user:",
            "1. Focus on collecting ALL the required inputs for the CURRENT step before completing it",
            "2. Reference exact step IDs when using workflow commands",
            "3. Validate user inputs against requirements (format, options) before completing steps",
            "4. Move through the workflow sequentially, completing each step fully",
            "5. Review collected information regularly to ensure accuracy"
        ]
        
        return "\n".join(prompt)
    
    def check_for_workflow_commands(self, message: str) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Check an assistant message for workflow commands.
        
        Args:
            message: Assistant message to check
            
        Returns:
            Tuple of (command_found, command_type, command_params, command_data)
            command_found: True if a command was found, False otherwise
            command_type: "start", "complete_step", "skip_step", "revisit_step", "complete", or "cancel"
            command_params: Additional parameters for the command (e.g., step_id)
            command_data: Additional data provided with the command
        """
        # Use the centralized tag parser
        workflow_action = tag_parser.get_workflow_action(message)
        
        if workflow_action and workflow_action["action"]:
            action = workflow_action["action"]
            
            if action == "start":
                workflow_id = workflow_action["id"]
                if workflow_id:
                    return True, "start", workflow_id, None
            elif action == "complete_step":
                step_id = workflow_action.get("id")
                data = workflow_action.get("data", {})
                if step_id:
                    return True, "complete_step", step_id, data
            elif action == "skip_step":
                step_id = workflow_action.get("id")
                if step_id:
                    return True, "skip_step", step_id, None
            elif action == "revisit_step":
                step_id = workflow_action.get("id")
                if step_id:
                    return True, "revisit_step", step_id, None
            elif action == "complete":
                return True, "complete", None, None
            elif action == "cancel":
                return True, "cancel", None, None
        
        return False, None, None, None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workflow manager state to a dictionary for serialization.
        
        Returns:
            Dictionary containing workflow manager state
        """
        return {
            "active_workflow_id": self.active_workflow_id,
            "completed_steps": list(self.completed_steps),
            "available_steps": list(self.available_steps),
            "workflow_data": self.workflow_data
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load workflow manager state from a dictionary.
        
        Args:
            data: Dictionary containing workflow manager state
        """
        self.active_workflow_id = data.get("active_workflow_id")
        self.completed_steps = set(data.get("completed_steps", []))
        self.available_steps = set(data.get("available_steps", []))
        self.workflow_data = data.get("workflow_data", {})
        
        self.logger.info(f"Loaded workflow manager state: active workflow: {self.active_workflow_id}")
        
        # If there's an active workflow, update tool access
        if self.active_workflow_id:
            self._update_tool_access()
    
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
        
    def determine_required_tools(self, workflow_id: str, step_id: str) -> List[str]:
        """
        Determine required tools for a workflow step using LLM reasoning.
        
        This function:
        1. Collects workflow data and task requirements
        2. Gathers tool descriptions from the tool repository
        3. Uses LLM to determine which tools are needed
        4. Returns a list of tool names to enable
        
        Args:
            workflow_id: ID of the current workflow
            step_id: ID of the step being processed
            
        Returns:
            List of tool names that should be enabled
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            self.logger.error(f"Cannot determine tools: workflow {workflow_id} not found")
            return []
            
        # Get current workflow data and step
        step = workflow["steps"].get(step_id)
        task_data = self.workflow_data.copy()
        
        # Get tool descriptions from repository
        tool_descriptions = {}
        for tool_name in self.tool_repo.get_all_tools():
            tool = self.tool_repo.get_tool(tool_name)
            if hasattr(tool, "simple_description"):
                tool_descriptions[tool_name] = tool.simple_description
            elif hasattr(tool, "description"):
                tool_descriptions[tool_name] = tool.description
            else:
                tool_descriptions[tool_name] = f"Tool: {tool_name}"
        
        # Create prompt for the LLM
        system_prompt = """
        You are a tool selection assistant. Based on the automation task description and requirements,
        determine which tools are necessary for this automation. Only select tools that are directly
        relevant to the described task.
        
        Return your answer as a JSON array of tool names, without explanation or additional text.
        Example: ["email_tool", "calendar_tool"]
        """
        
        # Prepare the user message with task details and available tools
        user_message = f"""
        TASK DESCRIPTION:
        {task_data.get('task_name', 'No task name provided')}
        {task_data.get('task_description', 'No description provided')}
        
        EXECUTION MODE: {task_data.get('execution_mode', 'Not specified')}
        FREQUENCY: {task_data.get('frequency', 'Not specified')}
        
        AVAILABLE TOOLS:
        """
        
        # Add available tools with descriptions
        for tool_name, description in tool_descriptions.items():
            user_message += f"- {tool_name}: {description}\n"
        
        user_message += "\nBased on this information, which tools will be necessary for this automation task? Return only the list of tool names."
        
        try:
            # Get LLM response
            response = self.llm_bridge.generate_response(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=system_prompt,
                temperature=0.0,  # Use zero temperature for deterministic selection
                stream=False
            )
            
            # Extract and parse tool list
            response_text = self.llm_bridge.extract_text_content(response)
            
            # Handle various response formats
            try:
                import json
                import re
                
                # Clean up response to extract just the JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    clean_json = json_match.group(0)
                    tool_list = json.loads(clean_json)
                    
                    # Validate that tools exist
                    valid_tools = [t for t in tool_list if t in tool_descriptions]
                    
                    self.logger.info(f"LLM determined required tools: {valid_tools}")
                    return valid_tools
                else:
                    # Fallback: parse comma-separated list if JSON parsing fails
                    tool_list = [t.strip() for t in response_text.replace('[', '').replace(']', '').split(',')]
                    valid_tools = [t.strip('"\'') for t in tool_list if t.strip('"\'') in tool_descriptions]
                    
                    self.logger.info(f"LLM determined required tools (fallback parsing): {valid_tools}")
                    return valid_tools
                    
            except Exception as e:
                self.logger.error(f"Error parsing tool list: {e}")
                return ["automation_tool"]  # Default fallback
                
        except Exception as e:
            self.logger.error(f"Error determining required tools via LLM: {e}")
            return ["automation_tool"]  # Default fallback