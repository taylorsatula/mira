"""
Tool relevance engine for dynamic tool management.

This module provides the ToolRelevanceEngine, a system-level utility responsible
for analyzing user messages and dynamically enabling the most relevant tools
for the conversation using a multi-label classification approach with contextual
topic coherence.
"""
import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, cast
from collections import deque
import numpy as np
from numpy.typing import NDArray

from errors import AgentError, ErrorCode, error_context
from config import config
from tools.repo import ToolRepository


class ToolRelevanceEngine:
    """
    System-level utility for dynamically managing tools based on message relevance.
    
    This class analyzes user messages to determine which tools are most relevant
    to the current conversation and dynamically enables/disables them to optimize
    the context window usage. It uses a multi-label classification approach with
    contextual topic coherence to decide which tools are relevant for a given message.
    
    Attributes:
        logger: Logger for tracking engine operations
        tool_repo: Repository of available tools
        classifier: Multi-label classifier for tool relevance
        tool_examples: Dictionary mapping tool names to training examples
        message_history: A short history of recent messages for context
        context_window_size: Number of previous messages to consider for context
        topic_coherence_threshold: Minimum similarity to consider messages related
        tool_activation_history: Tracks when each tool was last activated
        tool_persistence_messages: Minimum messages to keep a tool enabled
        message_counter: Counter for tracking message sequence
        suspended: Flag indicating whether automatic tool suggestion is suspended
    """
    
    def __init__(self, tool_repo: ToolRepository, model):
        """
        Initialize the ToolRelevanceEngine.
        
        Args:
            tool_repo: Repository of available tools
            model: Pre-loaded ONNX embedding model to use
        """
        self.logger = logging.getLogger("tool_relevance_engine")
        self.tool_repo = tool_repo
        
        # Initialize the classifier manager with config parameters and shared model
        self.classifier = MultiLabelClassifier(
            thread_limit=config.tool_relevance.thread_limit,
            cache_dir=os.path.join(config.paths.data_dir, "classifier"),
            model=model
        )
        
        # Tool training examples
        self.tool_examples: Dict[str, Dict[str, Any]] = {}
        
        # Initialize the data directories
        self.tools_data_dir = os.path.join(config.paths.data_dir, "tools")
        
        # Configure context window for conversation history from config
        self.context_window_size = config.tool_relevance.context_window_size
        self.message_history: deque[str] = deque(maxlen=self.context_window_size) #ANNOTATION If we aren't using the whole context window to determine do we need to keep track of the full conversation history and/or is there any performance hit to this?
        
        # Set topic coherence threshold from config
        self.topic_coherence_threshold = config.tool_relevance.topic_coherence_threshold
        
        # Tool persistence tracking
        self.tool_persistence_messages = config.tool_relevance.tool_persistence_messages
        self.tool_activation_history: Dict[str, int] = {}  # Maps tool_name to last_activation_message_id
        self.message_counter = 0  # Incremented for each analyzed message
        
        # Flag to suspend automatic tool suggestion
        self.suspended = False
        
        # Flag to track topic changes from LLM responses
        self.topic_changed = False
        
        # Load the tool examples and prepare classifier
        self._load_tool_examples()
        
    def _load_tool_examples(self) -> None:
        """
        Load tool examples from classifier_examples.json or autogen_classifier_examples.json files 
        and prepare classifier.
        
        This method discovers and reads all classifier_examples.json files from
        tool directories and passes the examples to the classifier for training.
        If no classifier_examples.json file exists but a tool is found, it will
        generate synthetic examples and save them to autogen_classifier_examples.json.
        """
        self.logger.info("Loading tool examples for classification")
        
        try:
            # Create tools data directory if it doesn't exist
            os.makedirs(self.tools_data_dir, exist_ok=True)
            
            # Track file hashes for change detection
            file_hashes = {}
            
            # Get all tool subdirectories
            tool_dirs = [f.path for f in os.scandir(self.tools_data_dir) if f.is_dir()]
            
            # Check for multitool directory and create if it doesn't exist
            multitool_dir = os.path.join(self.tools_data_dir, "multitool")
            if not os.path.exists(multitool_dir):
                os.makedirs(multitool_dir, exist_ok=True)
                tool_dirs.append(multitool_dir)
            elif multitool_dir not in tool_dirs:
                tool_dirs.append(multitool_dir)
                
            # Find directories that need synthetic examples
            tools_needing_examples = []
            
            for tool_dir in tool_dirs:
                tool_name = os.path.basename(tool_dir)
                examples_file = os.path.join(tool_dir, "classifier_examples.json")
                autogen_examples_file = os.path.join(tool_dir, "autogen_classifier_examples.json")
                
                # First case: classifier_examples.json exists - use it
                if os.path.exists(examples_file):
                    try:
                        # Get file hash
                        file_hash = self._calculate_file_hash(examples_file)
                        file_hashes[examples_file] = file_hash
                        
                        # Read examples
                        with open(examples_file, 'r') as f:
                            examples = json.load(f)
                        
                        self.logger.info(f"Loaded {len(examples)} examples for {tool_name}")
                        
                        # Store examples
                        self.tool_examples[tool_name] = {
                            "examples": examples,
                            "file_hash": file_hash,
                            "is_autogen": False
                        }
                    except Exception as e:
                        self.logger.error(f"Error loading examples from {examples_file}: {e}")
                
                # Second case: autogen_classifier_examples.json exists - use it, but warn
                elif os.path.exists(autogen_examples_file):
                    try:
                        # Get file hash
                        # This is still worthwhile because in the future some jabronis might edit 
                        #              the autogen file directly and wonder why it isn't updating.
                        file_hash = self._calculate_file_hash(autogen_examples_file)
                        file_hashes[autogen_examples_file] = file_hash
                        
                        # Read examples
                        with open(autogen_examples_file, 'r') as f:
                            examples = json.load(f)
                        
                        self.logger.warning(
                            f"Using auto-generated examples for {tool_name}. "
                            "For better results, consider creating a custom classifier_examples.json file."
                        )
                        self.logger.info(f"Loaded {len(examples)} auto-generated examples for {tool_name}")
                        
                        # Store examples
                        self.tool_examples[tool_name] = {
                            "examples": examples,
                            "file_hash": file_hash,
                            "is_autogen": True
                        }
                    except Exception as e:
                        self.logger.error(f"Error loading auto-generated examples from {autogen_examples_file}: {e}")
                
                # Third case: No examples found - need to generate them
                else:
                    tools_needing_examples.append(tool_name)
            
            # Generate synthetic examples for tools that need them
            if tools_needing_examples:
                # Check if we have existing file hashes to detect changes
                hashes_file = os.path.join(self.tools_data_dir, "classifier_file_hashes.json")
                old_hashes = {}
                
                if os.path.exists(hashes_file):
                    try:
                        with open(hashes_file, 'r') as f:
                            old_hashes = json.load(f)
                    except Exception as e:
                        self.logger.warning(f"Error reading old file hashes: {e}")
                
                # Check for changed, deleted, or new files that would require regeneration
                needs_regen = False
                # New tools always need generation
                if not old_hashes:
                    needs_regen = True
                    self.logger.info("No previous hash file found, will generate examples")
                else:
                    # Check for new tools that weren't in the old hashes
                    new_files = set(tools_needing_examples) - set([os.path.basename(os.path.dirname(f)) for f in old_hashes.keys() if os.path.basename(os.path.dirname(f)) in tools_needing_examples])
                    if new_files:
                        self.logger.info(f"New tools detected: {new_files}, will generate examples")
                        needs_regen = True
                    
                    # Check for changed or deleted existing files
                    if not needs_regen:
                        for file_path, old_hash in old_hashes.items():
                            if file_path not in file_hashes:
                                self.logger.info(f"File deleted: {file_path}")
                                needs_regen = True
                                break
                            if file_hashes[file_path] != old_hash:
                                self.logger.info(f"File changed: {file_path}")
                                needs_regen = True
                                break
                
                # Generate examples if needed
                if needs_regen:
                    self._generate_synthetic_examples(tools_needing_examples)
                    
                    # Add the newly generated examples to our tool_examples
                    for tool_name in tools_needing_examples:
                        tool_dir = os.path.join(self.tools_data_dir, tool_name)
                        autogen_examples_file = os.path.join(tool_dir, "autogen_classifier_examples.json")
                        
                        if os.path.exists(autogen_examples_file):
                            try:
                                file_hash = self._calculate_file_hash(autogen_examples_file)
                                file_hashes[autogen_examples_file] = file_hash
                                
                                with open(autogen_examples_file, 'r') as f:
                                    examples = json.load(f)
                                
                                self.logger.info(f"Using {len(examples)} newly generated examples for {tool_name}")
                                
                                self.tool_examples[tool_name] = {
                                    "examples": examples,
                                    "file_hash": file_hash,
                                    "is_autogen": True
                                }
                            except Exception as e:
                                self.logger.error(f"Error loading newly generated examples for {tool_name}: {e}")
                else:
                    self.logger.info(f"Skipping synthetic example generation for {len(tools_needing_examples)} tools "
                                   f"because no hash changes were detected") #ANNOTATION when would this happen in practice?
            
            # Determine if we need to retrain
            needs_retrain = False
            
            # Check for new or modified files
            for file_path, new_hash in file_hashes.items():
                if file_path not in old_hashes or old_hashes[file_path] != new_hash:
                    needs_retrain = True
                    self.logger.info(f"File changed or added: {file_path}")
                    break
            
            # Check for deleted files
            for file_path in old_hashes:
                if file_path not in file_hashes:
                    needs_retrain = True
                    self.logger.info(f"File deleted: {file_path}")
                    break
            
            # Save current hashes
            with open(hashes_file, 'w') as f:
                json.dump(file_hashes, f)
            
            # Prepare all examples for the classifier
            all_examples = []
            for tool_data in self.tool_examples.values():
                all_examples.extend(tool_data["examples"])
            
            # Train classifier if we have examples
            if all_examples:
                self.classifier.train_classifier(all_examples, force_retrain=needs_retrain)
            else:
                self.logger.warning("No tool examples found for training classifier")
        
        except Exception as e:
            self.logger.error(f"Error loading tool examples: {e}")
            
    def _generate_synthetic_examples(self, tool_names: List[str]) -> None:
        """
        Generate synthetic examples for tools that don't have classifier_examples.json.
        
        Args:
            tool_names: List of tool names that need examples
        """
        self.logger.info(f"Generating synthetic examples for {len(tool_names)} tools")
        
        try:
            # Defer importing SyntheticDataGenerator until needed
            # to save startup time when generation isn't necessary
            from tools.standalone_scripts.synthetic_data_generator import SyntheticDataGenerator
            
            # Use the shared embedding model
            embedding_model = self.classifier.model
            self.logger.info("Using shared embedding model for synthetic data generation")
            
            # Get the tools directory from the config
            # Default to 'tools' if not explicitly defined
            tools_dir = 'tools'
            if hasattr(config, 'paths') and hasattr(config.paths, 'tools_dir'):
                tools_dir = config.paths.tools_dir
            
            # Initialize generator with config settings
            try:
                # Get API key from config if available
                api_key = None
                if hasattr(config, 'api_key'):
                    api_key = config.api_key
                    
                # Get models from config if available
                analysis_model = None
                generation_model = None
                
                if hasattr(config, 'tools'):
                    if hasattr(config.tools, 'synthetic_data_analysis_model'):
                        analysis_model = config.tools.synthetic_data_analysis_model
                    if hasattr(config.tools, 'synthetic_data_generation_model'):
                        generation_model = config.tools.synthetic_data_generation_model
                
                # Create generator instance
                generator = SyntheticDataGenerator( #ANNOTATION these values should be in the ToolRelevanceEngine config
                    api_key=api_key,
                    analysis_model=analysis_model,  # Sonnet for code analysis
                    generation_model=generation_model,  # Haiku for example generation
                    examples_per_temp=0,  # Zero indicates value will be determined by the algorithm at runtime
                    temperatures=[0.2, 0.8],  # More effective temperature range
                    similarity_threshold=0.97,  # More aggressive deduplication
                    skip_llm_review=False,  # LLM review is required for quality control
                    embedding_model=embedding_model  # Pass the shared embedding model if available
                )
                
                for tool_name in tool_names:
                    try:
                        # Determine the tool file path
                        tool_file_path = os.path.join(tools_dir, f"{tool_name}.py")
                        
                        # Check if the tool file exists
                        if not os.path.exists(tool_file_path):
                            self.logger.warning(f"Tool file {tool_file_path} not found, skipping example generation")
                            continue
                        
                        # Determine output path
                        tool_data_dir = os.path.join(self.tools_data_dir, tool_name)
                        os.makedirs(tool_data_dir, exist_ok=True)
                        output_path = os.path.join(tool_data_dir, "autogen_classifier_examples.json")
                        
                        self.logger.info(f"Generating examples for {tool_name}, output to {output_path}")
                        
                        # Generate examples (with analysis saved automatically)
                        examples = generator.generate_for_tool(
                            tool_path=tool_file_path,
                            save_output=True  # Save both the analysis and examples
                        )
                        
                        # If output path is different from the default, save to that path as well
                        default_path = os.path.join(generator.config.data_dir, "tools", tool_name, "autogen_classifier_examples.json")
                        if default_path != output_path:
                            with open(output_path, 'w') as f:
                                json.dump(examples, f, indent=2)
                        
                        self.logger.info(f"Generated {len(examples)} examples for {tool_name}")
                    except Exception as tool_err:
                        self.logger.error(f"Error processing tool {tool_name} for example generation: {tool_err}")
                
            except Exception as gen_err:
                self.logger.error(f"Error initializing generator: {gen_err}")
        
        except Exception as e:
            self.logger.error(f"Error generating synthetic examples: {e}")
    
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
    
    def _build_contextual_message(self, message: str) -> str:
        """
        Build a contextual message by considering recent message history.
        
        This method uses the topic_changed flag set by the conversation manager
        to determine whether to maintain conversation context or start fresh.
        
        Args:
            message: Current user message
            
        Returns:
            Enhanced message with relevant context (if applicable)
        """
        if not self.message_history:
            # No history yet, just store the message and return as is
            self.message_history.append(message)
            return message
        
        # Check if topic has changed based on the flag
        # If topic has not changed, add to history; otherwise clear history
        if not self.topic_changed:
            self.logger.info("Topic continuing - maintaining conversation context")
            self.message_history.append(message)
            return message
        else:
            # Topic has changed - treat as topic change, clear history and start fresh
            self.logger.info("Topic changed - starting new conversation context")
            self.message_history.clear()
            self.message_history.append(message)
            # Reset the flag after handling the topic change
            self.topic_changed = False
            return message
    
    def analyze_message(self, message: str) -> List[Tuple[str, float]]:
        """
        Analyze a user message to find the most relevant tools with confidence scores.
        
        This method uses the multi-label classifier to determine which tools
        are most relevant to the user message, while maintaining the conversation
        context management via topic change detection.
        
        Args:
            message: User message to analyze
            
        Returns:
            List of (tool_name, confidence_score) tuples
        """
        # Log message (truncated for privacy)
        truncated_message = message[:50] + "..." if len(message) > 50 else message
        self.logger.info(f"Analyzing message: {truncated_message}")
        
        # Use error context for the analysis
        with error_context(
            component_name="ToolRelevanceEngine",
            operation="analyzing message",
            error_class=AgentError,
            error_code=ErrorCode.UNKNOWN_ERROR,
            logger=self.logger
        ):
            # Get the message to use for classification (with any context management)
            # Still using _build_contextual_message but now it manages context via topic_changed flag
            contextual_message = self._build_contextual_message(message)
            
            # Track if we're using a different message than the input
            if contextual_message != message:
                truncated_contextual = contextual_message[:75] + "..." if len(contextual_message) > 75 else contextual_message
                self.logger.info(f"Using contextual message: {truncated_contextual}")
            
            # Use the classifier as before to determine relevant tools
            relevant_tools = self.classifier.classify_message_with_scores(contextual_message)
            
            if not relevant_tools:
                self.logger.info("No relevant tools identified for this message")
                return []
            
            tool_names = [tool[0] for tool in relevant_tools]
            self.logger.info(f"Identified {len(relevant_tools)} relevant tools: {', '.join(tool_names)}")
            return relevant_tools
    
    def manage_tool_relevance(self, message: str) -> List[str]:
        """
        Manage which tools are enabled or disabled based on message relevance.
        
        This method analyzes the message, determines the most relevant tools,
        enables those tools in the tool repository, and disables tools that are
        no longer relevant (subject to persistence rules).
        
        Args:
            message: User message to analyze
            
        Returns:
            List of enabled tool names
        """
        import time
        start_time = time.time()
        self.logger.info("Managing tool relevance based on message")
        
        # Check if engine is suspended
        if self.suspended:
            self.logger.info("Tool relevance engine is suspended, skipping tool analysis")
            return []
        
        # Increment message counter for each new message
        self.message_counter += 1
        current_message_id = self.message_counter
        
        try:
            # Find the most relevant tools with confidence scores
            tool_relevance = self.analyze_message(message)
            
            # Get tool names from relevance scores
            newly_relevant_tools = [tool[0] for tool in tool_relevance] if tool_relevance else []
            
            # Determine tools to persist based on activation history
            persistent_tools = self._get_persistent_tools(current_message_id)
            
            # Combine newly relevant tools with persistent tools
            tools_to_enable = list(set(newly_relevant_tools + persistent_tools))
            
            # Update activation history for newly relevant tools
            for tool_name in newly_relevant_tools:
                self.tool_activation_history[tool_name] = current_message_id
            
            if not tools_to_enable:
                self.logger.info("No relevant or persistent tools found")
                # Don't disable any tools with the empty list - use explicitly defined empty list [] #ANNOTATION what does this mean?
                self._disable_irrelevant_tools([])
                return []
            
            # Enable relevant and persistent tools
            enabled_tools = self._enable_tools(tools_to_enable)
            
            # Disable truly irrelevant tools (those neither relevant nor persistent)
            self._disable_irrelevant_tools(tools_to_enable)
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            self.logger.info(f"Tool relevance management completed in {execution_time:.2f}ms")
            return enabled_tools
        
        except Exception as e:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            self.logger.error(f"Error managing tool relevance: {e} (took {execution_time:.2f}ms)")
            return []
    
    def _get_persistent_tools(self, current_message_id: int) -> List[str]:
        """
        Get tools that should persist due to recent activation.
        
        This method identifies tools that were activated recently enough
        that they should remain enabled according to the persistence rule.
        
        Args:
            current_message_id: ID of the current message being processed
            
        Returns:
            List of tool names that should persist
        """
        persistent_tools = []
        
        for tool_name, activation_message_id in self.tool_activation_history.items():
            # Keep tools enabled if they were activated within the persistence window
            messages_since_activation = current_message_id - activation_message_id
            
            if messages_since_activation < self.tool_persistence_messages:
                self.logger.info(f"Tool {tool_name} persisting due to recent activation ({messages_since_activation}/{self.tool_persistence_messages} messages ago)")
                persistent_tools.append(tool_name)
        
        return persistent_tools
    
    def _enable_tools(self, tools_to_enable: List[str]) -> List[str]: #ANNOTATION this appears to be a private method for enabling tools. Why not use a globally available enable/disable functionality?
        """
        Enable the specified tools in the tool repository.
        
        Args:
            tools_to_enable: List of tool names to enable
            
        Returns:
            List of successfully enabled tool names
        """
        enabled_tools = []
        
        for tool_name in tools_to_enable:
            try:
                if not self.tool_repo.is_tool_enabled(tool_name):
                    self.tool_repo.enable_tool(tool_name)
                    self.logger.info(f"Enabled tool: {tool_name}")
                
                enabled_tools.append(tool_name)
            
            except Exception as e:
                self.logger.error(f"Error enabling tool {tool_name}: {e}")
        
        return enabled_tools
    
    def _disable_irrelevant_tools(self, current_relevant_tools: List[str]) -> None: #ANNOTATION same goes for this private method. Why not have them globally available? Is there something special that these ones need to be private?
        """
        Disable tools that are no longer relevant to the conversation.
        
        This method compares the current set of relevant tools (including
        persistent tools) with previously enabled tools and disables those
        that are no longer needed.
        
        Args:
            current_relevant_tools: List of currently relevant tool names (including persistent tools)
        """
        # Get currently enabled tools from repo
        enabled_tools = self.tool_repo.get_enabled_tools()
        
        # Identify tools to disable (enabled but no longer relevant or persistent)
        to_disable = [tool for tool in enabled_tools if tool not in current_relevant_tools]
        
        if to_disable:
            self.logger.info(f"Disabling {len(to_disable)} tools that are no longer relevant: {', '.join(to_disable)}")
            
            for tool_name in to_disable:
                try:
                    self.tool_repo.disable_tool(tool_name)
                    # Also remove from activation history if we're disabling
                    if tool_name in self.tool_activation_history:
                        del self.tool_activation_history[tool_name]
                except Exception as e:
                    self.logger.error(f"Error disabling tool {tool_name}: {e}")
    
    def enable_relevant_tools(self, message: str) -> List[str]:
        """
        Enable the most relevant tools for a given user message.
        
        DEPRECATED: This method will be removed in a future version.
        Use manage_tool_relevance() instead.
        
        Args:
            message: User message to analyze
            
        Returns:
            List of enabled tool names
        """
        import warnings
        warnings.warn(
            "ToolRelevanceEngine.enable_relevant_tools() is deprecated and will be removed in a future version. "
            "Use ToolRelevanceEngine.manage_tool_relevance() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if self.suspended:
            self.logger.info("Tool relevance engine is suspended, skipping tool analysis")
            return []
            
        return self.manage_tool_relevance(message)
        
    def suspend(self) -> None:
        """
        Suspend automatic tool suggestion.
        
        When suspended, the engine will not analyze messages or enable tools.
        This is useful for workflows where tools are managed explicitly.
        """
        self.suspended = True
        self.logger.info("Tool relevance engine suspended")
        
    def resume(self) -> None:
        """
        Resume automatic tool suggestion.
        
        This re-enables the engine's analysis of messages and automatic tool enabling.
        """
        self.suspended = False
        self.logger.info("Tool relevance engine resumed")
        
    def set_topic_changed(self, topic_changed: bool) -> None:
        """
        Set the topic changed flag based on LLM response.
        
        This method is called by the conversation manager after analyzing
        the LLM's response for topic change tags.
        
        Args:
            topic_changed: Boolean indicating whether the topic has changed
        """
        self.topic_changed = topic_changed
        self.logger.info(f"Topic change flag set to: {topic_changed}")


class MultiLabelClassifier:
    """
    Multi-label classifier for determining tool relevance from messages.
    
    This class implements a one-vs-rest classification approach using MiniLM
    embeddings to determine which tools are relevant to a given user message.
    
    Attributes:
        logger: Logger for tracking classifier operations
        model: MiniLM model for embeddings
        cache_dir: Directory for storing cached classifier data
        thread_limit: Maximum number of threads to use for classification
        classifiers: Dictionary mapping tool names to their classifier data
        embedding_cache: Cache of precomputed embeddings for training examples
        examples: Training examples for all tools
    """
    
    def __init__(self, thread_limit: int = 2, cache_dir: str = "data/classifier", model=None):
        """
        Initialize the MultiLabelClassifier.
        
        Args:
            thread_limit: Maximum number of threads to use for classifier operations
            cache_dir: Directory for storing cached classifier data
            model: Pre-loaded ONNX embedding model to use (for sharing)
        """
        self.logger = logging.getLogger("multi_label_classifier")
        self.cache_dir = cache_dir
        self.thread_limit = thread_limit
        self.model = model
        self.thread_semaphore = threading.Semaphore(thread_limit)
        self.classifiers: Dict[str, Dict[str, Any]] = {}
        self.embedding_cache: Dict[str, List[float]] = {}
        self.examples: List[Dict[str, Any]] = []
        
        # Create the cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the model if not provided
        if self.model is None:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize the ONNX embedding model.
        
        This method is no longer needed as the model is passed in during initialization.
        """
        if self.model is None:
            self.logger.error("No ONNX model provided")
        else:
            self.logger.info("Using provided ONNX embedding model")
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0) where 1.0 is identical
        """
        try:
            # Get embeddings
            embedding1 = self._compute_embedding(text1)
            embedding2 = self._compute_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Normalize embeddings
            emb1_array = np.array(embedding1)
            emb2_array = np.array(embedding2)
            
            norm1 = np.linalg.norm(emb1_array)
            norm2 = np.linalg.norm(emb2_array)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            emb1_norm = emb1_array / norm1
            emb2_norm = emb2_array / norm2
            
            # Calculate cosine similarity
            similarity = float(np.dot(emb1_norm, emb2_norm))
            
            # Ensure value is in range [0, 1]
            return max(0.0, min(similarity, 1.0))
        
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    def train_classifier(self, examples: List[Dict[str, Any]], force_retrain: bool = False) -> None:
        """
        Train the one-vs-rest classifier using provided examples.
        
        This method either loads a cached classifier state or trains new classifiers
        based on the provided examples.
        
        Args:
            examples: List of dictionaries containing tool_name/tool_names and query pairs
            force_retrain: Whether to force retraining even if a cached state exists
        """
        self.logger.info(f"Training one-vs-rest classifier with {len(examples)} examples")
        
        # Store examples for reference
        self.examples = examples
        
        # Check if we have a cached classifier state
        cache_file = os.path.join(self.cache_dir, "classifier_state.json")
        embedding_cache_file = os.path.join(self.cache_dir, "embedding_cache.json")
        
        try:
            # Check if cache exists and whether to use it
            if os.path.exists(cache_file) and os.path.exists(embedding_cache_file) and not force_retrain:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                with open(embedding_cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                
                self.logger.info("Loading cached classifier data")
                self.classifiers = cache_data.get('classifiers', {})
                
                if self.classifiers:
                    self.logger.info(f"Loaded {len(self.classifiers)} classifiers from cache")
                    return
        except Exception as e:
            self.logger.warning(f"Error loading cached classifier data: {e}")
            # Reset caches if there was an error
            self.classifiers = {}
            self.embedding_cache = {}
        
        # If we get here, we need to train new classifiers
        self.logger.info("Training new classifiers")
        self.classifiers = {}
        
        # First, prepare all unique tool names from examples
        tool_names = set()
        
        for example in examples:
            # Handle both single tool and multiple tools in examples
            if "tool_name" in example:
                tool_names.add(example["tool_name"])
            elif "tool_names" in example and isinstance(example["tool_names"], list):
                for tool in example["tool_names"]:
                    tool_names.add(tool)
        
        self.logger.info(f"Found {len(tool_names)} unique tools in examples")
        
        # Compute embeddings for all unique queries (to avoid recomputing)
        self._precompute_embeddings(examples)
        
        # For each tool, create its one-vs-rest classifier
        for tool_name in tool_names:
            self.logger.info(f"Training classifier for {tool_name}")
            
            # Collect positive and negative examples for this tool
            positive_examples = []
            negative_examples = []
            
            for example in examples:
                if "query" not in example:
                    continue
                
                query = example["query"]
                
                # Determine if this example is positive or negative for this tool
                is_positive = False
                
                if "tool_name" in example and example["tool_name"] == tool_name:
                    is_positive = True
                elif "tool_names" in example and isinstance(example["tool_names"], list) and tool_name in example["tool_names"]:
                    is_positive = True
                
                if is_positive:
                    positive_examples.append(query)
                else:
                    negative_examples.append(query)
            
            # Only create classifier if we have positive examples
            if not positive_examples:
                self.logger.warning(f"No positive examples found for {tool_name}, skipping")
                continue
            
            self.logger.info(f"Tool {tool_name}: {len(positive_examples)} positive, {len(negative_examples)} negative examples")
            
            # Create classifier data
            classifier_data = self._create_tool_classifier(tool_name, positive_examples, negative_examples)
            
            if classifier_data:
                self.classifiers[tool_name] = classifier_data
        
        # Cache the classifier state
        self._cache_classifier_state()
        
        # Cache the embedding cache
        try:
            with open(embedding_cache_file, 'w') as f:
                json.dump(self.embedding_cache, f)
                
            self.logger.info(f"Cached {len(self.embedding_cache)} embeddings at {embedding_cache_file}")
        except Exception as e:
            self.logger.error(f"Error caching embeddings: {e}")
    
    def _precompute_embeddings(self, examples: List[Dict[str, Any]]) -> None:
        """
        Precompute embeddings for all unique queries in the examples.
        
        Args:
            examples: List of dictionaries containing tool_name/tool_names and query pairs
        """
        # Collect unique queries
        unique_queries = set()
        for example in examples:
            if "query" in example:
                unique_queries.add(example["query"])
        
        # Filter out queries that are already in the embedding cache
        queries_to_compute = [q for q in unique_queries if q not in self.embedding_cache]
        
        if not queries_to_compute:
            self.logger.info("All query embeddings already in cache")
            return
        
        self.logger.info(f"Precomputing embeddings for {len(queries_to_compute)} unique queries")
        
        # Compute embeddings in batches to save time
        import numpy as np
        batch_size = 16  # Adjust based on available memory
        
        for i in range(0, len(queries_to_compute), batch_size):
            batch = queries_to_compute[i:i+batch_size]
            
            try:
                # Limit concurrent computations
                with self.thread_semaphore:
                    # Skip long texts that could cause memory issues
                    valid_batch = [text for text in batch if len(text) <= 8192]
                    
                    if not valid_batch:
                        continue
                    
                    # Compute embeddings for the batch
                    model = self.model
                    if model is not None and hasattr(model, 'encode'):
                        embeddings = model.encode(valid_batch)
                        
                        # Store in cache
                        for j, text in enumerate(valid_batch):
                            self.embedding_cache[text] = embeddings[j].tolist()
                
                self.logger.debug(f"Computed embeddings for batch {i//batch_size + 1}/{(len(queries_to_compute) + batch_size - 1)//batch_size}")
            
            except Exception as e:
                self.logger.error(f"Error computing embeddings for batch: {e}")
        
        self.logger.info(f"Precomputed embeddings for {len(queries_to_compute)} queries")
    
    def _create_tool_classifier(self, tool_name: str, positive_examples: List[str], negative_examples: List[str]) -> Optional[Dict[str, Any]]:
        """
        Create a one-vs-rest classifier for a specific tool.
        
        Args:
            tool_name: Name of the tool
            positive_examples: List of queries that should use this tool
            negative_examples: List of queries that should not use this tool
            
        Returns:
            Dictionary containing classifier data and parameters or None if creation fails
        """
        try:
            # Get embeddings for positive examples from cache
            positive_embeddings: List[List[float]] = []
            for example in positive_examples:
                if example in self.embedding_cache:
                    positive_embeddings.append(self.embedding_cache[example])
            
            # Get embeddings for negative examples from cache
            negative_embeddings: List[List[float]] = []
            for example in negative_examples:
                if example in self.embedding_cache:
                    negative_embeddings.append(self.embedding_cache[example])
            
            if not positive_embeddings:
                self.logger.warning(f"No cached embeddings found for positive examples for {tool_name}")
                return None
            
            # Convert to numpy arrays
            pos_emb_array = np.array(positive_embeddings)
            
            # Compute centroid of positive examples (normalized)
            positive_centroid = np.mean(pos_emb_array, axis=0)
            positive_centroid_norm = np.linalg.norm(positive_centroid)
            if positive_centroid_norm > 0:
                positive_centroid = positive_centroid / positive_centroid_norm
            
            # Compute distances between positive examples and centroid
            distances = []
            for embedding in positive_embeddings:
                emb_array = np.array(embedding)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    normalized_embedding = emb_array / norm
                    # Distance = 1 - cosine similarity
                    dist = 1.0 - float(np.dot(normalized_embedding, positive_centroid))
                    distances.append(dist)
            
            # Calculate threshold adaptively
            if len(distances) >= 10:
                # If we have enough examples, use statistical method
                mean_distance = float(np.mean(distances))
                std_distance = float(np.std(distances))
                # Use mean + 2 standard deviations to cover ~97.5% of positive examples
                threshold = mean_distance + 2.0 * std_distance
            else:
                # With few examples, use a more generous threshold
                if distances:
                    max_distance = max(distances)
                    threshold = max_distance * 1.2  # Allow some margin
                else:
                    threshold = 0.3  # Default fallback
            
            # Verify threshold using negative examples
            if negative_embeddings:
                neg_distances = []
                
                for embedding in negative_embeddings:
                    emb_array = np.array(embedding)
                    norm = np.linalg.norm(emb_array)
                    if norm > 0:
                        normalized_embedding = emb_array / norm
                        dist = 1.0 - float(np.dot(normalized_embedding, positive_centroid))
                        neg_distances.append(dist)
                
                negative_distances = neg_distances
                
                # Check if threshold would incorrectly classify negative examples
                false_positives = sum(1 for d in negative_distances if d <= threshold)
                false_positive_rate = false_positives / len(negative_distances) if negative_distances else 0
                
                # If false positive rate is too high, adjust threshold
                if false_positive_rate > 0.1 and negative_distances:  # More than 10% false positives
                    # Find a better threshold that balances false positives and false negatives
                    all_distances = [(d, True) for d in distances] + [(d, False) for d in negative_distances]
                    all_distances.sort(key=lambda x: x[0])  # Sort by distance
                    
                    best_threshold = threshold
                    best_error = float('inf')
                    
                    # Try different thresholds to find the one with minimum error
                    for i in range(len(all_distances)):
                        candidate_threshold = all_distances[i][0]
                        false_negatives = sum(1 for d, pos in all_distances if pos and d > candidate_threshold)
                        false_positives = sum(1 for d, pos in all_distances if not pos and d <= candidate_threshold)
                        
                        error = false_negatives + false_positives
                        if error < best_error:
                            best_error = error
                            best_threshold = candidate_threshold
                    
                    threshold = best_threshold
            
            # Clamp threshold to reasonable range to prevent extreme values
            threshold = max(0.05, min(float(threshold), 0.6))
            
            # Create classifier data
            classifier_data: Dict[str, Any] = {
                "tool_name": tool_name,
                "positive_centroid": positive_centroid.tolist(),
                "threshold": threshold,
                "positive_count": len(positive_embeddings),
                "negative_count": len(negative_embeddings),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Created classifier for {tool_name} with threshold {threshold:.4f}")
            return classifier_data
            
        except Exception as e:
            self.logger.error(f"Error creating classifier for {tool_name}: {e}")
            return None
    
    def _cache_classifier_state(self) -> None:
        """
        Cache the current classifier state.
        """
        cache_file = os.path.join(self.cache_dir, "classifier_state.json")
        
        try:
            cache_data = {
                'classifiers': self.classifiers,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            self.logger.info(f"Cached classifier state at {cache_file}")
        
        except Exception as e:
            self.logger.error(f"Error caching classifier state: {e}")
    
    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Compute an embedding for text using the ONNX model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values, or None if computation fails
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        if self.model is None:
            self.logger.error("Model not initialized, cannot compute embedding")
            return None
        
        try:
            # Limit concurrent computations
            with self.thread_semaphore:
                # Truncate very long texts to avoid memory issues
                if len(text) > 8192:
                    self.logger.warning("Text too long, truncating to 8192 characters")
                    text = text[:8192]
                
                # Compute embedding using ONNX model
                embedding = self.model.encode(text)
                
                # Convert to Python list
                result = embedding.tolist()
                
                # Add to cache
                self.embedding_cache[text] = result
                
                return result
        
        except Exception as e:
            self.logger.error(f"Error computing embedding: {e}")
            return None
    
    def classify_message_with_scores(self, message: str) -> List[Tuple[str, float]]:
        """
        Classify a message to determine which tools are relevant, with confidence scores.
        
        This method uses the one-vs-rest classifiers to determine which tools
        are relevant to the given message and returns their confidence scores.
        
        Args:
            message: User message to classify
            
        Returns:
            List of (tool_name, confidence_score) tuples where confidence is 0.0-1.0
        """
        if not self.classifiers:
            self.logger.error("No classifiers available for classification")
            return []
        
        if self.model is None:
            self.logger.error("Model not initialized, cannot classify message")
            return []
        
        try:
            import numpy as np
            
            # Compute message embedding
            message_embedding = self._compute_embedding(message)
            
            if message_embedding is None:
                self.logger.error("Failed to compute message embedding")
                return []
            
            # Normalize message embedding
            message_embedding = np.array(message_embedding)
            message_norm = np.linalg.norm(message_embedding)
            if message_norm > 0:
                message_embedding = message_embedding / message_norm
            
            # Evaluate each classifier
            tool_scores = []
            
            for tool_name, classifier_data in self.classifiers.items():
                # Get classifier components
                positive_centroid = np.array(classifier_data["positive_centroid"])
                threshold = classifier_data["threshold"]
                
                # Calculate similarity and distance to positive centroid
                similarity = float(np.dot(message_embedding, positive_centroid))
                distance = 1.0 - similarity
                
                # Calculate confidence score (1.0 = perfect match, 0.0 = not relevant)
                # Normalize based on threshold: confidence is 0.5 at threshold, scaling up to 1.0
                if distance <= threshold:
                    # Convert distance to confidence score (inverted and scaled)
                    # At distance=0, confidence=1.0; at distance=threshold, confidence=0.5
                    confidence = 1.0 - (distance / (threshold * 2))
                    confidence = max(0.5, min(1.0, confidence))  # Clamp to [0.5, 1.0]
                    
                    tool_scores.append((tool_name, confidence))
                    self.logger.info(f"Tool {tool_name} is relevant (distance: {distance:.4f}, threshold: {threshold:.4f}, confidence: {confidence:.4f})")
                else:
                    self.logger.debug(f"Tool {tool_name} is not relevant (distance: {distance:.4f}, threshold: {threshold:.4f})")
            
            # Special case: if no tools are relevant but one is very close
            if not tool_scores:
                closest_tool = None
                closest_distance = float('inf')
                closest_threshold = 0.0
                
                for tool_name, classifier_data in self.classifiers.items():
                    positive_centroid = np.array(classifier_data["positive_centroid"])
                    threshold = classifier_data["threshold"]
                    
                    similarity = np.dot(message_embedding, positive_centroid)
                    distance = 1.0 - similarity
                    
                    # Track closest tool
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_tool = tool_name
                        closest_threshold = threshold
                
                # If closest tool is within a relaxed threshold, use it
                relaxed_factor = 1.5
                if closest_tool and closest_distance <= relaxed_factor * closest_threshold:
                    # Confidence score between 0.0 and 0.5
                    confidence = 0.5 - ((closest_distance - closest_threshold) / (closest_threshold * relaxed_factor))
                    confidence = max(0.2, min(0.5, confidence))  # Clamp to [0.2, 0.5]
                    
                    tool_scores.append((closest_tool, confidence))
                    self.logger.info(f"No tools met threshold, but {closest_tool} was close (distance: {closest_distance:.4f}, confidence: {confidence:.4f})")
            
            # Sort by confidence score (highest first)
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            return tool_scores
        
        except Exception as e:
            self.logger.error(f"Error classifying message: {e}")
            return []
    
    def classify_message(self, message: str) -> List[str]:
        """
        Classify a message to determine which tools are relevant.
        
        This method uses the one-vs-rest classifiers to determine which tools
        are relevant to the given message.
        
        Args:
            message: User message to classify
            
        Returns:
            List of tool names deemed relevant to the message
        """
        # Get classification with scores
        results_with_scores = self.classify_message_with_scores(message)
        
        # Extract just the tool names
        return [tool_name for tool_name, _ in results_with_scores]