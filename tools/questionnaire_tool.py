import json
import os
from typing import Dict, List, Any, Optional

from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from config import config
from api.llm_bridge import LLMBridge


class QuestionnaireTool(Tool):
    """
    Interactive questionnaire tool that collects structured user responses.

    This tool conducts a sequential question-answer session with the user,
    keeping the interaction local (not sending responses to LLM) until all
    questions are answered. At the end, it collects and returns all responses.
    """

    name = "questionnaire_tool"
    description = "Runs an interactive questionnaire with multiple questions for the user, collects their responses locally, and returns a structured result"
    usage_examples = [
        {
            "input": {"questionnaire_id": "preferences"},
            "output": {
                "questionnaire_id": "preferences",
                "completed": True,
                "responses": {
                    "Cuisine": "Italian",
                    "Ingredients to use": "Tomatoes",
                    "Dietary restrictions": "No",
                    "Cooking time": "Less than 30 minutes",
                    "Difficulty": "Intermediate"
                }
            }
        }
    ]

    def __init__(self, llm_bridge: LLMBridge):
        """
        Initialize the questionnaire tool.
        
        Args:
            llm_bridge: LLMBridge instance for generating dynamic content
        """
        super().__init__()
        # Tool-specific state
        self.data_dir = config.paths.data_dir
        self.llm_bridge = llm_bridge
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def run(
        self,
        questionnaire_id: str,
        custom_questions: Optional[List[Dict[str, Any]]] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run an interactive questionnaire session with the user.

        Args:
            questionnaire_id: ID of the predefined questionnaire to use, or custom ID for custom questions
            custom_questions: Optional list of custom questions to use instead of a predefined questionnaire
            context_data: Optional contextual data to use for dynamic question generation

        Returns:
            Dictionary containing questionnaire results including all responses

        Raises:
            ToolError: If questionnaire file not found or has invalid format
        """
        self.logger.info(f"Starting questionnaire: {questionnaire_id}")

        # Use the centralized error context for questionnaire operations
        with error_context(
            component_name=self.name,
            operation="running questionnaire",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Initialize context if not provided
            if context_data is None:
                context_data = {}
                
            # Load questions - either from custom_questions or from file
            if custom_questions:
                questions = custom_questions
                self.logger.debug(f"Using custom questions for questionnaire {questionnaire_id}")
            else:
                # Load predefined questionnaire
                # Try both with and without .json extension
                questionnaire_id_clean = questionnaire_id.replace('.json', '')
                questionnaire_path = os.path.join(self.data_dir, f"{questionnaire_id_clean}.json")
                if not os.path.exists(questionnaire_path):
                    raise ToolError(
                        f"Questionnaire not found: {questionnaire_id_clean}",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"questionnaire_id": questionnaire_id_clean}
                    )
                
                try:
                    with open(questionnaire_path, 'r') as f:
                        questionnaire_data = json.load(f)
                        
                    # Check if file contains valid questionnaire format
                    if not isinstance(questionnaire_data, list) or not questionnaire_data:
                        raise ToolError(
                            f"Invalid questionnaire format for {questionnaire_id}",
                            ErrorCode.TOOL_INVALID_INPUT
                        )
                    
                    questions = questionnaire_data
                    self.logger.debug(f"Loaded {len(questions)} questions for questionnaire {questionnaire_id}")
                except json.JSONDecodeError:
                    raise ToolError(
                        f"Invalid JSON in questionnaire file: {questionnaire_id}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )

            # Process dynamic questions if LLM bridge is available
            if self.llm_bridge:
                questions = self._process_dynamic_questions(questions, context_data)

            # Run the questionnaire
            responses = self._run_interactive_questionnaire(questions, context_data)
            
            # Format and return the results
            result = {
                "questionnaire_id": questionnaire_id,
                "completed": True,
                "responses": responses
            }
            
            self.logger.info(f"Completed questionnaire: {questionnaire_id}")
            return result
            
    def _process_dynamic_questions(
        self, 
        questions: List[Dict[str, Any]], 
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process questions with dynamic content from LLM.
        
        This handles questions that need personalization based on previous answers
        or other contextual data.
        
        Args:
            questions: List of question objects
            context_data: Contextual data including previous answers
            
        Returns:
            Updated list of questions with dynamic content resolved
        """
        processed_questions = []
        
        for question in questions:
            # Make a copy to avoid modifying the original
            q = question.copy()
            
            # Check if this is a dynamic question
            if q.get('dynamic_type') == 'llm_options':
                # Generate options using LLM
                prompt_template = q.get('prompt_template', '')
                
                # Replace placeholders in the prompt template
                prompt = prompt_template
                for key, value in context_data.items():
                    placeholder = f"{{{{{key}}}}}"
                    if placeholder in prompt:
                        prompt = prompt.replace(placeholder, str(value))
                
                # Get options from LLM
                if prompt and self.llm_bridge:
                    self.logger.debug(f"Generating dynamic options with prompt: {prompt}")
                    try:
                        # Create message format for the bridge
                        messages = [
                            {"role": "user", "content": prompt}
                        ]
                        
                        # Call the LLM to generate options
                        response = self.llm_bridge.generate_response(messages)
                        content = self.llm_bridge.extract_text_content(response)
                        
                        # Parse options (assuming one per line)
                        options = [line.strip() for line in content.strip().split('\n') if line.strip()]
                        q['options'] = options
                        self.logger.debug(f"Generated options: {options}")
                    except Exception as e:
                        self.logger.error(f"Error generating dynamic options: {e}")
                        # Fallback options
                        q['options'] = q.get('fallback_options', ['Option not available'])
            
            processed_questions.append(q)
            
        return processed_questions

    def _run_interactive_questionnaire(
        self, 
        questions: List[Dict[str, Any]], 
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs the interactive questionnaire with the user.
        
        This method conducts the local interaction loop with the user,
        presenting questions and collecting responses without sending
        intermediate responses to the LLM.
        
        Args:
            questions: List of question objects, each containing at minimum
                      'id', 'text', and either 'options' or 'free_text' flag
            context_data: Optional contextual data for dynamic content and
                         updating context between questions
                      
        Returns:
            Dictionary of responses mapped to question keys
        """
        # Initialize responses dictionary 
        responses = {}
        
        # Create a working copy of context data that will be updated as we go
        working_context = context_data.copy() if context_data else {}
        
        # Start questionnaire message
        print("\n--- Starting Interactive Questionnaire ---")
        print("Please answer the following questions. Your answers will be collected locally.")
        print("Type your answer or select from the options provided.\n")
        
        # Process each question
        for i, question in enumerate(questions):
            # Extract question details
            question_id = question.get('id', f"q{i+1}")
            question_text = question.get('text', "")
            options = question.get('options', [])
            
            # Display question
            print(f"\nQ{i+1}: {question_text}")
            
            # If multiple-choice question, display options
            if options:
                for j, option in enumerate(options):
                    print(f"  {j+1}. {option}")
                
                # Get and validate user response
                while True:
                    user_input = input("\nYour answer (enter number or text): ").strip()
                    
                    # Check if user entered a number
                    try:
                        option_num = int(user_input)
                        if 1 <= option_num <= len(options):
                            response = options[option_num-1]
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(options)}")
                    except ValueError:
                        # User entered text, check if it matches an option
                        if user_input in options:
                            response = user_input
                            break
                        else:
                            # Always accept free text (implicit allow_free_text for all questions)
                            response = user_input
                            break
            else:
                # Free text question
                response = input("\nYour answer: ").strip()
            
            # Store the response
            responses[question_id] = response
            print(f"Response recorded: {response}")
            
            # Update working context with this response
            working_context[question_id] = response
            
            # If there are dependent questions coming up, generate their content now
            if i < len(questions) - 1:
                next_questions = questions[i+1:]
                dynamic_next = [q for q in next_questions if q.get('dynamic_type') == 'llm_options']
                
                if dynamic_next and self.llm_bridge:
                    # Process the next dynamic questions with updated context
                    questions[i+1:] = self._process_dynamic_questions(next_questions, working_context)
        
        # End questionnaire message
        print("\n--- Questionnaire Complete ---")
        print("Thank you for your responses. Returning to the conversation.\n")
        
        # Map question IDs to any provided preference keys
        preference_mapping = {}
        for question in questions:
            qid = question.get('id')
            pref_key = question.get('preference_key', qid)
            if qid and pref_key != qid:
                preference_mapping[qid] = pref_key
        
        # Apply preference mapping to responses if specified
        if preference_mapping:
            mapped_responses = {}
            for qid, response in responses.items():
                mapped_key = preference_mapping.get(qid, qid)
                mapped_responses[mapped_key] = response
            return mapped_responses
        
        return responses