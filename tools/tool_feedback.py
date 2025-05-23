"""
Tool Feedback Module

This module handles collecting and storing feedback about tool activations to help
improve the tool training and classification process. It includes LLM-powered analysis
of feedback to provide insights for improving tool classification.
"""
import datetime
import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from config import config
from conversation import Conversation
from api.llm_bridge import LLMBridge


def analyze_feedback_with_llm(
    feedback_data: Dict[str, Any], 
    llm_bridge: LLMBridge
) -> Dict[str, Any]:
    """
    Analyze tool feedback using LLM to provide insights for improving tool classification.
    
    Args:
        feedback_data: Dictionary containing feedback and context information
        llm_bridge: LLM Bridge instance for making API calls
        
    Returns:
        Dictionary containing the LLM analysis of the feedback
    """
    # Create a prompt for the LLM
    prompt = f"""
    # Tool Classification Feedback Analysis
    
    ## User Feedback
    "{feedback_data['feedback']}"
    
    ## Recent Messages
    {json.dumps(feedback_data['last_messages'], indent=2)}
    
    ## Active Tools
    {', '.join(feedback_data['active_tools']) if feedback_data['active_tools'] else 'None'}
    
    ## Tool Classification Thresholds
    {json.dumps(feedback_data.get('tool_thresholds', {}), indent=2)}
    
    ## Similar Training Examples
    {json.dumps(feedback_data['nearest_examples'], indent=2)}
    
    """

# Provide a VERY CONCISE analysis (2-3 sentences maximum) that includes:
# 1. COMPARE: Specifically compare words in the user message with the closest matching examples
# 2. SUGGEST: Recommend a specific training example to add OR a precise threshold adjustment
# 
# Be extremely specific and actionable. Mention exact similarity scores, specific words that caused incorrect matches, 
# and suggest concrete examples like: "Add example: 'Check my Square bookings for next week' to square_tool".
# 
# Your entire analysis must be under 50 words and focus solely on this specific case.

    try:
        # Create a message structure for the LLM Bridge
        messages = [{"role": "user", "content": prompt}]
        
        # Make the API call
        response = llm_bridge.generate_response(
            messages=messages,
            system_prompt="""You are an expert AI system analyzer specializing in tool classification systems. Analyze semantic similarity patterns between queries and examples. Provide ONLY concrete suggestions like 'Add training example X' or 'Adjust threshold for tool Y from 0.8 to 0.7'. Focus on specific words/phrases that caused matching issues. Use 2-3 sentences maximum. Be extremely specific and actionable. Mention exact similarity scores, specific words that caused incorrect matches, and suggest concrete examples like: "Add example: 'Check my Square bookings for next week' to square_tool".""",
            temperature=0.2  # Lower temperature for more precise response
        )
        
        # Extract the text from the response
        response_text = llm_bridge.extract_text_content(response)
        
        # Return the analysis
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis": response_text
        }
    
    except Exception as e:
        logging.error(f"Error analyzing feedback with LLM: {e}")
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis": f"Error analyzing feedback: {str(e)}",
            "error": True
        }


def save_tool_feedback(system: Dict[str, Any], feedback_text: str, conversation: Conversation) -> Tuple[bool, Optional[str]]:
    """
    Save tool feedback along with contextual information to improve tool training data.
    Includes LLM analysis of the feedback for improving classification.
    
    Args:
        system: Dictionary of system components
        feedback_text: User feedback text
        conversation: Current conversation instance
        
    Returns:
        Tuple of (success_boolean, llm_analysis_text)
    """
    try:
        # Get the components from the system
        tool_relevance_engine = system.get('tool_relevance_engine')
        llm_bridge = system.get('llm_bridge')
        
        # Create feedback directory
        feedback_dir = Path(config.paths.persistent_dir) / "tool_feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Extract the last 3 messages from conversation if available, ensuring JSON serializable data
        last_messages = []
        for message in list(conversation.messages)[-3:]:
            # Safely extract content by converting complex objects to strings
            content = message.content
            if not isinstance(content, (str, int, float, bool, dict, list, type(None))):
                # Handle complex content objects by converting to string
                content = str(content)
            elif isinstance(content, list) and any(not isinstance(item, (str, int, float, bool, dict, list, type(None))) for item in content):
                # Handle lists with complex objects
                content = [str(item) if not isinstance(item, (str, int, float, bool, dict, list, type(None))) else item for item in content]
            
            # Use the Message's to_dict method to ensure proper serialization
            message_dict = message.to_dict()
            last_messages.append({
                "role": message_dict["role"],
                "content": content,  # Use our sanitized content
                "timestamp": message_dict["created_at"],  # Use the already serialized datetime
                "id": message_dict["id"]
            })
        
        # Get active tools information
        active_tools = []
        nearest_examples = {}
        threshold_info = {}
        if tool_relevance_engine:
            # Get currently active tools from activation history
            active_tools = list(tool_relevance_engine.tool_activation_history.keys())
            
            # Try to get threshold information for activated tools
            if hasattr(tool_relevance_engine, 'classifier') and hasattr(tool_relevance_engine.classifier, 'classifiers'):
                classifiers = tool_relevance_engine.classifier.classifiers
                for tool_name in active_tools:
                    if tool_name in classifiers:
                        threshold = classifiers[tool_name].get("threshold")
                        if threshold is not None:
                            threshold_info[tool_name] = threshold
            
            # Try to find nearest training examples if possible
            if hasattr(tool_relevance_engine, 'classifier') and hasattr(tool_relevance_engine.classifier, 'calculate_text_similarity'):
                # Get the last user message content
                last_user_message = ""
                for message in reversed(conversation.messages):
                    if message.role == "user" and isinstance(message.content, str):
                        last_user_message = message.content
                        break
                
                if last_user_message:
                    # Check similarity against tool examples for active tools
                    for tool_name in active_tools:
                        if tool_name in tool_relevance_engine.tool_examples:
                            examples = tool_relevance_engine.tool_examples[tool_name].get("examples", [])
                            similar_examples = []
                            
                            for example in examples:
                                if "query" in example:
                                    query = example["query"]
                                    similarity = tool_relevance_engine.classifier.calculate_text_similarity(last_user_message, query)
                                    similar_examples.append({
                                        "query": query,
                                        "similarity": similarity
                                    })
                            
                            # Sort by similarity and keep top 3
                            similar_examples.sort(key=lambda x: x["similarity"], reverse=True)
                            nearest_examples[tool_name] = similar_examples[:3]
        
        # Create feedback entry
        timestamp = datetime.datetime.now().isoformat()
        feedback_entry = {
            "timestamp": timestamp,
            "feedback": feedback_text,
            "conversation_id": conversation.conversation_id,
            "last_messages": last_messages,
            "active_tools": active_tools,
            "tool_thresholds": threshold_info,
            "nearest_examples": nearest_examples
        }
        
        # Generate LLM analysis if bridge is available
        llm_analysis = None
        if llm_bridge:
            logging.info("Generating LLM analysis of tool feedback...")
            analysis_result = analyze_feedback_with_llm(feedback_entry, llm_bridge)
            feedback_entry["llm_analysis"] = analysis_result
            llm_analysis = analysis_result.get("analysis")
        else:
            logging.warning("LLM Bridge not available, skipping feedback analysis")
            feedback_entry["llm_analysis"] = {
                "timestamp": timestamp,
                "analysis": "LLM analysis not available - LLM Bridge not found in system components",
                "error": True
            }
        
        # Generate a unique filename with timestamp
        filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = feedback_dir / filename
        
        # Save feedback to file - use a default handler for any objects that aren't JSON serializable
        with open(filepath, 'w') as f:
            json.dump(feedback_entry, f, indent=2, default=lambda o: str(o))
        
        logging.info(f"Tool feedback saved to {filepath}")
        return (True, llm_analysis)
    
    except Exception as e:
        logging.error(f"Error saving tool feedback: {e}")
        return (False, None)


def get_feedback_summary() -> Dict[str, Any]: #ANNOTATION <- Question: Where is this used outside this file?
    """
    Generate a summary of collected tool feedback.
    
    Returns:
        Dictionary with feedback statistics and summaries
    """
    feedback_dir = Path(config.paths.persistent_dir) / "tool_feedback"
    
    if not feedback_dir.exists():
        return {"count": 0, "feedback": []}
    
    feedback_files = list(feedback_dir.glob("feedback_*.json"))
    all_feedback = []
    tool_mentions = {}
    
    for file_path in feedback_files:
        try:
            with open(file_path, 'r') as f:
                feedback = json.load(f)
                all_feedback.append(feedback)
                
                # Track tool mentions in feedback
                for tool in feedback.get("active_tools", []):
                    tool_mentions[tool] = tool_mentions.get(tool, 0) + 1
                    
        except Exception as e:
            logging.error(f"Error reading feedback file {file_path}: {e}")
    
    # Sort tools by number of mentions
    sorted_tools = sorted(tool_mentions.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "count": len(all_feedback),
        "tools": dict(sorted_tools),
        "feedback": all_feedback[:10]  # Return most recent 10 feedback entries
    }