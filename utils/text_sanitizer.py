"""
Text sanitization utilities for safe message content processing.

Provides minimal, performant functions to ensure messages won't break
system components, without unnecessary overhead.
"""
import logging
from typing import Union, List, Dict, Any

logger = logging.getLogger(__name__)

# Length enforced at API layer (cns/api/chat.py MAX_TEXT_MESSAGE_LENGTH);
# sanitizer deliberately does not truncate.


def sanitize_message_content(content: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, Any]]]:
    """
    Minimal sanitization for safe message processing.
    
    Focuses on preventing actual breaking issues rather than
    theoretical edge cases.
    
    Args:
        content: Message content (string or multimodal array)
        
    Returns:
        Sanitized content in the same format as input
    """
    if isinstance(content, str):
        return _sanitize_text(content)
    elif isinstance(content, list):
        return _sanitize_multimodal(content)
    else:
        return _sanitize_text(str(content))


def _sanitize_text(text: str) -> str:
    """
    Minimal text sanitization focusing on actual breaking issues.
    
    Operations:
    1. Ensure string type
    2. Remove null bytes (breaks many systems)
    3. Ensure valid UTF-8 (prevents encoding errors)
    4. Apply reasonable length limit
    
    Args:
        text: Raw text content
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    if '\0' in text:
        text = text.replace('\0', '')
        logger.debug("Removed null bytes from message")
    
    # Ensure valid UTF-8 - critical for JSON encoding
    try:
        text.encode('utf-8')
    except UnicodeError:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        logger.warning("Fixed invalid UTF-8 sequences in message")

    return text


def _sanitize_multimodal(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitize multimodal content array.
    
    Only sanitizes text portions; validates image format.
    
    Args:
        content: Multimodal content array
        
    Returns:
        Sanitized multimodal content
    """
    sanitized = []
    
    for item in content:
        if not isinstance(item, dict):
            logger.warning(f"Skipping invalid multimodal item type: {type(item)}")
            continue
        
        item_copy = item.copy()
        
        if item_copy.get('type') == 'text' and 'text' in item_copy:
            item_copy['text'] = _sanitize_text(item_copy['text'])
        
        elif item_copy.get('type') == 'image_url':
            if not isinstance(item_copy.get('image_url'), dict):
                logger.warning("Invalid image_url structure")
                continue
        
        sanitized.append(item_copy)
    
    return sanitized
