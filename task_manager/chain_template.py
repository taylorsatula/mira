"""
Chain template module for parameter substitution in task chains.

This module provides the TemplateEngine class for resolving template expressions
in chain step parameters, allowing data to flow between steps.
"""

import logging
import re
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union, Match, Pattern

from errors import ToolError, ErrorCode, error_context

# Configure logger
logger = logging.getLogger(__name__)


class TemplateEngine:
    """
    Template engine for resolving parameter templates in task chains.
    
    Handles parameter substitution using a simple template language that
    supports dot notation, array indexing, and basic formatting.
    """
    
    # Regex for finding template expressions like {variable.path}
    TEMPLATE_PATTERN: Pattern = re.compile(r'{([^{}]*)}')
    
    # Regex for path segments in dot notation (handles array indexing)
    PATH_SEGMENT_PATTERN: Pattern = re.compile(r'([^\.\[\]]+)|\[(\d+)\]')
    
    # Regex for format specifiers like {date:YYYY-MM-DD}
    FORMAT_PATTERN: Pattern = re.compile(r'([^:]+):(.+)')
    
    def __init__(self):
        """Initialize the template engine."""
        self.logger = logging.getLogger(__name__)
        
    def resolve_template(self, template: Any, context: Dict[str, Any]) -> Any:
        """
        Resolve a template against the given context.
        
        Args:
            template: The template string, object, or value to resolve
            context: The context dictionary with variables for substitution
            
        Returns:
            The resolved value
            
        Raises:
            ToolError: If the template cannot be resolved
        """
        with error_context(
            component_name="TemplateEngine",
            operation="resolve_template",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # If template is None, return None
            if template is None:
                return None
                
            # If template is a string, process template expressions
            if isinstance(template, str):
                # Check if this is a simple template expression
                if template.startswith('{') and template.endswith('}') and template.count('{') == 1:
                    # This is a simple expression like "{variable.path}" - resolve the value directly
                    expr = template[1:-1]
                    try:
                        return self._resolve_expression(expr, context)
                    except Exception as e:
                        self.logger.warning(f"Error resolving expression '{expr}': {str(e)}")
                        # Return the original template if can't resolve
                        return template
                
                # Process complex template with multiple expressions
                return self._process_template_string(template, context)
                
            # If template is a dictionary, process each value
            elif isinstance(template, dict):
                return {k: self.resolve_template(v, context) for k, v in template.items()}
                
            # If template is a list, process each item
            elif isinstance(template, list):
                return [self.resolve_template(item, context) for item in template]
                
            # For other types, return as is
            return template
    
    def _process_template_string(self, template_str: str, context: Dict[str, Any]) -> str:
        """
        Process a template string by substituting all expressions.
        
        Args:
            template_str: The template string to process
            context: The context dictionary with variables
            
        Returns:
            The processed string with expressions substituted
        """
        def replace_func(match: Match) -> str:
            expr = match.group(1)
            try:
                value = self._resolve_expression(expr, context)
                # Convert value to string for substitution
                if value is None:
                    return "null"
                elif isinstance(value, (dict, list)):
                    return json.dumps(value)
                else:
                    return str(value)
            except Exception as e:
                self.logger.warning(f"Error resolving expression '{expr}': {str(e)}")
                # Return the original expression if can't resolve
                return match.group(0)
        
        # Replace all template expressions
        return self.TEMPLATE_PATTERN.sub(replace_func, template_str)
    
    def _resolve_expression(self, expr: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a single expression against the context.
        
        Args:
            expr: The expression to resolve (without braces)
            context: The context dictionary with variables
            
        Returns:
            The resolved value
            
        Raises:
            KeyError: If the path cannot be found in the context
            IndexError: If an array index is out of bounds
            ValueError: If the expression is invalid
        """
        # Check for format specifier
        format_match = self.FORMAT_PATTERN.match(expr)
        if format_match:
            path = format_match.group(1)
            format_spec = format_match.group(2)
            value = self._resolve_path(path, context)
            return self._apply_format(value, format_spec)
        
        # Regular path resolution
        return self._resolve_path(expr, context)
    
    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a path against the context using dot notation.
        
        Args:
            path: The path to resolve (e.g., "weather.temperature")
            context: The context dictionary with variables
            
        Returns:
            The resolved value
            
        Raises:
            KeyError: If the path cannot be found
            IndexError: If an array index is out of bounds
        """
        # Handle special built-in variables
        if path == "date":
            return datetime.now(timezone.utc)
        elif path == "time":
            return datetime.now(timezone.utc).strftime("%H:%M:%S")
        elif path == "timestamp":
            return datetime.now(timezone.utc).timestamp()
        
        # Extract path segments
        segments = []
        for match in self.PATH_SEGMENT_PATTERN.finditer(path):
            if match.group(1):  # Named segment (weather, temperature, etc.)
                segments.append(match.group(1))
            elif match.group(2):  # Array index [0], [1], etc.
                segments.append(int(match.group(2)))
        
        # Start with the entire context
        current = context
        
        # Navigate through path segments
        for segment in segments:
            if isinstance(current, dict):
                if isinstance(segment, str) and segment in current:
                    current = current[segment]
                else:
                    raise KeyError(f"Key '{segment}' not found in context")
            elif isinstance(current, list):
                if isinstance(segment, int) and 0 <= segment < len(current):
                    current = current[segment]
                else:
                    raise IndexError(f"Index {segment} out of bounds for list of length {len(current)}")
            else:
                raise ValueError(f"Cannot access '{segment}' on a non-container value")
        
        return current
    
    def _apply_format(self, value: Any, format_spec: str) -> str:
        """
        Apply a format specifier to a value.
        
        Args:
            value: The value to format
            format_spec: The format specifier
            
        Returns:
            The formatted value as string
        """
        # Handle datetime formatting
        if isinstance(value, datetime):
            # Simple datetime format replacement
            format_str = format_spec
            format_str = format_str.replace("YYYY", "%Y")
            format_str = format_str.replace("MM", "%m")
            format_str = format_str.replace("DD", "%d")
            format_str = format_str.replace("HH", "%H")
            format_str = format_str.replace("mm", "%M")
            format_str = format_str.replace("ss", "%S")
            
            return value.strftime(format_str)
        
        # Handle numeric formatting
        if isinstance(value, (int, float)):
            if format_spec.startswith("fixed"):
                # fixed(2) -> 2 decimal places
                decimal_places = int(format_spec.replace("fixed", "").strip("()"))
                return f"{value:.{decimal_places}f}"
                
            if format_spec == "percent":
                return f"{value * 100:.0f}%"
        
        # Handle string formatting
        if isinstance(value, str):
            if format_spec == "upper":
                return value.upper()
            elif format_spec == "lower":
                return value.lower()
            elif format_spec == "title":
                return value.title()
            elif format_spec.startswith("truncate"):
                # truncate(10) -> limit to 10 chars
                length = int(format_spec.replace("truncate", "").strip("()"))
                return value[:length] + "..." if len(value) > length else value
        
        # Default behavior: return as string
        return str(value)
    
    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a conditional expression against the context.
        
        Args:
            condition: The condition string (e.g., "{temperature} > 70")
            context: The context dictionary with variables
            
        Returns:
            The result of the condition evaluation (True or False)
            
        Raises:
            ToolError: If the condition cannot be evaluated
        """
        with error_context(
            component_name="TemplateEngine",
            operation="evaluate_condition",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Process templates in the condition
            processed_condition = self._process_template_string(condition, context)
            
            # Create a safe evaluation environment
            safe_globals = {
                "__builtins__": {
                    "True": True,
                    "False": False,
                    "None": None,
                    "int": int,
                    "float": float,
                    "str": str,
                    "len": len,
                    "bool": bool
                }
            }
            
            try:
                # Evaluate the condition
                return bool(eval(processed_condition, safe_globals, {}))
            except Exception as e:
                self.logger.error(f"Error evaluating condition '{condition}': {str(e)}")
                raise ToolError(
                    f"Invalid condition: {str(e)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )