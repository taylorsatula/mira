"""
A simple calculator tool that demonstrates basic tool implementation patterns.

This tool provides a simple calculator with basic arithmetic operations.
It follows the recommended patterns for BotWithMemory tools while
keeping the implementation straightforward and easy to understand.
"""

# Standard library imports
import logging
import math
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Third-party imports
from pydantic import BaseModel, Field

# Local imports
from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from config.registry import registry


# -------------------- CONFIGURATION --------------------

class CalculatorToolConfig(BaseModel):
    """
    Configuration for the calculator_tool.
    
    This defines the parameters that control the calculator's behavior.
    Every tool should have a configuration class registered with the registry.
    """
    # Standard configuration parameter - all tools should include this
    enabled: bool = Field(
        default=True, 
        description="Whether this tool is enabled by default"
    )
    
    # Tool-specific configuration
    decimal_places: int = Field(
        default=4, 
        description="Number of decimal places in calculation results"
    )
    max_value: float = Field(
        default=1e10, 
        description="Maximum allowed value for calculations to prevent overflow"
    )
    allow_complex: bool = Field(
        default=False, 
        description="Whether to allow complex number results (e.g., sqrt(-1))"
    )

# Register with registry - this makes the configuration available to the system
registry.register("calculator_tool", CalculatorToolConfig)


# -------------------- VALIDATION FUNCTIONS --------------------
# Simple tools can use standalone validation functions instead of a validation class

def validate_number(value: Any, param_name: str) -> float:
    """
    Validate that a value is a valid number.
    
    Args:
        value: The value to validate
        param_name: Parameter name for error messages
        
    Returns:
        The value as a float
        
    Raises:
        ToolError: If value is not a valid number
    """
    # Check if parameter is missing (None)
    if value is None:
        raise ToolError(
            f"{param_name} is required and must be a number",
            ErrorCode.TOOL_INVALID_INPUT,
            {param_name: value}  # Include the problematic value in error context
        )
        
    # Try to convert to float, catching any conversion errors
    try:
        return float(value)  # Return the validated and converted value
    except (ValueError, TypeError):
        raise ToolError(
            f"{param_name} must be a valid number, got {type(value).__name__}: {value}",
            ErrorCode.TOOL_INVALID_INPUT,
            {param_name: value}
        )


def validate_operation(operation: Any) -> str:
    """
    Validate operation parameter.
    
    Args:
        operation: Operation to perform
        
    Returns:
        Validated operation string (normalized to lowercase)
        
    Raises:
        ToolError: If operation is invalid
    """
    # Define the list of valid operations
    valid_operations = ["add", "subtract", "multiply", "divide", "power", "sqrt", "sin", "cos"]
    
    # Check if operation is missing or not a string
    if not operation or not isinstance(operation, str):
        raise ToolError(
            "Operation is required and must be a string",
            ErrorCode.TOOL_INVALID_INPUT,
            {"operation": operation}
        )
    
    # Normalize to lowercase for case-insensitive comparison
    operation = operation.lower()
    
    # Check if operation is valid
    if operation not in valid_operations:
        raise ToolError(
            f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}",
            ErrorCode.TOOL_INVALID_INPUT,
            {"operation": operation, "valid_operations": valid_operations}
        )
    
    # Return the validated and normalized operation
    return operation


# -------------------- MAIN TOOL CLASS --------------------

class CalculatorTool(Tool):
    """
    A simple calculator tool providing basic arithmetic operations.
    
    This tool demonstrates the basic patterns for implementing a tool
    in the BotWithMemory system. It provides arithmetic operations
    like addition, subtraction, multiplication, division, as well as
    some mathematical functions.
    """
    
    # Unique identifier for the tool
    name = "calculator_tool"
    
    # Detailed description that explains what the tool does, when to use it,
    # its operations, parameters, and limitations
    description = """
    Performs basic arithmetic calculations. Use this tool when the user wants to perform
    mathematical operations or calculations.
    
    OPERATIONS:
    - add: Adds two numbers
      Parameters:
        num1 (required): First number
        num2 (required): Second number
    
    - subtract: Subtracts the second number from the first
      Parameters:
        num1 (required): Number to subtract from
        num2 (required): Number to subtract
    
    - multiply: Multiplies two numbers
      Parameters:
        num1 (required): First number
        num2 (required): Second number
    
    - divide: Divides the first number by the second
      Parameters:
        num1 (required): Dividend (number to be divided)
        num2 (required): Divisor (number to divide by)
    
    - power: Raises the first number to the power of the second
      Parameters:
        num1 (required): Base number
        num2 (required): Exponent
    
    - sqrt: Calculates the square root of a number
      Parameters:
        num1 (required): Number to find the square root of
    
    - sin: Calculates the sine of an angle in radians
      Parameters:
        num1 (required): Angle in radians
    
    - cos: Calculates the cosine of an angle in radians
      Parameters:
        num1 (required): Angle in radians
    
    RESPONSE FORMAT:
    - All operations return the result of the calculation
    - Results are rounded to 4 decimal places by default
    
    LIMITATIONS:
    - Maximum value is limited to prevent overflow
    - Division by zero is not allowed
    - Complex numbers are not supported by default
    """
    
    # Examples showing how to use the tool and what responses to expect
    usage_examples = [
        {
            "input": {
                "operation": "add",
                "num1": 5,
                "num2": 3
            },
            "output": {
                "success": True,
                "result": 8,
                "operation": "add",
                "inputs": {"num1": 5, "num2": 3}
            }
        },
        {
            "input": {
                "operation": "divide",
                "num1": 10,
                "num2": 2
            },
            "output": {
                "success": True,
                "result": 5,
                "operation": "divide",
                "inputs": {"num1": 10, "num2": 2}
            }
        },
        {
            "input": {
                "operation": "sqrt",
                "num1": 16
            },
            "output": {
                "success": True,
                "result": 4,
                "operation": "sqrt",
                "inputs": {"num1": 16}
            }
        }
    ]
    
    def __init__(self):
        """Initialize the calculator tool."""
        # Call the parent class constructor which sets up the logger
        super().__init__()
        self.logger.info("CalculatorTool initialized")
    
    def run(
        self,
        operation: str,
        num1: Optional[Union[int, float]] = None,
        num2: Optional[Union[int, float]] = None
    ) -> Dict[str, Any]:
        # Special operation to test automation execution
        if operation == "get_time":
            return {
                "success": True,
                "result": datetime.now().isoformat(),
                "operation": "get_time",
                "inputs": {}
            }
        """
        Perform a calculation operation.
        
        This is the main entry point for the tool. All tools must implement
        a run method that takes parameters and returns a dictionary result.
        
        Args:
            operation: The arithmetic operation to perform
            num1: First number for the operation
            num2: Second number (required for some operations)
            
        Returns:
            Dictionary containing the calculation result with standardized format:
            {
                "success": True,              # Whether the operation succeeded
                "result": float or int,       # The calculation result
                "operation": str,             # The operation that was performed
                "inputs": Dict[str, Any]      # The input parameters
            }
            
        Raises:
            ToolError: If parameters are invalid or calculation fails
        """
        # Log the operation being performed
        self.logger.info(f"Running calculator with operation: {operation}")
        
        # Use error_context to wrap all operations - this provides consistent
        # error handling and diagnostics for all tools
        with error_context(
            component_name=self.name,
            operation=f"performing {operation} calculation",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Import config inside the method to avoid circular imports
            # This is a common pattern for accessing configuration
            from config import config
            
            # Get tool configuration values
            decimal_places = config.calculator_tool.decimal_places
            max_value = config.calculator_tool.max_value
            allow_complex = config.calculator_tool.allow_complex
            
            self.logger.debug(f"Using config: decimal_places={decimal_places}, "
                             f"max_value={max_value}, allow_complex={allow_complex}")
            
            # Validate operation first since it determines what other parameters are needed
            operation = validate_operation(operation)
            
            # Validate first number, which is required for all operations
            num1_float = validate_number(num1, "num1")
            
            # Check for value limits to prevent overflow or performance issues
            if abs(num1_float) > max_value:
                raise ToolError(
                    f"Value exceeds maximum allowed ({max_value})",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"num1": num1_float, "max_value": max_value}
                )
            
            # Process different operations
            if operation == "add":
                # For binary operations, validate the second number
                num2_float = validate_number(num2, "num2")
                self.logger.debug(f"Adding {num1_float} + {num2_float}")
                result = num1_float + num2_float
                
            elif operation == "subtract":
                num2_float = validate_number(num2, "num2")
                self.logger.debug(f"Subtracting {num1_float} - {num2_float}")
                result = num1_float - num2_float
                
            elif operation == "multiply":
                num2_float = validate_number(num2, "num2")
                self.logger.debug(f"Multiplying {num1_float} * {num2_float}")
                result = num1_float * num2_float
                
            elif operation == "divide":
                num2_float = validate_number(num2, "num2")
                
                # Special handling for division by zero
                if num2_float == 0:
                    raise ToolError(
                        "Division by zero is not allowed",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"num2": num2_float}
                    )
                
                self.logger.debug(f"Dividing {num1_float} / {num2_float}")
                result = num1_float / num2_float
                
            elif operation == "power":
                num2_float = validate_number(num2, "num2")
                
                # Special handling for negative bases with fractional exponents
                # which would result in complex numbers
                if num1_float < 0 and not num2_float.is_integer() and not allow_complex:
                    raise ToolError(
                        "Cannot raise negative number to fractional power (would result in complex number)",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"num1": num1_float, "num2": num2_float}
                    )
                
                self.logger.debug(f"Calculating power {num1_float} ^ {num2_float}")
                result = num1_float ** num2_float
                
            elif operation == "sqrt":
                # Special handling for negative inputs which would result in complex numbers
                if num1_float < 0 and not allow_complex:
                    raise ToolError(
                        "Cannot calculate square root of negative number (would result in complex number)",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"num1": num1_float}
                    )
                
                self.logger.debug(f"Calculating square root of {num1_float}")
                result = math.sqrt(num1_float)
                
            elif operation == "sin":
                self.logger.debug(f"Calculating sine of {num1_float} radians")
                result = math.sin(num1_float)
                
            elif operation == "cos":
                self.logger.debug(f"Calculating cosine of {num1_float} radians")
                result = math.cos(num1_float)
            
            # Check if result exceeds maximum allowed value
            if isinstance(result, (int, float)) and abs(result) > max_value:
                self.logger.warning(f"Result {result} exceeds maximum allowed value {max_value}")
                raise ToolError(
                    f"Result exceeds maximum allowed value ({max_value})",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                    {"result": result, "max_value": max_value}
                )
            
            # Format the result - round to specified decimal places
            if isinstance(result, float):
                original_result = result
                result = round(result, decimal_places)
                self.logger.debug(f"Rounded result from {original_result} to {result}")
                
                # Convert to integer if it's a whole number for cleaner output
                if result == int(result):
                    result = int(result)
                    self.logger.debug(f"Converted result to integer: {result}")
            
            # Prepare inputs for response for better debuggability
            inputs = {"num1": num1}
            if num2 is not None:
                inputs["num2"] = num2
            
            # Return standardized response format - all tools should follow this pattern
            return {
                "success": True,
                "result": result,
                "operation": operation,
                "inputs": inputs
            }