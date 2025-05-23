"""
Error logging configuration for separate system and tool error logs.

This module sets up dedicated loggers for different error types to ensure
error analysis and correlations are persisted for debugging and improvement.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Ensure logs directory exists
LOG_DIR = Path("persistent/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_error_loggers():
    """
    Set up separate loggers for system errors, tool errors, and error analysis.
    
    Returns:
        tuple: (system_logger, tool_logger, analysis_logger)
    """
    # Configure formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    
    # System error logger - for core system errors
    system_logger = logging.getLogger('errors.system')
    system_logger.setLevel(logging.ERROR)
    system_handler = RotatingFileHandler(
        LOG_DIR / 'system_errors.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    system_handler.setFormatter(detailed_formatter)
    system_logger.addHandler(system_handler)
    
    # Tool error logger - for tool-specific errors
    tool_logger = logging.getLogger('errors.tool')
    tool_logger.setLevel(logging.ERROR)
    tool_handler = RotatingFileHandler(
        LOG_DIR / 'tool_errors.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    tool_handler.setFormatter(detailed_formatter)
    tool_logger.addHandler(tool_handler)
    
    # Error analysis logger - for MIRA's error analysis
    analysis_logger = logging.getLogger('errors.analysis')
    analysis_logger.setLevel(logging.INFO)
    analysis_handler = RotatingFileHandler(
        LOG_DIR / 'error_analysis.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    analysis_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    analysis_handler.setFormatter(analysis_formatter)
    analysis_logger.addHandler(analysis_handler)
    
    return system_logger, tool_logger, analysis_logger


# Create singleton instances
system_error_logger, tool_error_logger, error_analysis_logger = setup_error_loggers()