"""
Unified logging utilities for the Object Detection & Tracking project.
Provides consistent logging setup across all scripts.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path


class PrintToLogger:
    """
    Redirect print statements to logger.
    Usage:
        print_redirect = PrintToLogger(logger)
        print = print_redirect  # Replace built-in print
    """
    
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self.original_print = __builtins__['print']
    
    def __call__(self, *args, **kwargs):
        """Redirect print calls to logger."""
        # Remove 'file' and 'flush' from kwargs if present (logger handles these)
        log_kwargs = {k: v for k, v in kwargs.items() if k not in ('file', 'flush')}
        
        # Convert all args to string
        message = ' '.join(str(arg) for arg in args)
        
        # Log the message
        self.logger.log(self.level, message)
        
        # Also print to console if flush was requested (for real-time updates)
        if kwargs.get('flush', False):
            self.original_print(*args, **kwargs, flush=True)


def setup_logging(
    log_dir: str,
    log_level: str = "INFO",
    log_file_name: Optional[str] = None,
    log_format: Optional[str] = None,
    script_name: Optional[str] = None,
    redirect_print: bool = False
) -> tuple[logging.Logger, str]:
    """
    Setup unified logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file_name: Name of log file (if None, uses timestamp-based name)
        log_format: Custom log format string (if None, uses default)
        script_name: Name of script (for default log filename)
        redirect_print: If True, redirects print() to logger
        
    Returns:
        tuple: (logger, log_file_path)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine log file name
    if log_file_name:
        log_file = os.path.join(log_dir, log_file_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if script_name:
            log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
        else:
            log_file = os.path.join(log_dir, f"log_{timestamp}.log")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        if hasattr(handler, 'close'):
            handler.close()
    
    # Create formatter
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove any existing handlers from this logger (let it propagate to root)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Log initialization
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log level: {log_level}")
    
    # Force flush to ensure first message is written
    file_handler.flush()
    
    # Store file_handler reference for later flushing
    logger._file_handler = file_handler
    root_logger._file_handler = file_handler
    
    # Optionally redirect print to logger
    if redirect_print:
        print_redirect = PrintToLogger(logger, logging.INFO)
        # Note: We don't replace built-in print globally to avoid breaking things
        # Instead, scripts can use: print = PrintToLogger(logger)
        logger._print_redirect = print_redirect
    
    return logger, log_file


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

