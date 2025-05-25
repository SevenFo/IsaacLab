"""
Logging utilities for the robot brain system.
"""

import logging
import os
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration for the robot brain system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_format: Optional custom log format

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[],
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)

    # Create file handler if specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)

    # Get robot brain system logger
    logger = logging.getLogger("robot_brain_system")
    logger.handlers.clear()  # Clear any existing handlers
    logger.addHandler(console_handler)
    if log_file:
        logger.addHandler(file_handler)
    logger.setLevel(getattr(logging, log_level.upper()))

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.

    Args:
        name: Logger name (will be prefixed with 'robot_brain_system.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"robot_brain_system.{name}")


# Pre-configured loggers for common components
def get_system_logger() -> logging.Logger:
    """Get logger for the main system."""
    return get_logger("system")


def get_brain_logger() -> logging.Logger:
    """Get logger for the brain component."""
    return get_logger("brain")


def get_simulator_logger() -> logging.Logger:
    """Get logger for the simulator."""
    return get_logger("simulator")


def get_skill_logger() -> logging.Logger:
    """Get logger for skills."""
    return get_logger("skills")
