"""
Logging utilities for the robot brain system.
"""

import logging
import os
import sys  # ## 新增：导入 sys 模块
from typing import Optional


# ## 新增：定义流重定向类
class StreamToLogger:
    """
    一个文件类对象，可以将流式输出 (如 sys.stdout) 重定向到日志记录器。
    A file-like object that redirects stream output (e.g., sys.stdout)
    to a logger instance.
    """

    def __init__(self, logger: logging.Logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf: str):
        # 将传入的缓冲区按行分割处理
        for line in buf.rstrip().splitlines():
            # 使用 logger 记录每一行，移除末尾的空白符
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # 这个方法是文件类接口所必需的，我们这里不需要任何操作。
        # This method is required for the file-like interface.
        pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    redirect_print: bool = True,  # ## 新增：控制是否重定向 print
) -> logging.Logger:
    """
    Setup logging configuration for the robot brain system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_format: Optional custom log format
        redirect_print: If True, redirects stdout and stderr to the logger. ## 新增
    """
    if log_format is None:
        log_format = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

    numeric_log_level = getattr(logging, log_level.upper())

    # Configure root logger - this is fine as a base
    logging.basicConfig(
        level=numeric_log_level,
        format=log_format,
        handlers=[],  # Start with no handlers on the root
    )

    # Create console handler
    console_handler = logging.StreamHandler(
        sys.__stdout__
    )  # ## 修改：确保 handler 输出到原始终端
    console_handler.setLevel(numeric_log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)

    # Get robot brain system logger
    logger = logging.getLogger("robot_brain_system")
    logger.handlers.clear()
    logger.propagate = False  # ## 新增：防止日志向上传播导致重复打印

    # Add console handler
    logger.addHandler(console_handler)

    # Create and add file handler if specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(numeric_log_level)

    # ## 新增：重定向标准输出和标准错误
    if redirect_print:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        logger.info("标准输出 (print) 已被重定向到日志系统。")

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
