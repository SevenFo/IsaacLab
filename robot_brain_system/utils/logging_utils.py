"""
Logging utilities for the robot brain system.
"""

import logging
import os
import sys
from typing import Optional

_console_ref = None
_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "system": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _get_console():
    global _console_ref
    if _console_ref is None:
        try:
            from robot_brain_system.ui.console import global_console

            _console_ref = global_console
        except ImportError:
            pass
    return _console_ref


class TUIHandler(logging.Handler):
    """
    自定义日志处理器，将日志转发给 ConsoleUI。
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            console = _get_console()

            # 映射 logging level 到 UI category
            category = "info"
            if record.levelno >= logging.ERROR:
                category = "error"
            elif record.levelno >= logging.WARNING:
                category = "wrarning"

            if console:
                # 发送到 UI 的日志窗口
                console.log(category, msg)
            else:
                # UI 未启动时，仍然输出到 stderr，避免日志静默
                sys.stderr.write(msg + "\n")
        except Exception:
            self.handleError(record)


def console_log(category: str, message: str):
    """
    Unified console logging helper.
    - If UI exists, use it.
    - Otherwise, fall back to standard logger (robot_brain_system).
    """
    console = _get_console()
    level = _level_map.get(category.lower(), logging.INFO)
    if console:
        console.log(category, message)
    else:
        logging.getLogger("robot_brain_system").log(level, message)


def silence_terminal_logging():
    """Remove default stream handlers to avoid breaking TUI layout.

    Call this after UI starts to prevent logs printing to the terminal.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(logging.NullHandler())
    root.propagate = False


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

    def isatty(self):
        """Mimic a file object behavior."""
        return False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    redirect_print: bool = False,  # [MODIFIED] 默认为 False，由 UI 接管
) -> logging.Logger:
    """
    Setup logging configuration.
    """
    if log_format is None:
        # 简化格式，因为 UI 已经有时间戳和分类颜色了
        log_format = "%(message)s"

    numeric_log_level = getattr(logging, log_level.upper())

    # 1. 获取 Logger
    logger = logging.getLogger("robot_brain_system")
    logger.handlers.clear()
    logger.setLevel(numeric_log_level)
    logger.propagate = False

    # 2. [CRITICAL] 添加 TUI Handler
    # 这确保了 logger.info() 的内容会进入 UI 窗口，而不是直接打印到屏幕
    tui_handler = TUIHandler()
    tui_handler.setLevel(numeric_log_level)
    tui_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(tui_handler)

    # 3. 文件 Handler (保持不变)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_log_level)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    # 4. 可选：在没有 UI 时，将 print 重定向到 logger（用于离线脚本）
    if redirect_print and _get_console() is None:
        sys.stdout = StreamToLogger(logger)
        sys.stderr = StreamToLogger(logger, log_level=logging.ERROR)

    # 注意：当 UI 启动后，ConsoleUI.run() 会接管 sys.stdout/err，
    # print 会进入 UI；logger 仍通过 TUIHandler 进入 UI + 文件。

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
