# robot_brain_system/ui/console.py

import threading
import queue
import sys
import os
import time
import atexit
from typing import Optional, Callable

from robot_brain_system.utils.logging_utils import silence_terminal_logging

from textual.app import App, ComposeResult
from textual.widgets import Footer, Input, Label, Static
from textual.containers import Container

from textual.widgets import RichLog
from rich.text import Text


_UI_INSTANCE = None


class InterventionDialog(Static):
    """A dialog box that appears when the agent needs human intervention."""

    DEFAULT_CSS = """
    InterventionDialog {
        layer: dialog;
        width: 80%;
        height: auto;
        max-height: 50%;
        margin: 1 2;
        padding: 2;
        background: #0f111a;
        border: tall #e94560;
        content-align: center middle;
    }
    
    InterventionDialog #dialog-content {
        width: 100%;
        color: #ffd700;
        text-align: center;
        text-style: bold;
        padding: 1;
    }
    
    InterventionDialog #dialog-hint {
        width: 100%;
        height: auto;
        padding: 0 1;
        color: #888888;
        text-align: center;
        text-style: italic;
    }
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        yield Static(f"INTERVENTION: {self.message}", id="dialog-content")
        yield Static(
            "💡 Please type your response in the input box below, or type 'c' to continue, 'f' to mark failed.",
            id="dialog-hint",
        )


class RobotBrainApp(App):
    CSS = """
    Screen { layout: vertical; }
    #status_bar { dock: top; height: 1; background: #333333; color: #eeeeee; content-align: center middle; text-style: bold; }
    
    #main_container {
        height: 1fr;
        layout: vertical;
    }
    
    #dialog_overlay {
        dock: top;
        height: auto;
        min-height: 4;
        max-height: 50%;
        width: 100%;
        align: center middle;
        layer: dialog;
    }
    
    RichLog { height: 1fr; border: solid green; background: #000000; overflow-y: scroll; }
    Input { dock: bottom; height: 3; border: solid #00ff00; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Label("Initializing System...", id="status_bar")
        with Container(id="main_container"):
            yield RichLog(
                id="log_view", highlight=True, markup=True, max_lines=10000, wrap=True
            )
        yield Input(placeholder="Type command here...", id="input_box")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Robot Brain Console"
        self.set_interval(0.1, self.poll_logs_from_queue)
        self.set_interval(0.5, self.update_status_bar)
        self.set_interval(0.2, self.check_intervention_request)
        self.query_one(Input).focus()
        self.query_one(RichLog).auto_scroll = False

    def poll_logs_from_queue(self) -> None:
        if not _UI_INSTANCE:
            return
        log_widget = self.query_one(RichLog)
        # 1. 获取当前的滚动状态
        # scroll_offset.y 是当前顶部显示的行号
        # max_scroll_y 是能够滚动的最大行号
        # size.height 是控件的高度

        # 判断当前是否在最底部（允许 2 行的误差，防止差一点点没对齐就失效）
        # 如果 (最大滚动距离 - 当前滚动位置) < 2，说明用户正在看最新的内容
        distance_from_bottom = log_widget.max_scroll_y - log_widget.scroll_offset.y
        is_at_bottom = distance_from_bottom <= 2

        count = 0
        has_new_content = False

        while not _UI_INSTANCE.log_queue.empty() and count < 50:
            try:
                text_obj = _UI_INSTANCE.log_queue.get_nowait()
                # 写入日志，这会增加 max_scroll_y，但通常不会改变 scroll_offset.y
                log_widget.write(text_obj)
                count += 1
                has_new_content = True
            except queue.Empty:
                break

        # --- 核心修改结束 ---
        # 2. 只有当原本就在底部，且有新内容时，才强制滚到底部
        if has_new_content and is_at_bottom:
            log_widget.scroll_end(animate=False)

    def check_intervention_request(self) -> None:
        """Check if there's an intervention dialog request."""
        if not _UI_INSTANCE:
            return
        try:
            # Check for show request
            while not _UI_INSTANCE.intervention_show_queue.empty():
                message = _UI_INSTANCE.intervention_show_queue.get_nowait()
                # 动态创建对话框
                dialog = InterventionDialog(message, id="intervention_dialog")
                self.mount(dialog)
                if _UI_INSTANCE:
                    _UI_INSTANCE.log(
                        "debug",
                        f"[UI] Mounted intervention dialog with message: {message[:50]}",
                    )

            # Check for hide request
            while not _UI_INSTANCE.intervention_hide_queue.empty():
                _UI_INSTANCE.intervention_hide_queue.get_nowait()
                # 查找并移除对话框
                try:
                    dialog = self.query_one("#intervention_dialog", InterventionDialog)
                    dialog.remove()
                    if _UI_INSTANCE:
                        _UI_INSTANCE.log("debug", "[UI] Removed intervention dialog")
                except:
                    pass
        except Exception as e:
            if _UI_INSTANCE:
                _UI_INSTANCE.log(
                    "error", f"[UI] Error in check_intervention_request: {e}"
                )

    def update_status_bar(self) -> None:
        if not _UI_INSTANCE or not _UI_INSTANCE.status_callback:
            return
        try:
            s = _UI_INSTANCE.status_callback()
            status_text = (
                f" SYS: {s.get('system_status', '?')} | "
                f"TASK: {str(s.get('current_task', 'None'))[:40]} | "
                f"SKILL: {s.get('current_skill', 'None')}"
            )
            self.query_one("#status_bar", Label).update(status_text)
        except:
            pass

    def on_input_submitted(self, message: Input.Submitted) -> None:
        val = message.value.strip()
        if not val:
            return
        if _UI_INSTANCE:
            _UI_INSTANCE.log("user", f"> {val}")
            if val in ["/exit", "/quit"]:
                self.exit()
                return
            _UI_INSTANCE.input_queue.put(val)
        message.input.value = ""


class ConsoleUI:
    def __init__(self):
        global _UI_INSTANCE
        _UI_INSTANCE = self

        self.log_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.intervention_show_queue = (
            queue.Queue()
        )  # Queue for showing intervention dialog
        self.intervention_hide_queue = (
            queue.Queue()
        )  # Queue for hiding intervention dialog
        self.status_callback: Optional[Callable[[], dict]] = None
        self.shutdown_callback: Optional[Callable[[], None]] = None
        self.app: Optional[RobotBrainApp] = None

        self.category_colors = {
            "system": "blue",
            "brain": "magenta",
            "skill": "green",
            "user": "yellow",
            "error": "red",
            "info": "white",
            "wrarning": "orange1",
            "server-out": "dim white",
            "server-err": "red",
            "isaacsim": "bold green",  # <--- 新增：Server-Client 专用颜色
            "adapter": "cyan",  # <--- 新增：Adapter 专用颜色
            "skill-exec": "bright_blue",  # <--- 新增：Skill Executor Client 专用颜色
        }

        self.debug_log_file = open("ui_debug_fallback.log", "w", encoding="utf-8")
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        atexit.register(self._cleanup)

    # ... (其余方法保持不变) ...
    def _cleanup(self):
        if self.debug_log_file:
            self.debug_log_file.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        try:
            os.system("stty sane")
        except:
            pass

    def set_status_provider(self, callback: Callable[[], dict]):
        self.status_callback = callback

    def set_shutdown_callback(self, callback: Callable[[], None]):
        """Register a callback to run when UI exits (e.g., system.shutdown)."""
        self.shutdown_callback = callback

    def run(self, worker_task: Callable):
        # Silence terminal logging to keep Textual layout intact
        silence_terminal_logging()
        sys.stdout = self
        sys.stderr = self

        def safe_worker():
            time.sleep(0.5)
            try:
                worker_task()
            except Exception:
                import traceback

                self.log("error", f"CRASH: {traceback.format_exc()}")

        t = threading.Thread(target=safe_worker, daemon=True)
        t.start()
        self.app = RobotBrainApp()
        self.app.run()
        # When UI exits, ensure backend is stopped
        try:
            if self.shutdown_callback:
                self.shutdown_callback()
        except Exception:
            pass
        self._cleanup()

    def write(self, text: str):
        if text.strip():
            self.log("info", text.strip())

    def flush(self):
        pass

    def isatty(self):
        return False

    def log(self, category: str, message: str):
        try:
            timestamp = time.strftime("%H:%M:%S")
            self.debug_log_file.write(f"[{timestamp}] [{category.upper()}] {message}\n")
            self.debug_log_file.flush()
        except:
            pass

        try:
            # [修改 3] 统一转换为小写匹配颜色
            color = self.category_colors.get(category.lower(), "white")
            safe_msg = str(message).replace("[", "\[")
            # 在消息前加分类 Tag，让区分更明显
            prefix = category.upper()
            text_obj = Text.from_markup(
                f"[{color}][{timestamp}] [{prefix}] {safe_msg}[/]"
            )
        except:
            text_obj = Text(f"[{timestamp}] [{category}] {message}")

        self.log_queue.put(text_obj)

    def show_intervention_dialog(self, reason: str):
        """
        Show the intervention dialog with the given reason.
        Called when HumanIntervention skill is activated.

        Args:
            reason: The reason why the agent needs human intervention.
        """
        self.intervention_show_queue.put(reason)
        self.log("system", f"🤖 Agent requests human intervention: {reason}")

    def hide_intervention_dialog(self):
        """Hide the intervention dialog."""
        self.intervention_hide_queue.put(True)

    def start(self):
        pass

    def stop(self):
        pass


global_console = ConsoleUI()
