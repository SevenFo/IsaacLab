import queue
from ..core.types import SkillType, ExecutionMode, Action, BaseSkill
from ..core.skill_manager import skill_register

from ..ui.console import global_console


def log(msg):
    global_console.log("skill", msg)


@skill_register(
    name="human_intervention",
    skill_type=SkillType.POLICY,
    execution_mode=ExecutionMode.STEPACTION,
    timeout=3600.0,
    criterion={"successed": "Human confirmed", "failed": "Human aborted"},
    requires_env=True,
    enable_monitoring=False,
)
class HumanIntervention(BaseSkill):
    """Pauses the robot and waits for human input when the task cannot be completed autonomously.
    Use this skill when: you encounter an unsolvable problem, need human guidance, or require manual intervention.
    The robot will hold its current position while waiting. Human can provide feedback to guide replanning.
    Args:
        reason (str, optional): The reason why human intervention is needed. Will be displayed to the user."""

    def __init__(self, policy_device: str = "cuda", **running_params):
        super().__init__()
        # [关键] 获取 Executor 注入的队列
        self.input_queue = running_params.get("_input_queue")
        if self.input_queue is None:
            self.input_queue = queue.Queue()  # Fallback 用于测试
        # 获取请求干预的原因
        self.reason = running_params.get(
            "reason", "Agent needs your guidance to proceed."
        )

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        # 显示干预对话框
        # Force reason to string and log for debugging
        reason_str = str(self.reason)
        global_console.log("skill", f"Showing intervention dialog: {reason_str}")
        global_console.show_intervention_dialog(reason_str)

        log("Robot paused. Waiting for input...")
        log("  - Type 'c' to continue")
        log("  - Type 'f' to mark as failed")
        log("  - Type '/reset_box' to reset the box/spanner (skill keeps waiting)")
        log("  - Type any other text to finish and send feedback")

        # 不需要构建 zero_action，Server 端会自动处理
        return None

    def cleanup(self):
        """Cleanup when skill ends - hide the intervention dialog."""
        global_console.hide_intervention_dialog()
        super().cleanup() if hasattr(super(), "cleanup") else None

    def select_action(self, obs_dict: dict) -> Action:
        """
        选择动作：检查用户输入并返回对应的 Action。

        返回值规范：
        - 等待时: Action(data=None, metadata={"update_only": True, "info": "waiting", ...})
        - 完成时: Action(data=None, metadata={"update_only": True, "info": "finished", ...})
        - 失败时: Action(data=None, metadata={"update_only": True, "info": "error", ...})

        执行器会根据 "update_only" 标记来决定调用 env_proxy.update() 而非 step()。
        """
        # 1. 非阻塞检查队列
        try:
            command = self.input_queue.get_nowait()
            return self._process_command(command)
        except queue.Empty:
            pass

        # 2. 如果没有指令，返回 update_only Action
        # 执行器会调用 env_proxy.update()，Server 自动使用 zero_action
        return Action(
            data=None,  # 不需要传递 action 数据
            metadata={
                "update_only": True,  # 标记：只需 update 不需 step
                "info": "waiting",
                "reason": "waiting for human input",
            },
        )

    def _process_command(self, command: str) -> Action:
        """
        处理用户输入的命令。

        支持的输入:
        - 'c', 'continue', 'done', 'success', 'ok', 'yes' -> 技能完成 (COMPLETED)
        - 'f', 'fail', 'abort', 'no' -> 技能失败 (FAILED)
        - 其他文字 -> 作为反馈触发 replan (INTERRUPTED with feedback)
        """
        log(f"Received interaction command: {command}")
        cmd = command.strip()
        cmd_lower = cmd.lower()

        # 用户输入 'c'，说明编辑完成了，或者想继续
        if cmd_lower in ["c", "continue", "done", "success", "ok", "yes"]:
            return Action(
                data=None,
                metadata={
                    "update_only": True,
                    "info": "finished",  # "finished" 对应 COMPLETED
                    "reason": "Human marked success",
                },
            )

        # 用户输入 'f'，标记为失败（会触发 Replan）
        elif cmd_lower in ["f", "fail", "abort", "no"]:
            return Action(
                data=None,
                metadata={
                    "update_only": True,
                    "info": "error",  # "error" 对应 FAILED
                    "reason": "Human marked failed",
                },
            )

        # 用户输入 reset_box，代表已执行环境重置但仍需继续等待确认
        elif cmd_lower in ["reset_box", "/reset_box"]:
            return Action(
                data=None,
                metadata={
                    "update_only": True,
                    "info": "success",
                    "reason": "reset_box applied; still waiting",
                },
            )

        # 其他输入作为反馈文本，结束技能（完成），携带人类反馈
        else:
            log(f"Human feedback received: {cmd}, finishing intervention...")
            return Action(
                data=None,
                metadata={
                    "update_only": True,
                    "info": "finished",  # 结束技能
                    "reason": f"Human feedback: {cmd}",
                    "human_feedback": cmd,
                },
            )
