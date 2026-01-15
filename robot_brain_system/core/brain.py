"""
Brain component that uses Qwen VL for task planning and monitoring.
"""

import copy
import time
import json
import numpy as np
import os
import io
from PIL import Image
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from functools import partial

from robot_brain_system.core.types import (
    Task,
    SkillPlan,
    SystemStatus,
    Observation,
    SystemState,
    SkillStatus,
    SkillStep,
)
from robot_brain_system.utils import extract_json_from_text
from robot_brain_system.core.model_adapters_v2 import (
    TransformersAdapter,
    LMDeployAdapter,
    VLLMAdapter,
    OpenAIAdapter,
)
from robot_brain_system.ui.console import global_console
from robot_brain_system.utils.retry_utils import retry

if TYPE_CHECKING:
    from robot_brain_system.core.skill_manager import SkillRegistry


@dataclass
class BrainState:
    """
    Brain 的内部工作状态。

    注意：Brain 不维护系统级 status，系统状态由 System 统一管理。
    Brain 只负责任务规划和监控，通过返回值通知 System 状态变化。
    """

    current_task: Optional[Task] = None
    current_plan: Optional[SkillPlan] = None
    current_skill_index: int = 0
    last_monitoring_time: float = 0.0
    error_message: Optional[str] = None

    # Task-level memory: 任务维度的长期记忆摘要
    # 记录任务执行的宏观进展，在 summary/replan 时作为上下文提供
    task_memory: str = ""


@dataclass
class BrainMemory:
    """Memory of the brain for storing past experiences and knowledge."""

    history: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def get_default_system_prompt(
        prompt: str = "You are a helpful assistant.",
    ):
        return {
            "role": "system",
            "content": [{"type": "text", "text": prompt}],
        }

    def print(self, msg: str):
        """Utility print function for consistent logging."""
        global_console.log("brain", msg)

    def add_system_prompt(self, prompt: str = "You are a helpful assistant."):
        """Add a system prompt to the memory."""

        self.history.append(self.get_default_system_prompt(prompt=prompt))

    def add_user_input(self, contents: List[str | Image.Image | list[Image.Image]]):
        """Add user input to the memory."""
        item = {
            "role": "user",
            "content": [],
        }
        for content in contents:
            if isinstance(content, str):
                item["content"].append({"type": "text", "text": content})
            elif isinstance(content, Image.Image):
                item["content"].append({"type": "image", "image": content})
            elif isinstance(content, list) and all(
                isinstance(img, Image.Image) for img in content
            ):
                item["content"].append({"type": "video", "video": content, "fps": 5})
            else:
                raise ValueError(
                    f"Unsupported content type: {type(content)}. Must be str, Image.Image, or list of Image.Image."
                )
        self.history.append(item)

    def add_assistant_output(self, output: str):
        """Add assistant output to the memory."""
        self.history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output}],
            }
        )

    def fetch_history(
        self, last_n: int = 5, prune_multimedia: bool = True
    ) -> List[Dict[str, Any]]:
        """
        获取系统提示以及最后 N 轮对话。

        新增了一个参数 `prune_multimedia` 来优化多模态历史记录的处理，减少 token 消耗。

        Args:
            last_n (int): 获取的对话轮数。一轮包含一个用户输入和一个助手输出。
            prune_multimedia (bool): 是否启用历史多媒体信息消除功能。
                - 如果为 True，则历史对话中的所有多媒体内容（图片、视频）将被
                  替换为文本占位符（如 '<image>'），但保留最新的用户输入中的
                  所有多媒体内容。
                - 如果为 False（默认），则返回原始的、包含所有多媒体信息的历史记录。

        Returns:
            List[Dict[str, Any]]: 格式化后的历史记录，可供模型使用。
        """
        if not self.history:
            return []

        # 1. 分离系统提示和对话历史
        system_prompt = self.history[0:1]
        conversation_history = self.history[1:]

        # 你的断言逻辑稍作调整，以处理偶数长度（例如刚添加完助手回复）
        if len(conversation_history) > 0 and len(conversation_history) % 2 == 0:
            num_entries_to_fetch = last_n * 2
        else:  # 奇数长度，最后一条是当前用户输入
            num_entries_to_fetch = last_n * 2 + 1

        history_slice = conversation_history[-num_entries_to_fetch:]

        # 2. 如果不启用剪枝，或者历史记录不足以进行剪枝，则按原样返回
        if not prune_multimedia or not history_slice:
            return system_prompt + history_slice

        # 3. 如果启用剪枝，处理历史记录
        # 使用深拷贝 (deepcopy) 来确保不会修改原始的 self.history
        processed_history = copy.deepcopy(history_slice)

        # 遍历除最后一个条目（即最新的用户输入）之外的所有历史记录
        for item in processed_history[:-1]:
            if "content" not in item:
                continue

            new_content_list = []
            text_buffer = []

            for content_part in item["content"]:
                if content_part["type"] == "text":
                    text_buffer.append(content_part["text"])
                else:
                    # 对于非文本类型，添加占位符
                    text_buffer.append(f"<{content_part['type']}>")

            # 将所有文本和占位符合并成一个单一的文本条目，进一步节省空间
            if text_buffer:
                new_content_list.append({"type": "text", "text": " ".join(text_buffer)})

            item["content"] = new_content_list

        # 4. 合并系统提示、处理过的历史以及（可能存在的）未经处理的最新用户输入
        return system_prompt + processed_history

    def clear(self):
        """Clear the memory."""
        self.history = []
        self.print("[BrainMemory] Memory cleared.")

    def format_memory_content(self) -> List:
        self.print(f"[QwenVLBrain] Formatting memory with {len(self.history)} entries")
        chat_content = []
        chat_content.append("## memory content:\n\n")
        for item in self.history:
            # take only one image from video to reduce GPU memory useage
            if item["role"] == "system":
                continue
            chat_content.append(f"### {item['role']}:\n")
            text_segment = ""
            for content in item["content"]:
                if content["type"] == "text":
                    text_segment = text_segment + content["text"]
                else:
                    if text_segment:
                        chat_content.append(text_segment)
                        text_segment = ""  # clear
                    if content["type"] == "video":
                        chat_content.append(content["video"][-1])
                    elif content["type"] == "image":
                        chat_content.append(content["image"])
                    else:
                        self.print(
                            f"[QwenVLBrain] format memory content encounted with unsupported content type: {content}"
                        )
            if text_segment:
                chat_content.append(text_segment)
                text_segment = ""  # clear
            chat_content.append("\n---\n")
        return chat_content

    def extract_images(self, n_max=-1):
        self.print(
            f"[QwenVLBrain] Extract images from memory with {len(self.history)} entries"
        )
        content_images = []
        for item in self.history:
            # take only one image from video to reduce GPU memory useage
            if item["role"] == "system":
                continue
            for content in item["content"]:
                if content["type"] == "text":
                    continue
                elif content["type"] == "video":
                    content_images.append(content["video"][-1])
                elif content["type"] == "image":
                    content_images.append(content["image"])
                else:
                    continue
        if n_max <= 0 or len(content_images) <= n_max:
            return content_images
        last_image = content_images[-1]
        remaining_images = content_images[:-1]
        if n_max == 1:
            return [last_image]
        indices = np.linspace(0, len(remaining_images) - 1, n_max - 1, dtype=int)
        resampled_images = [remaining_images[i] for i in indices] + [last_image]
        return resampled_images


class QwenVLBrain:
    """
    Brain component that uses Qwen VL for high-level task planning and monitoring.

    This component:
    1. Parses user input into structured tasks
    2. Plans skill sequences to complete tasks
    3. Monitors skill execution and can interrupt/modify as needed
    4. Makes decisions based on visual and task feedback
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.skill_registry: Optional["SkillRegistry"] = None
        self.state = BrainState()

        # Qwen VL configuration
        self.qwen_config = config.get("adapter", {})

        # Monitoring configuration
        self.skill_monitoring_interval = config.get(
            "skill_monitoring_interval", 1.0
        )  # seconds
        self.max_retries = config.get("max_retries", 3)
        self.visualize = config.get("visualize", False)
        self.log_path = config.get("log_path", "./logs")
        self.monitor_waiting_times = 0
        self.monitor_memory = BrainMemory()
        self.replan_memory = BrainMemory()
        self.replan_memory.add_system_prompt()
        self.current_skill_monitor_time = 0
        self.FRAME_JUMP = 15
        self.FRAME_TOTAL = 3

    def print(self, msg: str):
        """Utility print function for consistent logging."""
        global_console.log("brain", msg)

    def initialize(self):
        """Initialize the brain component."""
        self.print("[QwenVLBrain] Initializing...")
        # Initialize model adapter

        try:
            self._initialize_model_adapter()
        except Exception as e:
            self.print(
                f"[QwenVLBrain] Failed to initialize model adapter: {type(e).__name__}: {e}"
            )
            import traceback

            traceback.print_exc()
            self.print("[QwenVLBrain] Falling back to mock implementation")
            self.adapter_type = "mock"

        self.print(f"[QwenVLBrain] Initialized with adapter: {self.adapter_type}")

    def _initialize_model_adapter(self):
        """根据配置初始化新的、标准化的模型适配器。"""
        self.adapter_type = self.qwen_config.get(
            "adapter_type", "mock"
        )  # "qwen_vl", "openai", or "mock"
        # 从主配置中获取模型和 API 相关的通用配置
        model_path = self.qwen_config.get("model_path")
        device = self.qwen_config.get("device", "auto")
        model_name = self.qwen_config.get("model")  # for OpenAI
        self.insert_think_prompt = True
        if not self.insert_think_prompt:
            self.print(
                "[QwenVLBrain] Think prompt injection disabled for this adapter/model combination."
            )
        self.max_tokens = self.qwen_config.get("max_tokens", 512)

        self.print(
            f"[QwenVLBrain] self.insert_think_prompt: {self.insert_think_prompt}"
        )
        try:
            if self.adapter_type == "transformers":
                self.model_adapter = TransformersAdapter(
                    model_path=model_path,
                    device=device,
                    **self.qwen_config.get("transformers_adapter_args", {}),
                )
            elif self.adapter_type == "vllm":
                # vLLM 的特定参数也可以从 config 传入
                self.model_adapter = VLLMAdapter(
                    model_path=model_path,
                    **self.qwen_config.get("vllm_adapter_args", {}),
                )
            elif self.adapter_type == "lmdeploy":
                # LMDeploy 的特定参数
                self.model_adapter = LMDeployAdapter(
                    model_path=model_path,
                    **self.qwen_config.get("lmd_adapter_args", {}),
                )
            elif self.adapter_type == "openai":
                self.model_adapter = OpenAIAdapter(
                    model_name=model_name,
                    **self.qwen_config.get("args", {"api_key": "", "base_url": ""}),
                )
            else:
                # 保留 mock 作为备用
                self.print("[QwenVLBrain] 未知的 adapter_type，将使用 mock 实现。")
                self.adapter_type = "mock"
                self.model_adapter = None

            if self.model_adapter:
                self.print(f"[QwenVLBrain] 成功初始化适配器：{self.adapter_type}")

        except Exception as e:
            import traceback

            self.print(f"[QwenVLBrain] 初始化模型适配器失败：{e}")
            self.print(traceback.format_exc())
            self.print("[QwenVLBrain] 回退到 mock 实现。")
            self.adapter_type = "mock"
            self.model_adapter = None

    def set_skill_registry(self, skill_registry: "SkillRegistry"):
        """Set the skill registry for planning."""
        self.skill_registry = skill_registry
        self.print("[QwenVLBrain] Connected to skill registry")

    def set_system_state(self, state: SystemState):
        self.system_state = state

    # ========== 状态查询辅助方法 ==========

    def has_task(self) -> bool:
        """检查是否有当前任务"""
        return self.state.current_task is not None

    def has_plan(self) -> bool:
        """检查是否有执行计划"""
        return self.state.current_plan is not None

    def has_pending_skills(self) -> bool:
        """检查是否还有待执行的技能"""
        if not self.has_plan():
            return False
        return self.state.current_skill_index < len(self.state.current_plan.steps)

    def is_plan_complete(self) -> bool:
        """检查计划是否全部完成"""
        if not self.has_plan():
            return False
        return self.state.current_plan.is_complete()

    # ========== 计划管理方法 (Brain 作为 Source of Truth) ==========

    def get_current_plan(self) -> Optional[SkillPlan]:
        """获取当前执行计划 (只读引用)"""
        return self.state.current_plan

    def get_current_task(self) -> Optional[Task]:
        """获取当前任务 (只读引用)"""
        return self.state.current_task

    def get_current_skill_index(self) -> int:
        """获取当前技能索引"""
        return self.state.current_skill_index

    def mark_skill_status(self, index: int, status: SkillStatus):
        """
        标记指定技能的状态。System 通过此方法更新技能状态，
        而不是直接操作 plan。

        Args:
            index: 技能在计划中的索引
            status: 新状态
        """
        if not self.state.current_plan:
            self.print("[QwenVLBrain] Warning: No plan to mark status")
            return

        self.state.current_plan.mark_status(index, status)
        self.print(f"[QwenVLBrain] Marked skill {index} as {status.name}")

    def advance_skill_index(self):
        """
        前进到下一个技能索引。
        通常在技能成功完成后由 System 调用。
        """
        if self.state.current_plan:
            next_info = self.state.current_plan.get_next_pending_skill_with_index()
            if next_info:
                self.state.current_skill_index = next_info[0]
            else:
                self.state.current_skill_index = len(self.state.current_plan.steps)

    # ========================================

    def parse_task(
        self, instruction: str, image_data: Optional[Image.Image] = None
    ) -> Task:
        """
        Parse natural language instruction into a structured task.

        Args:
            instruction: Natural language task description
            image_data: Optional base64 encoded image

        Returns:
            Structured Task object
        """
        try:
            # Create basic task structure
            task = Task(
                id=f"task_{int(time.time())}",
                description=instruction,
                image=image_data,
                priority=1,
                metadata={"created_at": time.time()},
            )

            self.print(f"[QwenVLBrain] Parsed task: {task.id} - {instruction}")
            return task

        except Exception as e:
            raise RuntimeError(f"Failed to parse task: {e}")

    def plan_task(self, task: Task) -> SkillPlan:
        """
        Create a skill execution plan for the given task.

        Args:
            task: The task to plan for

        Returns:
            SkillPlan with sequence of skills and parameters
        """
        try:
            if not self.skill_registry:
                raise RuntimeError("Skill registry not available")

            # Use Qwen VL to analyze task and create plan
            plan = self._query_qwen_for_plan(
                task, self.skill_registry.get_skill_descriptions()
            )
            return plan

        except Exception as e:
            raise RuntimeError(f"Failed to create plan: {e}")

    def summary_skill_execution(
        self, skill_info: dict, obs: Observation, only_image=True
    ):
        # TODO 只提供 image 不提供 text 试试！
        # self.monitor_memory.add_user_input(
        #     contents=[f"skill execution result: {skill_info['result']}"]
        # )
        camera_side_image = Image.fromarray(
            obs.data["policy"]["inspector_side"][0].cpu().numpy()
        )
        camera_front_image = Image.fromarray(
            obs.data["policy"]["inspector_top"][0].cpu().numpy()
        )
        content = []
        if only_image:
            memory_content = self.monitor_memory.extract_images(n_max=8)
        else:
            memory_content = self.monitor_memory.format_memory_content()
        content.append(
            (
                "The robot are try to orchestrate skills to accomplish a task.\n\n"
                "## Skill Execution Context Info:\n"
                # f"Original Task: {self.state.current_task.description}\n"  # 加入 original task 描述 会让 llm 认为是在判断 original task 的执行结果 而不是 task 的执行情况
                f"Current Skill: {skill_info['name']}\n"
                f"Skill Description: {skill_info['description']}\n"
                f"Skill Parameters: {skill_info['parameters']}\n"
                f"Skill Criterion: {skill_info['criterion']}\n"
                f"Skill Execution Result (reported by skill itself): {skill_info['result']}, Reason: {skill_info['status_info'] if skill_info['result'] != 'completed' else 'N/A'}\n\n"
            )
        )
        content.append(
            "## Visual Evidence of Skill Execution:\n"
            "The following images are captured during the skill execution. "
            "Please analyze them to determine the true outcome of the skill.\n\n"
        )
        content.extend(
            memory_content if len(memory_content) else ["No visual evidence available."]
        )
        content.append("## Current Scene Images:\n")
        content.append("### left side view:\n")
        content.append(camera_side_image)
        content.append("### right side view:\n")
        content.append(camera_front_image)
        content.append(
            (
                "## Your Task: Analyze and Summarize the Execution\n"
                "Your primary goal is to determine the TRUE outcome of the last skill by analyzing the visual evidence (memory content), "
                "even if it contradicts the self-reported 'Skill Execution Result'. And give a detailed analysis of the skill execution.\n\n"
                "## Focus Points:\n"
                # The re-check question is now the primary instruction.
                f"1. **Verdict on the Last Skill:** Based *only* on the images, did the skill truly succeed or fail? Compare what you see against the goal: '{skill_info['criterion']['successed']}'. State your conclusion clearly (e.g., 'Verdict: The skill succeeded despite the timeout report.').\n"
                "2. **Not Succeed Reason:** If the skill did not succeed, provide your analysis of why it failed.\n"
                "3. **Scene Transformation:** How did the scene change from the beginning to the end of the skill execution?\n"
                "4. **Current State:** Describe the final state of the relevant objects in the scene.\n"
                "5. **Reflections:** Were there any unexpected movements or issues during the execution?\n"
                "\n## Output Template:\n"
                "[Verdict] Your explicit conclusion on the skill's true outcome.\n"
                "[Scene Change] Your description of the changes.\n"
                "[Current State] Your description of the final scene.\n"
                "[Reflections] Any other relevant observations."
            )
        )
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(camera_side_image)
        axs[0].set_title("side view")
        axs[0].axis("off")

        axs[1].imshow(camera_front_image)
        axs[1].set_title("front view")
        axs[1].axis("off")

        plt.tight_layout()
        out_path = os.path.join(
            self.log_path,
            f"{time.time()}_skill_execution_summary_{skill_info['name']}.png",
        )
        plt.savefig(out_path)
        plt.close(fig)

        sum_memory = BrainMemory()
        sum_memory.add_system_prompt("You are a professional dialogue summarizer.")
        sum_memory.add_user_input(contents=content)

        if self.model_adapter is None:
            # mock
            return "N/A"

        response_text, _ = self.model_adapter.generate(
            sum_memory.history, thinking=True
        )

        # 更新任务级长期记忆
        self.update_task_memory(skill_info, response_text)

        return response_text

    @retry(
        exceptions_to_retry=(ValueError, RuntimeError),
        logger_func=partial(global_console.log, "brain"),
    )
    def replan_task(
        self,
        task: Task,
        current_plan: SkillPlan,
        skill_history: list[dict],
        observation: Observation,
        human_feedback: str = "",
    ):
        """
        根据当前执行情况重新规划任务。

        注意：此方法不修改系统状态，只返回新计划。
        状态管理由 System 负责。

        Args:
            task: The task to plan for
            current_plan: 当前正在执行的计划
            skill_history: 技能执行历史
            observation: 当前观测
            human_feedback: 人类反馈

        Returns:
            SkillPlan with sequence of skills and parameters
        """
        try:
            if not self.skill_registry:
                raise RuntimeError("Skill registry not available")

            try:
                # 移除：self.state.status = SystemStatus.THINKING

                # 1. Prepare System Prompt (Static Instructions)
                system_prompt = self._format_system_prompt_for_replanning()

                # 2. Prepare User Prompt (Dynamic Context)
                text_prompt = self._format_user_prompt_for_replanning(
                    task, current_plan, skill_history, observation, human_feedback
                )
                camera_side_image = Image.fromarray(
                    observation.data["policy"]["inspector_side"][0].cpu().numpy()
                )
                camera_front_image = Image.fromarray(
                    observation.data["policy"]["inspector_top"][0].cpu().numpy()
                )
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(camera_side_image)
                axs[0].set_title("left side view")
                axs[0].axis("off")
                axs[1].imshow(camera_front_image)
                axs[1].set_title("right front view")
                axs[1].axis("off")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                # 3. 将缓冲区指针重置到开头
                buf.seek(0)
                # 4. 使用 PIL 从缓冲区中打开图像
                current_image = Image.open(buf).convert("RGB")
                # 5. 关闭缓冲区和 Matplotlib Figure 以释放内存
                buf.close()
                plt.close(fig)
                if self.visualize:
                    # 直接保存 PIL 图像，不再需要 matplotlib
                    current_image.save(
                        os.path.join(
                            self.log_path,
                            f"{time.time()}_{len(self.system_state.plan_history)}_replan_{task.description}_input.png",
                        )
                    )

                # Generate response
                # Use a fresh local memory for this replan session
                replan_session_memory = BrainMemory()
                replan_session_memory.add_system_prompt(system_prompt)
                replan_session_memory.add_user_input(
                    contents=[text_prompt, current_image]
                )

                self.print(f"[QwenVLBrain] 开始为任务进行再规划：{task.description}")
                if self.model_adapter is None:
                    response_text = "```json\n[\n]\n```\n\n"
                    reasoning_content = (
                        "brain model is unavaliable, return mock response"
                    )
                else:
                    response_text, reasoning_content = self.model_adapter.generate(
                        history=replan_session_memory.fetch_history(
                            last_n=0, prune_multimedia=True
                        ),
                        max_tokens=self.max_tokens,
                        thinking=True,
                    )  # type: ignore
                # Parse the response to extract skill plan

                # ----------------- Old implementation (to be removed) -----------------
                # operations = self._parse_json_response(response_text)
                # # 移除：self.state.status = SystemStatus.EXECUTING
                # if not operations:
                #     self.print(
                #         "[QwenVLBrain] No operations suggested by LLM. Plan remains unchanged."
                #     )
                # else:
                #     # Step 0: Handle modify_task operation first (changes task description)
                #     modify_task_ops = [
                #         op for op in operations if op.get("operation") == "modify_task"
                #     ]
                #     if modify_task_ops:
                #         modify_op = modify_task_ops[0]  # Only use the first one
                #         new_description = modify_op.get("new_description", "")
                #         modify_reason = modify_op.get("reason", "Human feedback")
                #         if new_description and self.state.current_task:
                #             old_description = self.state.current_task.description
                #             self.state.current_task.description = new_description
                #             self.print(
                #                 f"[QwenVLBrain] Task modified: '{old_description}' -> '{new_description}' (Reason: {modify_reason})"
                #             )
                #             # Update task memory to reflect the goal change
                #             self.update_task_memory(
                #                 {
                #                     "old_description": old_description,
                #                     "new_description": new_description,
                #                     "reason": modify_reason,
                #                 },
                #                 reasoning_content,
                #                 update_type="task_modified",
                #             )
                #         # Remove modify_task from operations list for subsequent processing
                #         operations = [
                #             op
                #             for op in operations
                #             if op.get("operation") != "modify_task"
                #         ]

                #     original_steps = list(current_plan.steps)
                #     new_steps = []

                #     inserts_after = {
                #         op["index"]: op
                #         for op in operations
                #         if op["operation"] == "insert"
                #     }
                #     deletes_at = {
                #         op["index"] for op in operations if op["operation"] == "delete"
                #     }
                #     updates_at = {
                #         op["index"]: op
                #         for op in operations
                #         if op["operation"] in ["update_status", "retry", "modify"]
                #     }

                #     for i, original_step in enumerate(original_steps):
                #         # Step 1: Decide whether to include the original step at this index
                #         if i not in deletes_at:
                #             step_to_add = original_step
                #             # Apply any non-structural changes (update, modify, retry)
                #             if i in updates_at:
                #                 op = updates_at[i]
                #                 op_type = op.get("operation")

                #                 if op_type == "retry":
                #                     step_to_add.status = SkillStatus.PENDING
                #                 elif op_type == "update_status":
                #                     try:
                #                         new_status_str = op.get(
                #                             "new_status", ""
                #                         ).upper()
                #                         step_to_add.status = SkillStatus[new_status_str]
                #                     except KeyError:
                #                         self.print(
                #                             f"[QwenVLBrain] Warning: Invalid status '{op.get('new_status')}' for index {i}. Skipping update."
                #                         )
                #                 elif op_type == "modify":
                #                     new_name = op.get("new_method")
                #                     new_params = op.get("new_params")
                #                     if new_name:
                #                         step_to_add.name = new_name
                #                     if new_params is not None:
                #                         step_to_add.params = new_params
                #                     step_to_add.status = (
                #                         SkillStatus.PENDING
                #                     )  # Reset on modify

                #             new_steps.append(step_to_add)

                #         # Step 2: Check if a new skill should be inserted AFTER this original step
                #         if i in inserts_after:
                #             op = inserts_after.pop(i)
                #             new_skill = SkillStep(
                #                 name=op["method"], params=op.get("params", {})
                #             )
                #             new_steps.append(new_skill)
                #     new_steps.extend(
                #         [
                #             SkillStep(name=op["method"], params=op.get("params", {}))
                #             for idx, op in sorted(inserts_after.items())
                #         ]
                #     )  # 防止遗漏在最后插入的技能
                #     current_plan.steps = new_steps
                #     self.print(
                #         "[QwenVLBrain] Successfully applied all operations and reconstructed the plan."
                #     )
                #     self.print("[QwenVLBrain] current_plan after replan:")
                #     self.print(current_plan.pretty_print())
                # --------------------------------------------------------------------

                operations = self._parse_json_response(response_text)

                if not operations:
                    self.print(
                        "[QwenVLBrain] No operations suggested by LLM. Plan remains unchanged."
                    )
                else:
                    # --- Step 1: Handle modify_task operation first ---
                    # Check if the first operation is a task modification
                    if operations[0].get("operation") == "modify_task":
                        modify_op = operations.pop(0)  # Remove and process it
                        new_description = modify_op.get("new_description", "")
                        modify_reason = modify_op.get("reason", "Human feedback")

                        if new_description and self.state.current_task:
                            old_description = self.state.current_task.description
                            self.state.current_task.description = new_description
                            self.print(
                                f"[QwenVLBrain] Task modified: '{old_description}' -> '{new_description}' (Reason: {modify_reason})"
                            )
                            # Update task memory to reflect the goal change
                            self.update_task_memory(
                                {
                                    "old_description": old_description,
                                    "new_description": new_description,
                                    "reason": modify_reason,
                                },
                                reasoning_content,
                                update_type="task_modified",
                            )

                    # --- Step 2: Reconstruct the Plan (Overwrite Strategy) ---
                    # If operations is empty after removing modify_task, we assume no plan changes needed
                    if not operations:
                        self.print(
                            "[QwenVLBrain] Task updated, but no plan steps provided. Keeping original plan."
                        )
                    else:
                        # The logic: Find the index where the new plan starts.
                        # The LLM returns [{"step": N, ...}, {"step": N+1, ...}]
                        # We keep steps 0 to N-1 from the original plan, and append the new list.

                        try:
                            # Get the start index from the first step in the new list
                            first_new_step_idx = operations[0].get("step")

                            if first_new_step_idx is None:
                                raise ValueError(
                                    "JSON objects must contain a 'step' field."
                                )

                            # Safety Check: Index shouldn't be negative
                            first_new_step_idx = max(0, int(first_new_step_idx))

                            # 1. Truncate the original plan up to the start index
                            # If start_index is 2, we keep steps 0 and 1.
                            current_steps = list(current_plan.steps)

                            # Ensure we don't slice out of bounds (though Python handles this gracefully)
                            if first_new_step_idx <= len(current_steps):
                                kept_steps = current_steps[:first_new_step_idx]
                            else:
                                # If LLM suggests starting at step 10 but we only have 5, just append to end
                                kept_steps = current_steps
                                self.print(
                                    f"[QwenVLBrain] Warning: LLM suggested step {first_new_step_idx}, but plan len is {len(current_steps)}. Appending."
                                )

                            # 2. Build the new skill steps
                            new_skill_steps = []
                            for op in operations:
                                method_name = op.get("method")
                                params = op.get("params", {})
                                if not method_name:
                                    continue

                                # Create new skill step
                                new_step = SkillStep(name=method_name, params=params)
                                # Newly proposed steps are typically PENDING
                                new_step.status = SkillStatus.PENDING
                                new_skill_steps.append(new_step)

                            # 3. Merge: Kept Steps + New Steps
                            current_plan.steps = kept_steps + new_skill_steps

                            self.print(
                                f"[QwenVLBrain] Plan updated. Rewrote steps starting from index {first_new_step_idx}."
                            )
                            self.print("[QwenVLBrain] current_plan after replan:")
                            self.print(current_plan.pretty_print())

                        except Exception as e:
                            self.print(
                                f"[QwenVLBrain] Error applying plan updates: {e}"
                            )

                # The plan is now modified in-place
                assert id(self.state.current_plan) == id(current_plan), (
                    "this should not be happened as we use reference"
                )
                next_skill_info = current_plan.get_next_pending_skill_with_index()

                # The plan is now modified in-place
                assert id(self.state.current_plan) == id(current_plan), (
                    "this should not be happened as we use reference"
                )
                next_skill_info = current_plan.get_next_pending_skill_with_index()

                if next_skill_info is None:
                    self.print(
                        "[QwenVLBrain] No more pending skills in the plan. Task is complete. Next loop will interrupt the task."
                    )
                    return None

                self.state.current_skill_index = (
                    next_skill_info[0] if next_skill_info else len(current_plan.steps)
                )

                self.initial_monitor()  # Re-initialize monitoring for the current/new skill

                self.print("[QwenVLBrain] Successfully replanned task. New plan state:")
                self.print(current_plan.pretty_print())

                # Update task memory with replan info
                changes_summary = f"Applied {len(operations)} operations: " + ", ".join(
                    [
                        f"{op.get('operation')} (idx {op.get('index')})"
                        for op in operations
                    ]
                )
                self.update_task_memory(
                    {"changes": changes_summary},
                    reasoning_content,
                    update_type="replan",
                )

                return current_plan

            except Exception as e:
                import traceback

                self.print(f"[QwenVLBrain] Error in planning: {e}")
                traceback.print_exc()
                raise RuntimeError(f"Failed to create plan: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to create plan: {e}")

    def execute_task(self, task: Task) -> SkillPlan:
        """
        开始执行任务，生成执行计划。

        注意：此方法不设置系统状态，只负责规划和存储任务信息。
        系统状态由 System 统一管理。

        Args:
            task: Task to execute

        Returns:
            The execution plan
        """
        try:
            # 移除：self.state.status = SystemStatus.THINKING
            self.state.current_task = task

            # Create execution plan
            plan = self.plan_task(task)
            self.state.current_plan = plan
            self.state.current_skill_index = 0

            # Initialize task memory
            self.update_task_memory(
                {"plan_summary": plan.pretty_print()},
                "",
                update_type="initial_plan",
            )

            # 移除：self.state.status = SystemStatus.EXECUTING
            self.initial_monitor()
            self.print(f"[QwenVLBrain] Started executing task: {task.description}")
            return plan

        except Exception as e:
            # 移除：self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            raise  # 让 System 处理错误状态

    def initial_monitor(self):
        assert self.state.current_task is not None

        current_plan = self.state.current_plan
        assert current_plan is not None

        skill_info = self.get_next_skill()
        assert skill_info is not None, (
            f"Skill info not found for index {self.state.current_skill_index}"
        )

        self.monitor_memory.clear()

        self.print(
            f"[QwenVLBrain] Monitor initialized with skill: {skill_info['name']}"
        )
        self.monitor_memory.add_system_prompt(
            self._format_system_prompt_for_monitoring(
                self.state.current_task, skill_info, None
            )
        )
        self.current_skill_monitor_time = 0

    def get_next_skill(self) -> Optional[Dict[str, Any]]:
        """
        Get the next skill to execute in the current plan.

        Returns:
            Dict with skill name and parameters, or None if plan complete
        """
        if not self.state.current_plan:
            return None

        next_skill_info = self.state.current_plan.get_next_pending_skill_with_index()

        if not next_skill_info:
            # No pending skills left in the plan
            return None

        index, skill_step = next_skill_info

        # Update the brain's internal index tracker
        self.state.current_skill_index = index

        # Now, get the full skill definition from the registry
        full_skill_info = self.skill_registry.get_skill_info(skill_step.name)
        if not full_skill_info:
            self.print(
                f"[QwenVLBrain] Warning: Skill '{skill_step.name}' found in plan but not in registry."
            )
            return None

        # Combine registry info with plan-specific params and return
        full_skill_info.update(
            {
                "parameters": skill_step.params,
                "index": index,
            }
        )
        return full_skill_info

    def should_monitor(
        self, obs_history, system_status: SystemStatus | None = None
    ) -> bool:
        """Check if it's time to monitor the current skill execution."""
        if system_status is not None and system_status != SystemStatus.EXECUTING:
            return False
        if not self.state.current_plan:
            return False
        current_skill_info = self.get_next_skill()
        if not current_skill_info:
            return False
        if not current_skill_info.get("enable_monitoring", True):
            return False
        if not self.state.current_plan or not self.state.current_task:
            self.print("[QwenVLBrain] should_monitor: No active task")
            return False
        if len(obs_history) == 0:
            # self.print(
            #     "[QwenVLBrain] should_monitor: No observation data available for monitoring, len obs_history is 0"
            # )
            return False
        if not (
            self.calculate_indicesv2(
                self.FRAME_TOTAL, len(obs_history), self.FRAME_JUMP * self.FRAME_TOTAL
            )
        ):
            self.print(
                "[QwenVLBrain] should_monitor: No observation data available for monitoring"
            )
            return False

        current_time = time.time()
        return (
            current_time - self.state.last_monitoring_time
        ) >= self.state.current_plan.skill_monitoring_interval

    def monitor_skill_execution(self, obs_history: list[Any] = []) -> Dict[str, Any]:
        """
        监控当前技能执行并给出决策建议。

        注意：此方法不设置系统状态，只返回监控结果。
        System 根据返回结果决定是否需要状态转换。

        Args:
            obs_history: obs_history

        Returns:
            Dict with monitoring result (success, failed, progress, etc.)
        """
        try:
            self.state.last_monitoring_time = time.time()

            # Get current skill info
            current_skill = self.get_next_skill()
            assert current_skill

            # 移除：self.state.status = SystemStatus.MONITORING

            # Use Qwen VL to analyze current situation
            result = self._query_qwen_for_monitoring(
                self.state.current_task, current_skill, obs_history
            )
            # 移除：self.state.status = SystemStatus.EXECUTING
            return result

        except Exception as e:
            # 移除：self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            return {"result": "progress", "reason": str(e)}

    def interrupt_task(self, reason: str = "User interrupt"):
        """
        中断当前任务执行，清理 Brain 的任务数据。

        注意：此方法不设置系统状态，只清理 Brain 内部数据。
        系统状态由 System 统一管理。
        """
        if self.state.current_task:
            self.print(f"[QwenVLBrain] Interrupting task: {reason}")
            # 移除：self.state.status = SystemStatus.IDLE
            self.state.current_task = None
            self.state.current_plan = None
            self.state.current_skill_index = 0
            self.monitor_memory.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current brain status."""
        return {
            # 移除 status 字段，改为返回任务和计划状态
            "has_task": self.has_task(),
            "has_plan": self.has_plan(),
            "has_pending_skills": self.has_pending_skills(),
            "is_plan_complete": self.is_plan_complete(),
            "current_task": self.state.current_task.description
            if self.state.current_task
            else None,
            "current_skill_index": self.state.current_skill_index,
            "total_skills": len(self.state.current_plan.steps)
            if self.state.current_plan
            else 0,
            "error_message": self.state.error_message,
            "last_monitoring": self.state.last_monitoring_time,
        }

    def reset(self) -> bool:
        self.skill_registry: Optional["SkillRegistry"] = None
        self.state = BrainState()
        # Monitoring configuration
        self.monitor_waiting_times = 0
        self.monitor_memory = BrainMemory()
        self.replan_memory = BrainMemory()
        self.replan_memory.add_system_prompt()
        self.current_skill_monitor_time = 0

        self.FRAME_JUMP = 15
        self.FRAME_TOTAL = 3
        return True

    # ========== Task Memory Management (任务级长期记忆) ==========

    def get_task_memory(self) -> str:
        """获取当前任务的长期记忆摘要"""
        return self.state.task_memory

    def update_task_memory(
        self, info: dict, analysis: str, update_type: str = "skill_execution"
    ):
        """
        更新任务级记忆。

        Args:
            info: 相关的上下文信息
                - skill_execution: {'name': str, 'result': str, ...}
                - replan: {'changes': str}
            analysis: LLM 生成的分析或推理 (execution_summary or replan_reasoning)
            update_type: 更新类型 ("skill_execution" | "replan")
        """
        if not self.state.current_task:
            return

        current_memory = (
            self.state.task_memory
            if self.state.task_memory
            else "(Task just started, no prior memory)"
        )

        if update_type == "skill_execution":
            event_description = (
                f"**Latest Skill Executed:**\n"
                f"- Name: {info.get('name', 'unknown')}\n"
                f"- Result: {info.get('result', 'unknown')}\n"
                f"- Analysis: {analysis[:500] if analysis else 'N/A'}...\n"
            )
            context_instruction = "Summarize what has been accomplished so far."
        elif update_type == "replan":
            event_description = (
                f"**Replanning Event:**\n"
                f"- Reasoning: {analysis[:800]}...\n"
                f"- Changes: {info.get('changes', 'N/A')}\n"
            )
            context_instruction = (
                "Update the summary to reflect that the plan has changed and why."
            )
        elif update_type == "task_modified":
            event_description = (
                f"**Task Goal Modified:**\n"
                f"- Old Task: {info.get('old_description', 'N/A')}\n"
                f"- New Task: {info.get('new_description', 'N/A')}\n"
                f"- Reason: {info.get('reason', 'Human feedback')}\n"
            )
            context_instruction = (
                "Update the summary to reflect that the task goal itself has changed. "
                "Note what was accomplished under the old goal and what the new goal is."
            )
        elif update_type == "initial_plan":
            event_description = (
                f"**Initial Plan Created:**\n"
                f"- Plan Overview: {info.get('plan_summary', 'N/A')}\n"
            )
            context_instruction = (
                "Initialize the task memory with the starting plan and task goal."
            )
        else:
            self.print(f"[QwenVLBrain] Unknown update_type: {update_type}")
            return

        # 构建更新 prompt
        update_prompt = (
            f"You are updating a running task memory log ({update_type}).\n"
            "Given the old summary and the latest event, create a NEW, concise summary.\n\n"
            f"**Original Task:** {self.state.current_task.description}\n\n"
            f"**Previous Task Memory:**\n{current_memory}\n\n"
            f"{event_description}\n"
            "**Instructions:**\n"
            f"1. {context_instruction}\n"
            "2. Note any important state changes or progress.\n"
            "3. Keep it under 200 words.\n\n"
            "**New Task Memory Summary:**"
        )

        try:
            if self.model_adapter:
                temp_memory = BrainMemory()
                temp_memory.add_user_input([update_prompt])
                new_summary, _ = self.model_adapter.generate(
                    temp_memory.history, max_tokens=300, thinking=False
                )
                self.state.task_memory = new_summary.strip()
                self.print(
                    f"[QwenVLBrain] Task memory updated ({update_type}): {self.state.task_memory[:100]}..."
                )
            else:
                # Fallback: 简单拼接
                self._fallback_update_task_memory(info, update_type)

        except Exception as e:
            self.print(f"[QwenVLBrain] Failed to update task memory: {e}")
            self._fallback_update_task_memory(info, update_type)

    def _fallback_update_task_memory(self, info: dict, update_type: str):
        """Fallback method for task memory update when LLM is unavailable."""
        if update_type == "skill_execution":
            self.state.task_memory += f"\n- {info.get('name')}: {info.get('result')}"
        elif update_type == "replan":
            self.state.task_memory += f"\n[Replan] {info.get('changes')}"
        elif update_type == "task_modified":
            self.state.task_memory += f"\n[Task Modified] {info.get('old_description')} -> {info.get('new_description')} (Reason: {info.get('reason')})"
        elif update_type == "initial_plan":
            self.state.task_memory += f"\n[Initial Plan] {info.get('plan_summary')}"

    def _get_task_memory_context(self) -> str:
        """
        获取用于注入到 prompt 中的任务记忆上下文。
        结合 task_memory (宏观摘要) 和最近的 monitor_memory (微观细节)。

        Returns:
            格式化的记忆上下文字符串
        """
        context_parts = []

        if self.state.task_memory:
            context_parts.append(
                "## Task Progress Summary (Long-term Memory):\n"
                f"{self.state.task_memory}\n"
            )

        # 可选：添加最近几步的原始记录作为微观细节
        # 这里保持简单，只返回 task_memory

        return "\n".join(context_parts) if context_parts else ""

    def _query_qwen_for_plan(
        self, task: Task, skill_descriptions: str, use_mock=False
    ) -> SkillPlan:
        if self.model_adapter is None or use_mock:
            self.print("[QwenVLBrain] Mock 模式：返回一个预设的计划。")

            mock_plan = """```json
[
    {
        "step": 1,
        "method": "move_box_to_suitable_position",
        "params": {}
    },
    {
        "step": 2,
        "method": "open_box",
        "params": {}
    },
    {
        "step": 3,
        "method": "grasp_spanner",
        "params": {}
    },
    {
        "step": 4,
        "method": "move_to_target_object",
        "params": {
            "target_object": "palm",
            "gripper_state": "-1"
        }
    }
]
"""
            return self._parse_plan_response(mock_plan, task)

        # 1. 准备一个临时的 BrainMemory 用于本次规划
        planning_memory = BrainMemory()

        # 2. 格式化并添加系统提示 (静态内容：角色、技能、输出格式)
        system_prompt = self._format_system_prompt_for_planning()
        planning_memory.add_system_prompt(system_prompt)

        # 3. 添加用户输入 (动态内容：任务描述、图像)
        user_prompt = self._format_user_prompt_for_planning(task)
        user_content = [user_prompt]
        if task.image:
            user_content.append(task.image)
            task.image.save(
                os.path.join(
                    self.log_path,
                    f"{time.time()}_plan_task_{task.id}_input_image.png",
                )
            )
        planning_memory.add_user_input(contents=user_content)

        # 4. 使用新的单一接口调用模型
        try:
            response_text, _ = self.model_adapter.generate(
                history=planning_memory.history,
                max_tokens=self.max_tokens,
                thinking=True,
            )  # type: ignore
            plan = self._parse_plan_response(response_text, task)
            self.print(
                f"[QwenVLBrain] 使用 {self.adapter_type} 生成了计划:\n{plan.pretty_print()}"
            )
            return plan
        except Exception as e:
            import traceback

            self.print(f"[QwenVLBrain] 规划时出错：{e}")
            traceback.print_exc()
            # 出错时回退
            raise RuntimeError("Failed to create plan")

    def _mock_plan_task(self, task: Task, skill_descriptions: str) -> SkillPlan:
        """Mock implementation for planning when model adapter is not available."""
        # Simple heuristic planning based on task description
        skill_sequence = []
        skill_params = []

        instruction_lower = task.description.lower()

        # Example planning logic (replace with actual Qwen VL calls)
        if "pick" in instruction_lower and "place" in instruction_lower:
            skill_sequence = ["pick_and_place"]
            skill_params = [
                {
                    "pickup_position": [0.4, 0.0, 0.3],
                    "place_position": [0.4, 0.3, 0.3],
                }
            ]
        elif "reach" in instruction_lower:
            skill_sequence = ["reach_position"]
            skill_params = [{"target_position": [0.4, 0.0, 0.4], "tolerance": 0.01}]
        elif "grasp" in instruction_lower:
            skill_sequence = ["grasp_object"]
            skill_params = [{"grasp_force": 0.5}]
        elif "home" in instruction_lower or "reset" in instruction_lower:
            skill_sequence = ["reset_to_home"]
            skill_params = [{}]
        elif "wait" in instruction_lower:
            skill_sequence = ["wait"]
            skill_params = [{"duration": 2.0}]
        elif "state" in instruction_lower or "status" in instruction_lower:
            skill_sequence = ["get_current_state"]
            skill_params = [{}]
        else:
            # Default fallback - reset to home
            skill_sequence = ["assemble_object"]
            skill_params = [{}]

        return SkillPlan(
            task_id=task.id,
            skill_sequence=skill_sequence,
            skill_params=skill_params,
            skill_monitoring_interval=self.skill_monitoring_interval,
            expected_duration=len(skill_sequence) * 10.0,  # Rough estimate
        )

    @staticmethod
    def calculate_indicesv2(total, available, mini_available):
        if available < mini_available or total > available or total < 2:
            return []

        step = (available - 1) / (total - 1)
        indices = [round(i * step) for i in range(total)]
        indices[-1] = available - 1  # Ensure the last index is correct
        return indices

    def _query_qwen_for_monitoring(
        self,
        task: Task,
        current_skill: Dict[str, Any],
        obs_history: Optional[Any],
        enable: bool = False,
    ) -> Dict[str, Any]:
        """
        Query Qwen VL to make monitoring decisions.

        Uses the configured model adapter to analyze the current situation
        and output the current skill execution status.
        """
        if not enable or self.adapter_type is None:
            # Fallback to mock implementation
            return self._mock_monitoring_decision(task, current_skill, obs_history)
        assert type(obs_history) is list
        try:
            if self.visualize:
                inspector_rgb = (
                    obs_history[-1].data["policy"]["inspector_side"][0].cpu().numpy()
                )
                front_rgb = (
                    obs_history[-1].data["policy"]["inspector_top"][0].cpu().numpy()
                )
                import matplotlib.pyplot as plt

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 创建 1 行 2 列的子图
                axs[0].imshow(inspector_rgb)
                axs[0].axis("off")
                axs[0].set_title("Inspector View")
                axs[1].imshow(front_rgb)
                axs[1].axis("off")
                axs[1].set_title("Front View")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.log_path,
                        f"{len(self.system_state.plan_history)}_{self.state.current_skill_index}_monitor_{current_skill['name']}_input_{len(self.monitor_memory.history)}.png",
                    )
                )
                plt.close()

            video_frames_inspect = []
            video_frames_front = []

            # TODO 提取 obs 这部分应该要放在外面才对，这里面只进行 query_qwen 的逻辑
            # 计算可用的观察帧数
            available_frames = len(obs_history)

            indices = self.calculate_indicesv2(
                self.FRAME_TOTAL, available_frames, self.FRAME_JUMP * self.FRAME_TOTAL
            )

            # 按索引顺序排序（从旧到新）
            indices.sort()
            # 提取帧
            for frame_index in indices:
                # self.print(
                #     f"[QwenVLBrain] obs_shape: { {key: val.shape for key, val in obs_history[frame_index].data['policy'].items() if isinstance(val, torch.Tensor)} }, "
                # )
                video_frames_inspect.append(
                    Image.fromarray(
                        obs_history[frame_index]
                        .data["policy"]["inspector_side"][0]
                        .cpu()
                        .numpy()
                    )
                )
                video_frames_front.append(
                    Image.fromarray(
                        obs_history[frame_index]
                        .data["policy"]["inspector_top"][0]
                        .cpu()
                        .numpy()
                    )
                )
            if self.visualize:
                # 获取原始视频帧率（示例值，需根据实际情况替换）
                original_fps = 2
                duration = 1000 // original_fps  # 每帧持续时间（毫秒）

                all_frames = []
                max_length = max(len(video_frames_inspect), len(video_frames_front))
                last_inspect = (
                    video_frames_inspect[-1] if video_frames_inspect else None
                )
                last_front = video_frames_front[-1] if video_frames_front else None

                for i in range(max_length):
                    frame_inspect = (
                        video_frames_inspect[i]
                        if i < len(video_frames_inspect)
                        else last_inspect
                    )
                    frame_front = (
                        video_frames_front[i]
                        if i < len(video_frames_front)
                        else last_front
                    )

                    if frame_inspect and frame_front:
                        # 保持宽高比对齐
                        target_height = frame_inspect.size[1]
                        inspect_ratio = frame_inspect.size[0] / target_height
                        front_ratio = frame_front.size[0] / target_height

                        inspect_width = int(target_height * inspect_ratio)
                        front_width = int(target_height * front_ratio)

                        combined = Image.new(
                            "RGB",
                            (inspect_width + front_width, target_height),
                            (0, 0, 0),
                        )
                        combined.paste(
                            frame_inspect.resize((inspect_width, target_height)), (0, 0)
                        )
                        combined.paste(
                            frame_front.resize((front_width, target_height)),
                            (inspect_width, 0),
                        )
                        all_frames.append(combined)
                    elif frame_inspect:
                        all_frames.append(frame_inspect)
                    elif frame_front:
                        all_frames.append(frame_front)

                if all_frames:
                    gif_path = os.path.join(
                        self.log_path,
                        f"{len(self.system_state.plan_history)}_{self.state.current_skill_index}_monitor_{current_skill['name']}_input_{len(self.monitor_memory.history)}.gif",
                    )
                    all_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=all_frames[1:],
                        duration=duration,
                        loop=0,
                        optimize=False,  # 保留图像质量
                    )
            image_data = Image.fromarray(inspector_rgb)
            task.image = image_data  # Update task with image
            # Prepare input for the model adapte
            self.monitor_memory.add_user_input(
                contents=[
                    "belowing is current scene observation from side camera",
                    video_frames_inspect,
                    "belowing is current scene observation from front camera",
                    video_frames_front,
                ]
            )
            # Generate response
            self.print(
                f"[QwenVLBrain] Monitoring task: {task.description}, skill: {current_skill['name']}"
            )
            response_text, _ = self.model_adapter.generate(
                self.monitor_memory.fetch_history(last_n=0),
                max_tokens=self.max_tokens // 2,  # Use fewer tokens for monitoring
                thinking=False,
            )
            self.monitor_memory.add_assistant_output(response_text)
            # Parse the response to extract monitoring decision
            decision = self._parse_monitoring_response(response_text)
            self.print(
                f"[QwenVLBrain] Monitoring result using {self.adapter_type}: {decision['result']}"
            )
            # !!!---!!!
            # obs_history.clear()  # 在 handle 完 obs result 后再 clean
            return decision

        except Exception as e:
            self.print(
                f"[QwenVLBrain] Error in monitoring: {e.__class__.__name__}: {e}"
            )
            import traceback

            traceback.print_exc()
            self.print("[QwenVLBrain] Falling back to mock monitoring")
            return self._mock_monitoring_decision(task, current_skill, obs_history)

    def _mock_monitoring_decision(
        self,
        task: Task,
        current_skill: Dict[str, Any],
        observation: Optional[Any],
    ) -> Dict[str, Any]:
        """Mock monitoring decision when model adapter is not available."""
        # Simple heuristic monitoring
        if not observation:
            return {"action": "continue", "reason": "No observation data"}

        # For demonstration, just continue execution
        return {
            "action": "continue",
            "reason": "Skill execution proceeding normally",
            "confidence": 0.8,
        }

    def _parse_json_response(
        self, response_text: str, repair_by_llm=False
    ) -> Dict | List:
        """
        从响应文本中提取并解析 JSON 内容。
        如果初次解析失败，会尝试请求模型生成有效的 JSON 再次解析。

        Args:
            response_text: 包含可能 JSON 内容的原始文本

        Returns:
            解析成功的字典或列表对象，失败返回 None
        """
        parsed_data = extract_json_from_text(response_text, repair=not repair_by_llm)

        if parsed_data is not None:
            return parsed_data
        else:
            self.print("[QwenVLBrain] JSON parsing failed.")
            raise ValueError(
                "[QwenVLBrain] Unable to parse JSON from response text, even after recovery attempt."
            )

    def _parse_monitoring_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the monitoring response from the model adapter.

        Args:
            response_text: The raw text response from the model

        Returns:
            Dict with monitoring decision
        """
        try:
            # Try to parse JSON response
            decision_data = self._parse_json_response(response_text)

            if decision_data is not None:
                result = decision_data.get("result", "progress")
                reason = decision_data.get("reason", "No reason provided")
                confidence = decision_data.get("confidence", 0.5)

                valid_options = [
                    "successed",
                    "failed",
                    "progress",
                ]
                if result not in valid_options:
                    self.print(
                        f"[QwenVLBrain] Invalid result '{result}', defaulting to 'progress'"
                    )
                    result = "progress"

                return {
                    "result": result,
                    "reason": reason,
                    "confidence": confidence,
                }
            else:
                # 原有文本解析逻辑
                raise ValueError(
                    "[QwenVLBrain] Failed to parse valid JSON from response text"
                )

        except json.JSONDecodeError as e:
            self.print(f"[QwenVLBrain] Failed to parse monitoring JSON: {e}")
            return self._parse_monitoring_text(response_text)
        except Exception as e:
            self.print(f"[QwenVLBrain] Error parsing monitoring response: {e}")
            return {
                "result": "progress",
                "reason": "Error in parsing response",
            }

    def _format_system_prompt_for_planning(self) -> str:
        """
        Format the SYSTEM prompt for planning.

        System Prompt 包含：
        - 角色定义 (你是谁)
        - 可用技能列表 (你能做什么)
        - 输出格式要求 (你应该怎么输出)
        - 示例 (参考格式)

        这些都是静态的、不随每次请求变化的内容。
        """
        prompt = (
            "You are a helpful robot task planner. Your goal is to create a JSON execution plan based on a task description and available skills.\n\n"
            "**Available Skills:**\n"
            f"{self.skill_registry.get_skill_descriptions()}\n\n"
            "**Planning Guidelines:**\n"
            "1. **Analyze Preconditions:** Carefully analyze the task and the initial state (from the image) to determine the necessary preconditions for each step.\n"
            "2. **Logical Order:** Ensure the sequence of skills is logical. For example, interacting with a container must happen before taking an object from it.\n"
            "3. **Follow Skill Instructions:** Pay close attention to the instructions within each skill's description.\n"
            "4. **Plan-time human intervention rule:** Only add `human_intervention` when the task cannot even start with the current scene (key objects/containers absent or unreachable). In that case, make it the first and only step so the system waits for human setup; later skills will be generated after human feedback via replanning.\n\n"
            "**Output Format:**\n"
            "Provide the execution plan as a JSON array. Each object must include:\n"
            "- `step`: Sequential integer starting from 1\n"
            "- `method`: The skill name\n"
            "- `params`: Parameters for the skill\n\n"
            "DO NOT USE any placeholder values in the JSON.\n"
            "BEFORE JSON CONTENT, include a brief explanation of your reasoning including your analysis of the scene and task.\n\n"
            "**Example Response (normal case):**\n"
            "reasoning: There is a closed drawer on the table. To pick up the tool inside, we first need to open the drawer, then grasp the tool.\n"
            "```json\n"
            "[\n"
            '    {"step": 1, "method": "open_drawer", "params": {}},\n'
            '    {"step": 2, "method": "grasp_object", "params": {"target": "tool"}}\n'
            "]\n"
            "```\n"
            "\n**Example Response (cannot start, requires human help):**\n"
            "reasoning: The task asks to grasp the tool, but the scene shows an empty workspace (no container or tool present). We must first request human help to place the items, then replan after feedback.\n"
            "```json\n"
            "[\n"
            '    {"step": 1, "method": "human_intervention", "params": {"reason": "Required items not visible; please place them and confirm."}}\n'
            "]\n"
            "```\n"
        )
        return prompt

    def _format_user_prompt_for_planning(self, task: Task) -> str:
        """
        Format the USER prompt for planning.

        User Prompt 包含：
        - 当前任务描述 (动态)
        - 当前场景图像 (动态，在调用处添加)

        这些是每次请求都不同的动态内容。
        """
        prompt = (
            f"**Task:** {task.description}\n\n"
            "Please analyze the provided image and create an execution plan for this task."
        )
        return prompt

    def _format_system_prompt_for_replanning_operation(self) -> str:
        """
        Format the SYSTEM prompt for replanning.

        System Prompt 包含所有静态内容：
        - 角色定义 (Expert Plan Corrector)
        - 可用技能列表
        - 决策流程指导
        - 可用操作定义 (update_status, insert, delete, retry)
        - 详细示例
        - 输出格式要求
        """
        prompt = (
            "You are an expert robot task plan corrector. Your goal is to analyze execution results and correct the plan *only if* there is a failure or issue.\n\n"
            "## Available Skills\n"
            f"{self.skill_registry.get_skill_descriptions()}\n\n"
            "**Handling Missing Skills:**\n"
            "If the task requires an action (e.g., 'give to human', 'pour water') but no specific skill exists for it:\n"
            "- Use generic skills like `move_to_target_pose` or `move_to_target_object` to approximate the action.\n"
            "- OR use `human_intervention` to ask for help.\n"
            "- **NEVER** simply stop and claim success if the action is not performed.\n"
            "## Decision Process\n"
            "Follow this process for every replan request:\n"
            "1. **Analyze the Past (Outcome Analysis):**\n"
            "   - Use the `Execution Summary` to determine the TRUE outcome of the last skill. Visual evidence is the absolute truth.\n"
            "   - If the reported status contradicts the visual evidence, correct it using `update_status`.\n"
            "2. **Validate the Future (Plan Validation):**\n"
            "   - Review remaining `PENDING` skills.\n"
            "   - Ask: 'Given the current scene and outcome of the last skill, can this plan achieve the Original Task?'\n"
            "3. **Choose Operations:**\n"
            "   - If corrections are needed, provide a JSON array of operations.\n"
            "   - If the plan is correct, provide an empty array `[]`.\n\n"
            "## Available Operations (index starts from 0)\n"
            "1. **update_status**: Correct the status of a skill if visual evidence contradicts the reported result. THIS IS YOUR MOST IMPORTANT CORRECTION TOOL.\n"
            '   - Format: `{"operation": "update_status", "index": N, "new_status": "COMPLETED"|"FAILED"}`\n'
            "2. **insert**: Add a new skill AFTER the specified index. Especially if the last skill index is i, if you want to add a skill at the end, use index i (not i+1).\n"
            '   - Format: `{"operation": "insert", "index": N, "method": "skill_name", "params": {}}`\n'
            "3. **delete**: Remove a skill at the specified index.\n"
            '   - Format: `{"operation": "delete", "index": N}`\n'
            "4. **retry**: Re-run a skill that has a `FAILED` status.\n"
            '   - Format: `{"operation": "retry", "index": N}`\n'
            "5. **modify_task**: Modify the original task description based on human feedback or new understanding. Use this ONLY when the user explicitly requests a change to the task goal itself.\n"
            '   - Format: `{"operation": "modify_task", "new_description": "updated task description", "reason": "why the task needs to change"}`\n\n'
            "## Critical Rules\n"
            "- All `index` values MUST refer to the ORIGINAL plan indices.\n"
            "- DO NOT use placeholder/hypothetical/example values in params. The params you provide are FINAL.\n"
            "- DO NOT calculate new indices from your own operations; the system handles index shifts.\n\n"
            "## Detailed Example\n"
            "**Scenario:** Current plan is:\n"
            "```\n"
            "Step 0: [COMPLETED] open_drawer({})\n"
            "Step 1: [FAILED] grasp_object({})\n"
            "Step 2: [PENDING] close_drawer({})\n"
            "```\n"
            "**Analysis:** `grasp_object` at index 1 failed. Need to insert `set_gripper` before it and retry.\n"
            "**Solution:** Insert after index 0, then retry index 1 (using original indices).\n"
            "```json\n"
            '[{"operation": "insert", "index": 0, "method": "set_gripper", "params": {"state": 1.0}}, {"operation": "retry", "index": 1}]\n'
            "```\n\n"
            "**Example (Plan is correct):**\n"
            "If the last skill succeeded and remaining plan is valid, output:\n"
            "```json\n[]\n```\n\n"
            "## Detailed Example\n"
            "**Scenario:** Current plan is:\n"
            "```\n"
            'Step 0: [COMPLETED] navigate_to({"location": "kitchen_counter"})\n'
            "```\n"
            "**Analysis:** The original task is 'Wipe the kitchen counter and confirm it is clean.' The current plan only moves the robot to the counter but omits all subsequent actions: detecting dirt, wiping the surface, and verifying cleanliness. Visual evidence shows the counter is visibly soiled and no cleaning tool has been deployed. The plan is incomplete and cannot achieve the task with only one step.\n"
            "**Solution:** Insert the three missing skills—`detect_surface_condition`, `wipe_surface`, and `verify_cleanliness`—in logical order after the original step (index 0), using original indices.\n"
            "```json\n"
            '[{"operation": "insert", "index": 0, "method": "detect_surface_condition", "params": {"surface": "kitchen_counter"}}, {"operation": "insert", "index": 0, "method": "wipe_surface", "params": {"surface": "kitchen_counter", "tool": "microfiber_cloth"}}, {"operation": "insert", "index": 0, "method": "verify_cleanliness", "params": {"surface": "kitchen_counter"}}]\n'
            "```\n\n"
            "## Output Format\n"
        )

        if self.insert_think_prompt:
            prompt += "First, think step-by-step in `<think>` tags. Then provide a JSON array of operations.\n"
        else:
            prompt += "Provide a JSON array of operations. If the plan is correct, output `[]`.\n"

        return prompt

    def _format_system_prompt_for_replanning(self) -> str:
        """
        Format the SYSTEM prompt for replanning.

        System Prompt 包含所有静态内容：
        - 角色定义 (Expert Plan Corrector)
        - 可用技能列表
        - 决策流程指导
        - 可用操作定义 (update_status, insert, delete, retry)
        - 详细示例
        - 输出格式要求
        """
        prompt = (
            "You are an expert robot task plan corrector.\n"
            "Your PRIMARY GOAL is to ensure the robot **fully completes the Original Task**.\n"
            "You must analyze the current situation and update the plan if:\n"
            "1. The last skill failed.\n"
            "2. The current plan is **incomplete** (missing steps to reach the final goal).\n"
            "3. The plan is inefficient or wrong.\n\n"
            "## Available Skills\n"
            f"{self.skill_registry.get_skill_descriptions()}\n\n"
            "## Decision Process\n"
            "Follow this process for every replan request:\n"
            "1. **Analyze the Past (Outcome Analysis):**\n"
            "   - Use the `Execution Summary` to determine the TRUE outcome of the last skill. Visual evidence is the absolute truth.\n"
            "   - If the reported status contradicts the visual evidence, note this for your replanning.\n"
            "2. **Check Goal Completion:**\n"
            "   - Compare the **Current Robot State** and **Scene** against the **Original Task** requirements.\n"
            "   - Ask: 'Is the task *fully* finished?' (e.g., Is the object actually in the hand? Is the drawer closed?)\n"
            "   - If the Plan is empty but the Task is NOT finished, you **MUST** insert new steps.\n"
            "3. **Validate the Future (Plan Validation):**\n"
            "   - Review remaining `PENDING` skills.\n"
            "   - If pending skills are missing, insert them.\n"
            "4. **Choose Operations:**\n"
            "   - Identify the index of the first step that requires correction, insertion, or is simply the next logical step to execute (let's call this Step N).\n"
            "   - Provide a JSON array representing the **entire remaining plan** starting from Step N.\n"
            "   - **CRITICAL:** You must output Step N and **ALL** subsequent steps required to complete the task, even if those subsequent steps are unchanged from the original plan.\n"
            "   - Format:\n"
            "       - `step`: Sequential integer starting from N (the first modified/executed step)\n"
            "       - `method`: The skill name\n"
            "       - `params`: Parameters for the skill\n"
            "       - DO NOT USE any placeholder values in the JSON.\n"
            "   - If the plan is completely correct and the next pending step in the original plan needs no changes, output `[]`.\n\n"
            "   - **Task Modification:** If the original task is impossible or human feedback changes the goal, the **first** object in the array MUST be a `modify_task` operation, followed by the new plan steps.\n"
            '     Format: `{"operation": "modify_task", "new_description": "...", "reason": "..."}`\n\n'
            "## Detailed Example\n"
            "**Scenario:** Current plan is:\n"
            "```\n"
            "Step 0: [COMPLETED] open_drawer({})\n"
            "Step 1: [FAILED] grasp_object({})\n"
            "Step 2: [PENDING] close_drawer({})\n"
            "```\n"
            "**Analysis:** `grasp_object` at Step 1 failed because the gripper was closed. \n"
            "Correction must start at Step 1. We need to insert `set_gripper`, retry `grasp_object`, and **must also include** the original pending `close_drawer` at the end.\n"
            "**Solution:** Output the new sequence starting from Step 1.\n"
            "```json\n"
            "[\n"
            '  {"step": 1, "method": "set_gripper", "params": {"state": 1.0}},\n'
            '  {"step": 2, "method": "grasp_object", "params": {}},\n'
            '  {"step": 3, "method": "close_drawer", "params": {}}\n'
            "]\n"
            "```\n\n"
            "## Detailed Example (Task Modification)\n"
            "**Scenario:** Current plan is:\n"
            "```\n"
            'Step 0: [COMPLETED] navigate_to({"location": "kitchen"})\n'
            "Step 1: [PENDING] pick_apple({})\n"
            "```\n"
            "**Analysis:** The user provides feedback: 'Actually, I want an orange, not an apple'. The task description needs to change, and the plan needs to update from Step 1 onwards.\n"
            "**Solution:** First modify the task, then provide the new plan steps starting from Step 1.\n"
            "```json\n"
            "[\n"
            '  {"operation": "modify_task", "new_description": "Find and pick up an orange", "reason": "User changed instruction"},\n'
            '  {"step": 1, "method": "detect_object", "params": {"object": "orange"}},\n'
            '  {"step": 2, "method": "pick_object", "params": {"object": "orange"}}\n'
            "]\n"
            "```\n\n"
            "## Output Format\n"
        )

        if self.insert_think_prompt:
            prompt += "First, think step-by-step in `<think>` tags. Then provide a JSON array of operations.\n"
        else:
            prompt += "Provide a JSON array of operations. If the plan is correct, output `[]`.\n"

        return prompt

    def _format_user_prompt_for_replanning(
        self,
        task: Task,
        current_plan: SkillPlan,
        skill_history: list[dict],
        observation: Observation,
        human_feedback: str = "",
    ) -> str:
        """
        Format the USER prompt for replanning.

        User Prompt 只包含动态内容：
        - 原始任务描述
        - 任务进度记忆
        - 当前计划状态
        - 最后技能执行信息
        - 机器人当前状态
        - 人类反馈 (如果有)
        """
        current_plan_state = current_plan.pretty_print()
        last_skill_info = skill_history[-1]
        last_execution_info = (
            f"Skill Index: {last_skill_info['index']}\n"
            f"Skill Name: {last_skill_info['name']}\n"
            f"Skill Execution Result (skill reported): {last_skill_info.get('result', 'unknown')}\n"
            f"Execution Summary (visual evidence analysis): {last_skill_info.get('execution_summary', 'No summary as it complete success')}"
        )

        eef_pos_str = np.array2string(
            observation.data["policy"]["eef_pos"].cpu().numpy(),
            precision=3,
            separator=", ",
        )
        eef_quat_str = np.array2string(
            observation.data["policy"]["eef_quat"].cpu().numpy(),
            precision=3,
            separator=", ",
        )

        # 构建动态部分
        prompt = f"## Original Task\n{task.description}\n\n"

        # 任务进度记忆
        task_memory_context = self._get_task_memory_context()
        if task_memory_context:
            prompt += f"{task_memory_context}\n\n"

        prompt += (
            f"## Current Plan:\n{current_plan_state}\n\n"
            f"## Last Skill Execution Info\n{last_execution_info}\n\n"
            f"## Current Robot State\n- pose: {eef_pos_str}, quat: {eef_quat_str}\n\n"
        )

        # 人类反馈
        if human_feedback:
            prompt += (
                "## 🚨 URGENT HUMAN FEEDBACK (INTERRUPTION)\n"
                f'The user interrupted with: "{human_feedback}"\n'
                "You MUST prioritize this feedback. It likely indicates a failure or desired change.\n"
                "**Important:** If the feedback suggests changing the GOAL of the task (not just how to achieve it), "
                "use the `modify_task` operation to update the task description.\n\n"
            )

        prompt += "Please analyze the current situation and provide your correction operations."
        prompt += (
            "## FINAL CHECK\n"
            "If the plan is empty `[]` but the physical goal (e.g. object delivery) is not visible in the scene, "
            "do NOT return `[]`. You MUST generate the missing steps."
        )
        return prompt

    def _format_system_prompt_for_monitoring(
        self,
        task: Task,
        current_skill: Dict[str, Any],
        observation: Optional[Any] = None,
    ) -> str:
        """Format a prompt for Qwen VL monitoring."""
        prompt = (
            "The robot are try to orchestrate skills to accomplish a task.\n"
            "You are monitoring robot's SKILL execution. Analyze the current situation and output the current execution result.\n"
            "You should only consider the execution of CURRENT SKILL rather than the completion of tasks.\n"
            "\n"
            f"Current Skill: {current_skill['name']}\n"
            f"Skill Description: {current_skill['description']}\n"
            f"Skill Parameters: {current_skill['parameters']}\n"
            f"Skill Criterion: {current_skill['criterion']}\n"
            "\n"
            "For each round of conversation, you will receive the current observation in Image format, You should analyze the image and response with a JSON object containing:\n"
            "- result: The value for this field must be one of the options from `"
            + str(list(current_skill["criterion"].keys()))
            + "`\n"
            "- reason: Explanation for the result.\n"
            "- current_scene_state: Description what happened in the scene, based on the observation\n"
            "- confidence: Float between 0 and 1\n"
            "\n"
            "Example response:\n"
            "```json\n"
            "{\n"
            '    "result": "failed",\n'
            '    "reason": "The object was not grasped correctly",\n'
            '    "current_scene_state": "The robot attempted to grasp the object but it slipped",\n'
            '    "confidence": 0.9\n'
            "}\n"
            "```\n"
            "\n"
            "Next, Let's start first monitoring round.\n"
        )
        return prompt

    def _parse_plan_response(self, response_text: str, task: Task) -> SkillPlan:
        """
        Parse the response from the model adapter to extract a skill plan.
        Assumes the response is a JSON array of skill objects.
        """
        try:
            plan_data = self._parse_json_response(response_text)

            if plan_data is None or not isinstance(plan_data, list):
                # 如果解析失败，或者解析结果不是一个列表，则回退
                self.print(
                    "[QwenVLBrain] No valid JSON array found in response, return empty plan."
                )
                return SkillPlan(task_id=task.id, skill_list=[])

            return SkillPlan(
                task_id=task.id,
                skill_list=plan_data,
                skill_monitoring_interval=self.skill_monitoring_interval,
            )
        except Exception as e:
            self.print(f"[QwenVLBrain] Error parsing initial plan response: {e}")
            return SkillPlan(task_id=task.id, skill_list=[])

    # def _parse_monitoring_text(self, response_text: str) -> Dict[str, Any]:
    #     """Parse monitoring decision from text response."""
    #     response_lower = response_text.lower()

    #     # Simple keyword-based parsing
    #     if any(
    #         word in response_lower for word in ["stop", "interrupt", "halt", "abort"]
    #     ):
    #         return {
    #             "action": "interrupt",
    #             "reason": "Model suggested interruption",
    #         }
    #     elif any(word in response_lower for word in ["retry", "again", "restart"]):
    #         return {"action": "retry", "reason": "Model suggested retry"}
    #     elif any(word in response_lower for word in ["complete", "done", "finished"]):
    #         return {
    #             "action": "complete",
    #             "reason": "Model indicated completion",
    #         }
    #     elif any(word in response_lower for word in ["error", "fail", "problem"]):
    #         return {"action": "error", "reason": "Model detected error"}
    #     else:
    #         return {
    #             "action": "continue",
    #             "reason": "Model suggested continuation",
    #         }
