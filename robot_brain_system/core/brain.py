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

if TYPE_CHECKING:
    from robot_brain_system.core.skill_manager import SkillRegistry


@dataclass
class BrainState:
    """
    Brain的内部工作状态。

    注意: Brain不维护系统级status,系统状态由System统一管理。
    Brain只负责任务规划和监控,通过返回值通知System状态变化。
    """

    current_task: Optional[Task] = None
    current_plan: Optional[SkillPlan] = None
    current_skill_index: int = 0
    last_monitoring_time: float = 0.0
    error_message: Optional[str] = None


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
        获取系统提示以及最后N轮对话。

        新增了一个参数 `prune_multimedia` 来优化多模态历史记录的处理，减少token消耗。

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
        # 使用深拷贝(deepcopy)来确保不会修改原始的 self.history
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
        print("[BrainMemory] Memory cleared.")

    def format_memory_content(self) -> List:
        print(f"[QwenVLBrain] Formatting memory with {len(self.history)} entries")
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
                        print(
                            f"[QwenVLBrain] format memory content encounted with unsupported content type: {content}"
                        )
            if text_segment:
                chat_content.append(text_segment)
                text_segment = ""  # clear
            chat_content.append("\n---\n")
        return chat_content

    def extract_images(self, n_max=-1):
        print(
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

    def initialize(self):
        """Initialize the brain component."""
        print("[QwenVLBrain] Initializing...")
        # Initialize model adapter

        try:
            self._initialize_model_adapter()
        except Exception as e:
            print(
                f"[QwenVLBrain] Failed to initialize model adapter: {type(e).__name__}: {e}"
            )
            import traceback

            traceback.print_exc()
            print("[QwenVLBrain] Falling back to mock implementation")
            self.adapter_type = "mock"

        print(f"[QwenVLBrain] Initialized with adapter: {self.adapter_type}")

    def _initialize_model_adapter(self):
        """根据配置初始化新的、标准化的模型适配器。"""
        self.adapter_type = self.qwen_config.get(
            "adapter_type", "mock"
        )  # "qwen_vl", "openai", or "mock"
        # 从主配置中获取模型和API相关的通用配置
        model_path = self.qwen_config.get("model_path")
        device = self.qwen_config.get("device", "auto")
        model_name = self.qwen_config.get("model")  # for OpenAI
        self.insert_think_prompt = False
        if not self.insert_think_prompt:
            print(
                "[QwenVLBrain] Think prompt injection disabled for this adapter/model combination."
            )
        self.max_tokens = self.qwen_config.get("max_tokens", 512)

        print(f"[QwenVLBrain] self.insert_think_prompt: {self.insert_think_prompt}")
        try:
            if self.adapter_type == "transformers":
                self.model_adapter = TransformersAdapter(
                    model_path=model_path,
                    device=device,
                    **self.qwen_config.get("transformers_adapter_args", {}),
                )
            elif self.adapter_type == "vllm":
                # vLLM的特定参数也可以从config传入
                self.model_adapter = VLLMAdapter(
                    model_path=model_path,
                    **self.qwen_config.get("vllm_adapter_args", {}),
                )
            elif self.adapter_type == "lmdeploy":
                # LMDeploy的特定参数
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
                # 保留mock作为备用
                print("[QwenVLBrain] 未知的 adapter_type，将使用 mock 实现。")
                self.adapter_type = "mock"
                self.model_adapter = None

            if self.model_adapter:
                print(f"[QwenVLBrain] 成功初始化适配器: {self.adapter_type}")

        except Exception as e:
            import traceback

            print(f"[QwenVLBrain] 初始化模型适配器失败: {e}")
            traceback.print_exc()
            print("[QwenVLBrain] 回退到 mock 实现。")
            self.adapter_type = "mock"
            self.model_adapter = None

    def set_skill_registry(self, skill_registry: "SkillRegistry"):
        """Set the skill registry for planning."""
        self.skill_registry = skill_registry
        print("[QwenVLBrain] Connected to skill registry")

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

    # ========================================

    def parse_task(self, instruction: str, image_data: Optional[str] = None) -> Task:
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

            print(f"[QwenVLBrain] Parsed task: {task.id} - {instruction}")
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
        # TODO 只提供image不提供text试试！
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
                f"Skill Criterion: {skill_info['criterion']}\n\n"
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
        response_text, _ = self.model_adapter.generate(
            sum_memory.history, thinking=True
        )
        return response_text

    # @retry
    def replan_task(
        self,
        task: Task,
        current_plan: SkillPlan,
        skill_history: list[dict],
        observation: Observation,
    ):
        """
        根据当前执行情况重新规划任务。

        注意: 此方法不修改系统状态,只返回新计划。
        状态管理由System负责。

        Args:
            task: The task to plan for
            current_plan: 当前正在执行的计划
            skill_history: 技能执行历史
            observation: 当前观测

        Returns:
            SkillPlan with sequence of skills and parameters
        """
        try:
            if not self.skill_registry:
                raise RuntimeError("Skill registry not available")

            if self.model_adapter is None:
                # Fallback to mock implementation
                raise ValueError("self.model_adapter is None")

            try:
                # 移除: self.state.status = SystemStatus.THINKING
                text_prompt = self._format_prompt_for_replanning(
                    task, current_plan, skill_history, observation
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
                self.replan_memory.add_user_input(contents=[text_prompt, current_image])
                print(f"[QwenVLBrain] 开始为任务进行再规划: {task.description}")
                response_text, _ = self.model_adapter.generate(
                    history=self.replan_memory.fetch_history(
                        last_n=0, prune_multimedia=True
                    ),  # 获取最近的对话历史
                    max_tokens=self.max_tokens,
                    thinking=True,
                )  # type: ignore
                self.replan_memory.add_assistant_output(response_text)
                # Parse the response to extract skill plan
                operations = self._parse_json_response(response_text)
                # 移除: self.state.status = SystemStatus.EXECUTING
                if not operations:
                    print(
                        "[QwenVLBrain] No operations suggested by LLM. Plan remains unchanged."
                    )
                else:
                    original_steps = list(current_plan.steps)
                    new_steps = []

                    inserts_after = {
                        op["index"]: op
                        for op in operations
                        if op["operation"] == "insert"
                    }
                    deletes_at = {
                        op["index"] for op in operations if op["operation"] == "delete"
                    }
                    updates_at = {
                        op["index"]: op
                        for op in operations
                        if op["operation"] in ["update_status", "retry", "modify"]
                    }

                    for i, original_step in enumerate(original_steps):
                        # Step 1: Decide whether to include the original step at this index
                        if i not in deletes_at:
                            step_to_add = original_step
                            # Apply any non-structural changes (update, modify, retry)
                            if i in updates_at:
                                op = updates_at[i]
                                op_type = op.get("operation")

                                if op_type == "retry":
                                    step_to_add.status = SkillStatus.PENDING
                                elif op_type == "update_status":
                                    try:
                                        new_status_str = op.get(
                                            "new_status", ""
                                        ).upper()
                                        step_to_add.status = SkillStatus[new_status_str]
                                    except KeyError:
                                        print(
                                            f"[QwenVLBrain] Warning: Invalid status '{op.get('new_status')}' for index {i}. Skipping update."
                                        )
                                elif op_type == "modify":
                                    new_name = op.get("new_method")
                                    new_params = op.get("new_params")
                                    if new_name:
                                        step_to_add.name = new_name
                                    if new_params is not None:
                                        step_to_add.params = new_params
                                    step_to_add.status = (
                                        SkillStatus.PENDING
                                    )  # Reset on modify

                            new_steps.append(step_to_add)

                        # Step 2: Check if a new skill should be inserted AFTER this original step
                        if i in inserts_after:
                            op = inserts_after[i]
                            new_skill = SkillStep(
                                name=op["method"], params=op.get("params", {})
                            )
                            new_steps.append(new_skill)

                    current_plan.steps = new_steps
                    print(
                        "[QwenVLBrain] Successfully applied all operations and reconstructed the plan."
                    )

                # The plan is now modified in-place
                assert id(self.state.current_plan) == id(current_plan), (
                    "this should not be happened as we use reference"
                )
                next_skill_info = current_plan.get_next_pending_skill_with_index()

                if next_skill_info is None:
                    print(
                        "[QwenVLBrain] No more pending skills in the plan. Task is complete. Next loop will interrupt the task."
                    )
                    # # No more skills in the plan. The brain/task is considered finished.
                    # if self.state.status == SystemStatus.EXECUTING:
                    #     self.interrupt_task("Plan completed successfully.")

                    # if self.state.status == SystemStatus.IDLE:
                    #     self.interrupt_task("Plan completed successfully.")
                    #     print(
                    #         "[RobotBrainSystem] All skills in the plan are complete. Task finished."
                    #     )
                    #     self.state.status = SystemStatus.IDLE
                    #     self.state.current_task = None
                    return None

                self.state.current_skill_index = (
                    next_skill_info[0] if next_skill_info else len(current_plan.steps)
                )

                self.initial_monitor()  # Re-initialize monitoring for the current/new skill

                print("[QwenVLBrain] Successfully replanned task. New plan state:")
                print(current_plan.pretty_print())
                return current_plan

            except Exception as e:
                import traceback

                print(f"[QwenVLBrain] Error in planning: {e}")
                traceback.print_exc()
                raise RuntimeError(f"Failed to create plan: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to create plan: {e}")

    def execute_task(self, task: Task) -> SkillPlan:
        """
        开始执行任务,生成执行计划。

        注意: 此方法不设置系统状态,只负责规划和存储任务信息。
        系统状态由System统一管理。

        Args:
            task: Task to execute

        Returns:
            The execution plan
        """
        try:
            # 移除: self.state.status = SystemStatus.THINKING
            self.state.current_task = task

            # Create execution plan
            plan = self.plan_task(task)
            self.state.current_plan = plan
            self.state.current_skill_index = 0
            # 移除: self.state.status = SystemStatus.EXECUTING
            self.initial_monitor()
            print(f"[QwenVLBrain] Started executing task: {task.description}")
            return plan

        except Exception as e:
            # 移除: self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            raise  # 让System处理错误状态

    def initial_monitor(self):
        assert self.state.current_task is not None

        current_plan = self.state.current_plan
        assert current_plan is not None

        skill_info = self.get_next_skill()
        assert skill_info is not None, (
            f"Skill info not found for index {self.state.current_skill_index}"
        )

        self.monitor_memory.clear()

        print(f"[QwenVLBrain] Monitor initialized with skill: {skill_info['name']}")
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
            print(
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
            print("[QwenVLBrain] should_monitor: No active task")
            return False
        if len(obs_history) == 0:
            print(
                "[QwenVLBrain] should_monitor: No observation data available for monitoring, len obs_history is 0"
            )
            return False
        if not (
            self.calculate_indicesv2(
                self.FRAME_TOTAL, len(obs_history), self.FRAME_JUMP * self.FRAME_TOTAL
            )
        ):
            print(
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

        注意: 此方法不设置系统状态,只返回监控结果。
        System根据返回结果决定是否需要状态转换。

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

            # 移除: self.state.status = SystemStatus.MONITORING

            # Use Qwen VL to analyze current situation
            result = self._query_qwen_for_monitoring(
                self.state.current_task, current_skill, obs_history
            )
            # 移除: self.state.status = SystemStatus.EXECUTING
            return result

        except Exception as e:
            # 移除: self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            return {"result": "progress", "reason": str(e)}

    def interrupt_task(self, reason: str = "User interrupt"):
        """
        中断当前任务执行,清理Brain的任务数据。

        注意: 此方法不设置系统状态,只清理Brain内部数据。
        系统状态由System统一管理。
        """
        if self.state.current_task:
            print(f"[QwenVLBrain] Interrupting task: {reason}")
            # 移除: self.state.status = SystemStatus.IDLE
            self.state.current_task = None
            self.state.current_plan = None
            self.state.current_skill_index = 0
            self.monitor_memory.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current brain status."""
        return {
            # 移除 status 字段,改为返回任务和计划状态
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

    def _query_qwen_for_plan(
        self, task: Task, skill_descriptions: str, use_mock=False
    ) -> SkillPlan:
        if self.model_adapter is None or use_mock:
            print("[QwenVLBrain] Mock模式：返回一个预设的计划。")
            mock_plan = """```json
[
    {
        "step": 1,
        "method": "object_tracking",
        "params": {
            "target_object": "palm"
        }
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
            "gripper_state": -1
        }
    }
]
```"""
            mock_plan = """```json
[
    {
        "step": 1,
        "method": "open_box",
        "params": {}
    },
    {
        "step": 2,
        "method": "grasp_spanner",
        "params": {}
    },
    {
        "step": 3,
        "method": "object_tracking",
        "params": {
            "target_object": "palm"
        }
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

        # 2. 格式化并添加系统提示
        system_prompt = self._format_prompt_for_planning(
            task, skill_descriptions
        )  # 这个格式化函数现在只生成文本
        planning_memory.add_system_prompt(system_prompt)

        # 3. 添加用户输入（现在包括文本和图像）
        # 注意：用户输入现在是一个内容列表
        user_content = []
        user_content.append(
            "Please generate a plan based on my task and the provided image."
        )
        if task.image:
            user_content.append(task.image)
        planning_memory.add_user_input(contents=user_content)

        # 4. 使用新的单一接口调用模型
        try:
            response_text, _ = self.model_adapter.generate(
                history=planning_memory.history,
                max_tokens=self.max_tokens,
                thinking=True,
            )  # type: ignore
            plan = self._parse_plan_response(response_text, task)
            print(
                f"[QwenVLBrain] 使用 {self.adapter_type} 生成了计划:\n{plan.pretty_print()}"
            )
            return plan
        except Exception as e:
            import traceback

            print(f"[QwenVLBrain] 规划时出错: {e}")
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
        if not enable:
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

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 创建1行2列的子图
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

            # TODO 提取 obs 这部分应该要放在外面才对，这里面只进行query_qwen的逻辑
            # 计算可用的观察帧数
            available_frames = len(obs_history)

            indices = self.calculate_indicesv2(
                self.FRAME_TOTAL, available_frames, self.FRAME_JUMP * self.FRAME_TOTAL
            )

            # 按索引顺序排序（从旧到新）
            indices.sort()
            # 提取帧
            for frame_index in indices:
                # print(
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
            print(
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
            print(
                f"[QwenVLBrain] Monitoring result using {self.adapter_type}: {decision['result']}"
            )
            # !!!---!!!
            # obs_history.clear()  # 在 handle 完 obs result 后再 clean
            return decision

        except Exception as e:
            print(f"[QwenVLBrain] Error in monitoring: {e.__class__.__name__}: {e}")
            import traceback

            traceback.print_exc()
            print("[QwenVLBrain] Falling back to mock monitoring")
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
        从响应文本中提取并解析JSON内容.
        如果初次解析失败,会尝试请求模型生成有效的JSON再次解析.

        Args:
            response_text: 包含可能JSON内容的原始文本

        Returns:
            解析成功的字典或列表对象，失败返回None
        """
        parsed_data = extract_json_from_text(response_text, repair=not repair_by_llm)

        if parsed_data is not None:
            return parsed_data

        # If initial parsing failed, log it and attempt recovery by re-prompting LLM
        print(
            f"[QwenVLBrain] Initial JSON parsing failed. Attempting LLM recovery. Original text snippet: '{response_text[:200]}...'"
        )

        # Prepare prompt for LLM to fix/provide JSON
        # Enclosing the problematic text in a markdown block for clarity for the LLM
        error_handling_prompt = (
            f"Failed to parse JSON from the following text:\n```text\n{response_text}\n```\n\n"
            "Please analyze the text above and provide a valid JSON response based on its content. "
            "Reply *only* with the valid JSON object or array. If possible, use a markdown code block, like: "
            '```json\n{"key": "value"}\n```'
        )

        try:
            temp_memory = BrainMemory()
            temp_memory.add_user_input([error_handling_prompt])
            new_llm_response_text, _ = self.model_adapter.generate(
                temp_memory.fetch_history(), max_tokens=self.max_tokens, thinking=False
            )
        except Exception as e_generate:  # Catch potential errors during LLM call itself
            print(
                f"[QwenVLBrain] Error during LLM call for JSON recovery: {e_generate}"
            )
            raise ValueError(
                "[QwenVLBrain] Unable to parse JSON from response text, even after recovery attempt."
            )

        # Attempt to parse the new response from LLM using the same robust helper
        print(
            f"[QwenVLBrain] Received new response from LLM for JSON recovery: '{new_llm_response_text}'"
        )
        parsed_data_after_recovery = extract_json_from_text(
            new_llm_response_text, repair=not repair_by_llm
        )

        if parsed_data_after_recovery is not None:
            print("[QwenVLBrain] Successfully parsed JSON after LLM recovery.")
            return parsed_data_after_recovery
        else:
            print(
                f"[QwenVLBrain] JSON parsing failed even after LLM recovery attempt. Final LLM response: '{new_llm_response_text[:200]}...'"
            )
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
                    print(
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
            print(f"[QwenVLBrain] Failed to parse monitoring JSON: {e}")
            return self._parse_monitoring_text(response_text)
        except Exception as e:
            print(f"[QwenVLBrain] Error parsing monitoring response: {e}")
            return {
                "result": "progress",
                "reason": "Error in parsing response",
            }

    def _format_prompt_for_planning(self, task: Task, skill_descriptions: str) -> str:
        """Format a prompt for Qwen VL planning."""
        prompt = (
            "You are a helpful robot task planner. Your goal is to create a JSON execution plan based on a task description and available skills.\n"
            "**Planning Guidelines:**\n"
            "1.  **Analyze Preconditions:** Carefully analyze the task and the initial state (from the image) to determine the necessary preconditions for each step.\n"
            "2.  **Logical Order:** Ensure the sequence of skills is logical. For example, interacting with a container (like pressing a button on a box) must happen before taking an object from it.\n"
            "3.  **Follow Skill Instructions:** Pay close attention to the instructions within each skill's description, especially regarding required preceding skills.\n"
            f"Task: {task.description}\n"
            "Available Skills:\n"
            f"{skill_descriptions}\n"
        )

        if self.insert_think_prompt:
            prompt += "First, think step-by-step about the user's request and the available skills. Lay out your reasoning for the chosen sequence of skills. Enclose your entire thinking process within `<think>` and `</think>` tags.\n"

        prompt += (
            "\n"
            "Then, provide the final execution plan as a JSON array.\n"
            "-   Each object in the array represents one step in the plan.\n"
            "-   Each object **must** include a `step` key, which is a sequential integer starting from 1, indicating the logical execution order.\n"
            "-   DO NOT USE ANY Placeholder in the JSON.\n"
            "\n"
            "## Example response format:\n"
        )

        if self.insert_think_prompt:
            prompt += (
                "<think>\n"
                "Here I will describe my thought process.\n"
                "1. First, I need to achieve X. The skill `skill_A` seems appropriate for this.\n"
                "2. Then, the user wants to do Y. The description for `skill_B` says it must be preceded by `skill_C`.\n"
                "3. Therefore, the logical sequence is `skill_A`, then `skill_C`, then `skill_B`.\n"
                "</think>\n"
            )

        prompt += (
            "```json\n"
            "[\n"
            "    {\n"
            '        "step": 1,\n'
            '        "method": "skill_A",\n'
            '        "params": {}\n'
            "    },\n"
            "    {\n"
            '        "step": 2,\n'
            '        "method": "skill_C",\n'
            '        "params": {\n'
            '            "param1": "value"\n'
            "        }\n"
            "    },\n"
            "    {\n"
            '        "step": 3,\n'
            '        "method": "skill_B",\n'
            '        "params": {}\n'
            "    }\n"
            "]\n"
            "```\n"
            "### Example response (move and open grasp):\n"
        )

        if self.insert_think_prompt:
            prompt += (
                "<think>\n"
                "1. The task requires moving to a target object and then opening the gripper.\n"
                "2. The skill `move_to_target_object` is suitable for the first step\n"  # , and it need to add tracking skill before it.
                "3. The skill `move_to_target_object` can set the gripper state to open after reaching the target.\n"
                "</think>\n"
            )

        prompt += (
            "```json\n"
            "[\n"
            # "    {\n"
            # '        "step": 1,\n'
            # '        "method": "object_tracking",\n'
            # '        "params": {\n'
            # '            "target_object": "white_hand_palm"\n'
            # "        }\n"
            # "    },\n"
            "    {\n"
            '        "step": 1,\n'
            '        "method": "move_to_target_object",\n'
            '        "params": {\n'
            '            "target_object": "white_hand_palm",\n'
            '            "gripper_state": "1" # 1 means open gripper after reaching the target\n'
            "        }\n"
            "    }\n"
            "]\n"
            "```\n"
        )

        return prompt

    def _format_prompt_for_replanning(
        self,
        task: Task,
        current_plan: SkillPlan,
        skill_history: list[dict],
        observation: Observation,
    ) -> str:
        """Format a prompt for Qwen VL planning."""
        current_plan_state = current_plan.pretty_print()
        last_skill_info = skill_history[-1]
        last_execution_info = (
            "Skill Index: " + str(last_skill_info["index"]) + "\n"
            "Skill Name: " + last_skill_info["name"] + "\n"
            "Skill Execution Result (skill reported): "
            + (last_skill_info.get("result", "unknown"))
            + "\n"
            "Execution Summary (Your expert analysis of the visual evidence): "
            + (
                last_skill_info.get(
                    "execution_summary", "No summary as it complete success"
                )
            )
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

        prompt = (
            # --- MODIFICATION START ---
            # Rephrased the goal to be about correction, not decision-making.
            "You are an expert robot task plan corrector. Your goal is to analyze the previous execution attempt and correct the plan's state *only if* there is a failure or the plan is incorrect. Otherwise, you will approve the plan to continue as is.\n"
            # --- MODIFICATION END ---
            f"**Original Task:** {task.description}\n"
            "\n"
            "**Available Skills:**\n"
            f"{self.skill_registry.get_skill_descriptions()}\n"
            "\n"
            "**Current Plan State:**\n"
            f"{current_plan_state}\n"
            "\n"
            "**Execution Info of last skill:**\n"
            f"{last_execution_info}\n"
            "\n"
            "**Current Robot Low-dimensional State:**\n"
            f"- pose: {eef_pos_str}, quat: {eef_quat_str}\n"
            "\n"
            "**Instructions for Your Response (Your Decision Process):**\n"
            "Follow this two-step process:\n"
            "**1. Analyze the Past (Outcome Analysis):**\n"
            "   - First, use the `Execution Summary` to determine the true outcome of the last skill. The visual evidence is the absolute truth.\n"
            "   - If the reported status (e.g., 'FAILED', 'TIMEOUT') contradicts the visual evidence, your first operation should be to correct it using `update_status`.\n"
            "\n"
            "**2. Validate the Future (Plan Validation):**\n"
            "   - After establishing the true state of the world, review the remaining `PENDING` skills in the plan.\n"
            "   - Ask yourself: 'Given the current scene and the outcome of the last skill, is this plan able to achieve the **Original Task**?'\n"
            "   - DO NOT USE Hypothetical target pose / Placeholder for actual target pose / Example target pose etc in skill params, as the params you give is the final version. You should always use realistic skill params.\n"
            "\n"
            "**3. Choose Your Operations:**\n"
            "   - If the plan needs correction based on your validation, provide a JSON array of operations (`insert`, `delete`, `retry`, etc.).\n"
            "   - **CRITICAL RULE FOR INDICES:** All `index` values in your entire JSON response MUST refer to the step numbers in the **'Current Plan State'** you were given. Do not try to calculate new indices that result from your own operations.\n"
            "   - **Only if** the last skill's status is correct AND the rest of the plan is valid, provide an empty array `[]`.\n"
        )

        if self.insert_think_prompt:
            prompt += (
                "\n"
                "First, think step-by-step in `<think>` tags. Then, generate a JSON array of operations to modify the plan.\n"
            )

        prompt += (
            "\n"
            "**Available Operations (index is starting from 0):**\n"
            "1.  **update_status**: Corrects the status of a skill if the visual summary contradicts the reported result (e.g., reported 'failed' but visually 'succeeded'). THIS IS YOUR MOST IMPORTANT CORRECTION TOOL.\n"
            '    - `operation`: "update_status"\n'
            "    - `index`: The index of the skill whose status needs correction.\n"
            '    - `new_status`: The TRUE status ("COMPLETED", "FAILED").\n'
            "2.  **insert**: Adds a new skill. IMPORTANT: A new skill is inserted *after* the specified `index` from the original plan.\n"
            '    - `operation`: "insert"\n'
            "    - `index`: new skill will be inserted AFTER this index\n"
            "    - `method`: The name of the skill to insert.\n"
            "    - `params`: The parameters for the skill.\n"
            "3.  **delete**: Removes an skill.\n"
            '    - `operation`: "delete", \n'
            "    - `index`: The index of the skill to delete.\n"
            "4. **retry**: Re-runs a skill that has a `FAILED` status. Only use this for skills that have already been attempted and failed.\n"
            '    - `operation`: "retry"\n'
            "    - `index`: The index of the skill to retry.\n"
            "\n"
            "NO NOT USE Hypothetical target pose for demonstration purposes in skill params OR ANY PLACEHOLDER, TEMPORARY VALUE, EXAMPLE VALUE etc. THE PARAMS YOU GIVE IS THE FINAL VERSION, WILL NOT BE MODIFIED ANYMORE. YOU SHOULD ALWAYS USE REALISTIC SKILL PARAMS.\n"
            "### Detailed Example: How `insert` and other operations work together\n"
            "Imagine the current plan is:\n"
            "```\n"
            "--- Skill Plan for Task: task_123 ---\n"
            "  Step 0: [COMPLETED] open_drawer({})\n"
            "  Step 1: [FAILED] grasp_object({})\n"
            "  Step 2: [PENDING] close_drawer({})\n"
            "```\n"
            "**Analysis:** The `grasp_object` at **original index 1** failed. We need to insert a `set_gripper` skill before it and then retry the grasp.\n"
            "**Logic and Indexing Rule:**\n"
            "1. To insert *before* index 1, we must specify `index: 0` for the `insert` operation.\n"
            "2. We also need to retry the failed `grasp_object` skill. According to the **CRITICAL RULE**, we must use its original index, which is `1`.\n"
            "3. The system will handle the index shifts automatically. You just need to provide the operations based on the original plan.\n"
        )

        if self.insert_think_prompt:
            prompt += (
                "<think>\n"
                "1. **Analyze the Past:** `grasp_object` at index 1 failed.\n"
                "2. **Validate the Future:** The plan is wrong. I need to insert `set_gripper` before the grasp and then retry the grasp.\n"
                "3. **Choose an Operation:** I will add an `insert` operation after original index `0`. Then I will add a `retry` operation for the `grasp_object` at its original index, which is `1`.\n"
                "</think>\n"
            )

        prompt += (
            "**Required Operations:**\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "operation": "insert",\n'
            '    "index": 0,\n'
            '    "method": "set_gripper",\n'
            '    "params": { "state": 1.0 } // Insert AFTER original index 0\n'
            "  },\n"
            "  {\n"
            '    "operation": "retry",\n'
            '    "index": 1 // Retry the skill at ORIGINAL index 1\n'
            "  }\n"
            "]\n"
            "```\n"
            "\n"
            "### Example Response (Plan is correct and should continue):\n"
        )

        if self.insert_think_prompt:
            prompt += (
                "<think>\n"
                "1. **Reflection:** The skill `open_box` at index 0 self-reported `completed`. The visual summary confirms that the red box's lid is open and the yellow spanner is visible inside.\n"
                "2. **Analysis:** The last skill succeeded. The current plan shows the next step is `grasp_spanner` at index 1, which is logical. The plan does not need any corrections.\n"
                "3. **Conclusion:** According to the CRITICAL RULE, since the plan is correct, I must output an empty array `[]` to allow the system to proceed automatically.\n"
                "</think>\n"
            )

        prompt += "```json\n[]\n```\n"

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
                print(
                    "[QwenVLBrain] No valid JSON array found in response, return empty plan."
                )
                return SkillPlan(task_id=task.id, skill_list=[])

            return SkillPlan(
                task_id=task.id,
                skill_list=plan_data,
                skill_monitoring_interval=self.skill_monitoring_interval,
            )
        except Exception as e:
            print(f"[QwenVLBrain] Error parsing initial plan response: {e}")
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
