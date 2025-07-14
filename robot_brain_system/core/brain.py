"""
Brain component that uses Qwen VL for task planning and monitoring.
"""

import sys
import time
import json
import os
from PIL import Image
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from robot_brain_system.core.types import (
    Task,
    SkillPlan,
    SystemStatus,
    Observation,
    SystemState,
)
from robot_brain_system.core.skill_manager import SkillRegistry
from robot_brain_system.utils import extract_json_from_text
from robot_brain_system.core.model_adapters_v2 import (
    TransformersAdapter,
    LMDeployAdapter,
    VLLMAdapter,
    OpenAIAdapter,
)


@dataclass
class BrainState:
    """Current state of the brain."""

    status: SystemStatus = SystemStatus.IDLE
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

    def fetch_history(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """
        获取系统提示（第一个条目）以及最后N轮对话。
        这个方法能正确处理历史记录较短的情况，不会产生重复条目。
        一轮对话包含一个用户输入和一个助手输出。
        """
        # 如果历史记录为空，直接返回空列表
        if not self.history:
            return []

        # 1. 系统提示总是被包含在内
        system_prompt = self.history[0:1]  # 使用切片[0:1]确保结果是列表

        # 2. 对话历史是除了系统提示外的所有内容
        conversation_history = self.history[1:]
        assert len(conversation_history) % 2 == 1, f"len of conversation_history must be odd, while get: {len(conversation_history)}, may need add user input firstly"

        # 3. 从对话历史中获取最后 n 轮 (n * 2 个条目)
        # Python的负索引切片很安全，如果条目数不足，它会返回所有可用的条目
        num_entries_to_fetch = last_n * 2 + 1 # 最后一条为当前的输入
        last_n_rounds = conversation_history[-num_entries_to_fetch:]

        # 4. 将系统提示和最近的对话历史合并返回
        return system_prompt + last_n_rounds

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
    
    def extract_images(self):
        print(f"[QwenVLBrain] Extract images from memory with {len(self.history)} entries")
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
        return content_images
    
    
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
        self.skill_registry: Optional[SkillRegistry] = None
        self.state = BrainState()

        # Qwen VL configuration
        self.qwen_config = config.get("qwen", {})
        self.model_name = self.qwen_config.get("model", "qwen-vl")
        self.api_key = self.qwen_config.get("api_key", "")
        self.base_url = self.qwen_config.get("base_url", "")
        self.model_path = self.qwen_config.get("model_path", "")
        self.adapter_type = self.qwen_config.get(
            "adapter_type", "mock"
        )  # "qwen_vl", "openai", or "mock"
        self.adapter_device = self.qwen_config.get("device", "auto")
        self.max_tokens = self.qwen_config.get("max_tokens", 512)

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
        
        self.FRAME_JUMP = 3
        self.FRAME_TOTAL = 2

    def initialize(self):
        """Initialize the brain component."""
        print("[QwenVLBrain] Initializing...")
        # Initialize model adapter
        self.model_adapter = None
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
        self.current_skill_monitor_time = 0
        print(f"[QwenVLBrain] Initialized with adapter: {self.adapter_type}")

    def _initialize_model_adapter(self):
        """根据配置初始化新的、标准化的模型适配器。"""
        adapter_type = (
            self.adapter_type
        )  # e.g., "transformers", "vllm", "openai", "lmdeploy"

        # 从主配置中获取模型和API相关的通用配置
        model_path = self.qwen_config.get("model_path")
        device = self.qwen_config.get("device", "auto")
        api_key = self.qwen_config.get("api_key")
        base_url = self.qwen_config.get("base_url")
        model_name = self.qwen_config.get("model")  # for OpenAI
        try:
            if adapter_type == "transformers":
                self.model_adapter = TransformersAdapter(
                    model_path=model_path,
                    device=device,
                    **self.qwen_config.get("transformers_adapter_args", {}),
                )
            elif adapter_type == "vllm":
                # vLLM的特定参数也可以从config传入
                self.model_adapter = VLLMAdapter(
                    model_path=model_path,
                    **self.qwen_config.get("vllm_adapter_args", {}),
                )
            elif adapter_type == "lmdeploy":
                # LMDeploy的特定参数
                self.model_adapter = LMDeployAdapter(
                    model_path=model_path,
                    **self.qwen_config.get("lmd_adapter_args", {}),
                )
            elif adapter_type == "openai":
                self.model_adapter = OpenAIAdapter(
                    model_name=model_name, api_key=api_key, base_url=base_url
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

    def set_skill_registry(self, skill_registry: SkillRegistry):
        """Set the skill registry for planning."""
        self.skill_registry = skill_registry
        print("[QwenVLBrain] Connected to skill registry")

    def set_system_state(self, state: SystemState):
        self.system_state = state

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

            # Get available skills
            available_skills = self.skill_registry.list_skills()

            # Use Qwen VL to analyze task and create plan
            plan = self._query_qwen_for_plan(
                task, self.skill_registry.get_skill_descriptions()
            )

            print(
                f"[QwenVLBrain] Created plan for task {task.id}: {len(plan.skill_sequence)} skills"
            )
            return plan

        except Exception as e:
            raise RuntimeError(f"Failed to create plan: {e}")

    def summary_skill_execution(self, skill_info: dict, only_image = True):
        # TODO 只提供image不提供text试试！
        # self.monitor_memory.add_user_input(
        #     contents=[f"skill execution result: {skill_info['result']}"]
        # )
        content = []
        if only_image:
            memory_content = self.monitor_memory.extract_images()
        else:
            memory_content = self.monitor_memory.format_memory_content()
        content.append(
            (
                "The robot are try to orchestrate skills to accomplish a task.\n\n"
                "## Skill Execution Context Info:\n"
                f"Original Task: {self.state.current_task.description}\n"
                f"Current Skill: {skill_info['name']}\n"
                f"Skill Description: {skill_info['description']}\n"
                f"Skill Parameters: {skill_info['parameters']}\n"
                f"Skill Criterion: {skill_info['criterion']}\n\n"
                f"Skill Execution Result (reported by skill itself): {skill_info['result']}\n\n"
            ))
        content.extend(memory_content)
        content.append(
            (
                "## Summary memory content as follows:\n"
                "Extract key information as bullet points.\n"
                "Your focus may include the following points: \n"
                # "(timeout not means failed, timeout only means the skill is finished while whather successed or not is unkown! you need juedge by yourself from those Image, especially last Image)"
                f"0. Did the task finished? Please Recheck the skill execution result! Did \'{skill_info['criterion']['successed']}\' is real happened nor not?\n"
                "1. Did the execution of the skill achieve the intended goal?\n"
                "2. How does the scene change as the skill is executed?\n"
                "3. Reflection skills execution process.\n"
                "4. What does the scene look like now?\n"
                "5. Any other points you think could be mentioned?\n\n"
                "## Output template:\n"
                "[focus 1] Specific summary content 1\n"
                "[focus 2] Specific summary content 2\n..."
            )
        )

        sum_memory = BrainMemory()
        sum_memory.add_system_prompt("You are a professional dialogue summarizer.")
        sum_memory.add_user_input(contents=content)
        response_text, _ = self.model_adapter.generate(sum_memory.history)
        return response_text

    # @retry
    def replan_task(
        self,
        task: Task,
        last_plan_info: SkillPlan,
        skill_history: list[dict],
        observation: Observation,
    ):
        """
        Replan a skill execution plan for the given task.

        Args:
            task: The task to plan for
            last_plan_info:
            skill_history:
            observation:
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
                self.state.status = SystemStatus.THINKING
                text_prompt = self._format_prompt_for_replanning(
                    task, last_plan_info, skill_history
                )
                print("--- Replan Prompt ---")
                print(text_prompt)
                print("\n")
                inspector_rgb = (
                    observation.data["policy"]["camera_side"][0].cpu().numpy()
                )
                current_image = Image.fromarray(inspector_rgb)
                if self.visualize:
                    import matplotlib.pyplot as plt

                    plt.imshow(inspector_rgb)
                    plt.axis("off")
                    plt.savefig(
                        os.path.join(
                            self.log_path,
                            f"{len(self.system_state.plan_history)}_replan_{task.description}_input.png",
                        )
                    )
                # Generate response
                self.replan_memory.add_user_input(contents=[text_prompt, current_image])
                print(f"[QwenVLBrain] 开始为任务进行再规划: {task.description}")
                response_text, _ = self.model_adapter.generate(
                    history=self.replan_memory.fetch_history(
                        last_n=3
                    ),  # 获取最近的对话历史
                    max_tokens=self.max_tokens,
                )
                self.replan_memory.add_assistant_output(response_text)
                # Parse the response to extract skill plan
                plan = self._parse_plan_response(response_text, task)
                if not plan.skill_sequence:
                    # no plan was created, mean task finished successed
                    self.interrupt_task(
                        "[QwenVLBrain] No plan is created, mean task finished successed!"
                    )
                    return plan
                self.state.current_plan = plan
                self.state.current_skill_index = 0
                self.state.status = SystemStatus.EXECUTING
                self.initial_monitor()
                print(
                    f"[QwenVLBrain] Generated plan using {self.adapter_type}: {plan.skill_sequence}"
                )
                return plan

            except Exception as e:
                import traceback

                print(f"[QwenVLBrain] Error in planning: {e}")
                traceback.print_exc()
                raise RuntimeError(f"Failed to create plan: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to create plan: {e}")

    def execute_task(self, task: Task) -> SkillPlan:
        """
        Start executing a task.

        Args:
            task: Task to execute

        Returns:
            The execution plan
        """
        try:
            self.state.status = SystemStatus.THINKING
            self.state.current_task = task

            # Create execution plan
            plan = self.plan_task(task)
            self.state.current_plan = plan
            self.state.current_skill_index = 0
            self.state.status = SystemStatus.EXECUTING
            self.initial_monitor()
            print(f"[QwenVLBrain] Started executing task: {task.description}")
            return plan

        except Exception as e:
            self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            raise

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
        if not self.state.current_plan or self.state.current_skill_index >= len(
            self.state.current_plan.skill_sequence
        ):
            return None

        skill_name = self.state.current_plan.skill_sequence[
            self.state.current_skill_index
        ]
        skill_params = self.state.current_plan.skill_params[
            self.state.current_skill_index
        ]
        skill_info = self.skill_registry.get_skill_info(skill_name)
        assert skill_info is not None

        # skill info:
        # return {
        #     "name": skill.name,
        #     "type": skill.skill_type.value,
        #     "execution_mode": skill.execution_mode.value,
        #     "description": skill.description,
        #     "timeout": skill.timeout,
        #     "requires_env": skill.requires_env,
        #     "criterion": skill.criterion,
        #     "enable_monitoring": skill.enable_monitoring,
        #     "function_name": skill.function.__name__,  # function object itself is not easily serializable
        # }
        skill_info.update(
            {
                "parameters": skill_params,
                "index": self.state.current_skill_index,
            }
        )
        return skill_info

    def advance_skill(self):
        """Advance to the next skill in the plan."""
        if self.state.current_plan:
            self.state.current_skill_index += 1
            if self.state.current_skill_index >= len(
                self.state.current_plan.skill_sequence
            ):
                # Plan completed
                self.state.status = SystemStatus.IDLE
                self.state.current_task = None
                self.state.current_plan = None
                self.state.current_skill_index = 0
                print("[QwenVLBrain] Task execution completed")
                return
            self.initial_monitor()

    def should_monitor(self,obs_history) -> bool:
        """Check if it's time to monitor the current skill execution."""
        if self.state.status != SystemStatus.EXECUTING or not self.state.current_plan:
            return False
        current_skill_info = self.get_next_skill()
        if not current_skill_info["enable_monitoring"]:
            return False
        if not self.state.current_plan or not self.state.current_task:
            print(f"[QwenVLBrain] should_monitor: No active task")
            return False
        if len(obs_history) == 0:
            print("[QwenVLBrain] should_monitor: No observation data available for monitoring")
            return False
        if not (
            self.calculate_indicesv2(self.FRAME_TOTAL, len(obs_history), self.FRAME_JUMP * self.FRAME_TOTAL)
        ):
            print("[QwenVLBrain] should_monitor: No observation data available for monitoring")
            return False
        
        current_time = time.time()
        return (
            current_time - self.state.last_monitoring_time
        ) >= self.state.current_plan.skill_monitoring_interval

    def monitor_skill_execution(self, obs_history: list[Any] = []) -> Dict[str, Any]:
        """
        Monitor current skill execution and make decisions. WE HAVED SUPPORTED MONITORING THE WHOLE TASK EXECUTION

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

            self.state.status = SystemStatus.MONITORING

            # Use Qwen VL to analyze current situation
            result = self._query_qwen_for_monitoring(
                self.state.current_task, current_skill, obs_history
            )
            self.state.status = SystemStatus.EXECUTING
            return result

        except Exception as e:
            self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            return {"result": "progress", "reason": str(e)}

    def interrupt_task(self, reason: str = "User interrupt"):
        """Interrupt the current task execution."""
        if self.state.current_task:
            print(f"[QwenVLBrain] Interrupting task: {reason}")
            self.state.status = SystemStatus.IDLE
            self.state.current_task = None
            self.state.current_plan = None
            self.state.current_skill_index = 0
            self.monitor_memory.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current brain status."""
        return {
            "status": self.state.status.value,
            "current_task": self.state.current_task.description
            if self.state.current_task
            else None,
            "current_skill_index": self.state.current_skill_index,
            "total_skills": len(self.state.current_plan.skill_sequence)
            if self.state.current_plan
            else 0,
            "error_message": self.state.error_message,
            "last_monitoring": self.state.last_monitoring_time,
        }



    def _query_qwen_for_plan(self, task: Task, skill_descriptions: str) -> SkillPlan:
        if self.model_adapter is None:
            print("[QwenVLBrain] Mock模式：返回一个预设的计划。")
            return self._mock_plan_task(task, skill_descriptions)

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
            )
            plan = self._parse_plan_response(response_text, task)
            print(
                f"[QwenVLBrain] 使用 {self.adapter_type} 生成了计划，包含 {len(plan.skill_sequence)} 个技能。"
            )
            return plan
        except Exception as e:
            import traceback

            print(f"[QwenVLBrain] 规划时出错: {e}")
            traceback.print_exc()
            # 出错时回退
            return self._mock_plan_task(task, skill_descriptions)

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
    ) -> Dict[str, Any]:
        """
        Query Qwen VL to make monitoring decisions.

        Uses the configured model adapter to analyze the current situation
        and output the current skill execution status.
        """
        if self.model_adapter is None:
            # Fallback to mock implementation
            return self._mock_monitoring_decision(task, current_skill, obs_history)
        assert type(obs_history) is list
        try:
            if self.visualize:
                inspector_rgb = (obs_history[-1].data["policy"]["camera_side"][0].cpu().numpy())
                front_rgb = obs_history[-1].data["policy"]["camera_top"][0].cpu().numpy()
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

            def calculate_indices(jump, total, available):
                if available >= total * jump + 1:
                    return list(range(-total * jump + 1, 0, jump))
                else:
                    return None



            # TODO 提取 obs 这部分应该要放在外面才对，这里面只进行query_qwen的逻辑
            # 计算可用的观察帧数
            available_frames = len(obs_history)

            indices = self.calculate_indicesv2(self.FRAME_TOTAL, available_frames, self.FRAME_JUMP * self.FRAME_TOTAL)

            # 按索引顺序排序（从旧到新）
            indices.sort()
            # 提取帧
            for frame_index in indices:
                video_frames_inspect.append(
                    Image.fromarray(
                        obs_history[frame_index]
                        .data["policy"]["camera_side"][0]
                        .cpu()
                        .numpy()
                    )
                )
                video_frames_front.append(
                    Image.fromarray(
                        obs_history[frame_index]
                        .data["policy"]["camera_top"][0]
                        .cpu()
                        .numpy()
                    )
                )
            if self.visualize:

                # 获取原始视频帧率（示例值，需根据实际情况替换）
                original_fps = 30
                duration = 1000 // original_fps  # 每帧持续时间（毫秒）

                all_frames = []
                max_length = max(len(video_frames_inspect), len(video_frames_front))
                last_inspect = video_frames_inspect[-1] if video_frames_inspect else None
                last_front = video_frames_front[-1] if video_frames_front else None

                for i in range(max_length):
                    frame_inspect = video_frames_inspect[i] if i < len(video_frames_inspect) else last_inspect
                    frame_front = video_frames_front[i] if i < len(video_frames_front) else last_front

                    if frame_inspect and frame_front:
                        # 保持宽高比对齐
                        target_height = frame_inspect.size[1]
                        inspect_ratio = frame_inspect.size[0] / target_height
                        front_ratio = frame_front.size[0] / target_height

                        inspect_width = int(target_height * inspect_ratio)
                        front_width = int(target_height * front_ratio)

                        combined = Image.new('RGB', (inspect_width + front_width, target_height), (0, 0, 0))
                        combined.paste(frame_inspect.resize((inspect_width, target_height)), (0, 0))
                        combined.paste(frame_front.resize((front_width, target_height)), (inspect_width, 0))
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
                        optimize=False  # 保留图像质量
                    )
            image_data = Image.fromarray(
                inspector_rgb
            )
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
                temp_memory.fetch_history(), max_tokens=self.max_tokens
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
        prompt = f"""
You are a helpful robot task planner. Your goal is to create a JSON execution plan based on a task description and available skills.

**Planning Guidelines:**
1.  **Analyze Preconditions:** Carefully analyze the task and the initial state (from the image) to determine the necessary preconditions for each step.
2.  **Logical Order:** Ensure the sequence of skills is logical. For example, interacting with a container (like pressing a button on a box) must happen before taking an object from it.
3.  **Follow Skill Instructions:** Pay close attention to the instructions within each skill's description, especially regarding required preceding skills.

Task: {task.description}

Available Skills:
{skill_descriptions}

First, think step-by-step about the user's request and the available skills. Lay out your reasoning for the chosen sequence of skills. Enclose your entire thinking process within `<thinking>` and `</thinking>` tags.

After your thinking process, provide the final execution plan as a JSON array.
-   Each object in the array represents one step in the plan.
-   Each object **must** include a `step` key, which is a sequential integer starting from 1, indicating the logical execution order.
-   The JSON should be the only thing after the closing `</thinking>` tag.

Example response format:
<thinking>
Here I will describe my thought process.
1. First, I need to achieve X. The skill `skill_A` seems appropriate for this.
2. Then, the user wants to do Y. The description for `skill_B` says it must be preceded by `skill_C`.
3. Therefore, the logical sequence is `skill_A`, then `skill_C`, then `skill_B`.
</thinking>
```json
[
    {{
        "step": 1,
        "method": "skill_A",
        "params": {{}}
    }},
    {{
        "step": 2,
        "method": "skill_C",
        "params": {{
            "param1": "value"
        }}
    }},
    {{
        "step": 3,
        "method": "skill_B",
        "params": {{}}
    }}
]
"""
        return prompt

    def _format_prompt_for_replanning(
        self, task: Task, last_plan_info: SkillPlan, skill_history: list[dict]
    ) -> str:
        """Format a prompt for Qwen VL planning."""
        skills_execution_info = "\n\n".join(
            [
                f"""
Skill Name: {skill_info["name"]}
Skill Parameters: {skill_info["parameters"]}
Skill Criterion: {skill_info["criterion"]}
Skill Execution Result (skill reported): {skill_info["result"] if skill_info["result"] != "timeout" else "unkown"}
Execution Summary: {skill_info.get("execution_summary", "No summary as it compelete success")}
"""
                for skill_info in skill_history
            ]
        )

        prompt = f"""
You are a helpful robot task planner. Your goal is to create a JSON execution plan based on a task description, the result of a previous attempt, and available skills.

Task: {task.description}

## Available Skills:
{self.skill_registry.get_skill_descriptions()}

## Last Planning Info:
Skill Sequence: {last_plan_info.skill_sequence}
Skill Params: {last_plan_info.skill_params}

## Skills Execution Info from Last Attempt:
{skills_execution_info}

## Instructions for Your Response
First, think step-by-step about the previous execution results and the current situation shown in the image. Your thinking process should include:
1.  **Reflection:** What was the outcome of the last plan? Did it succeed, fail, or result in an unexpected state?
2.  **Analysis:** Based on your reflection and the current visual evidence, what needs to happen next? Is the task finished? Does a step need to be retried? Or is a new approach required?
Enclose your entire thinking process within `<thinking>` and `</thinking>` tags.

After your thinking process, provide the final execution plan as a JSON array.
-   Each object in the array represents one step in the plan.
-   Each object must include a `step` key (a sequential integer starting from 1), a `method` key (the skill name), and a `params` key.
-   If you determine the task is finished, output an empty JSON array `[]`.
-   The JSON should be the only thing after the closing `</thinking>` tag.

### Example Response Format:
<thinking>
1.  **Reflection:** YOUR REFLECTION ON THE PREVIOUS EXECUTION AND CURRENT STATE.
2.  **Analysis:** YOUR ANALYSIS FOR WHY YOU ARE CREATING THE NEW PLAN.
3.  **Conclusion:** YOUR CONCLUSION ON WHAT NEEDS TO BE DONE NEXT.
</thinking>
```json
[
    {{
        "step": 1,
        "method": "skill1",
        "params": {{
            "param1_of_skill1": "value",
            "param2_of_skill1": "value"
        }}
    }},
    {{
        "step": 2,
        "method": "skill2",
        "params": {{}} 
    }}
]
```
"""

        # Example response:
        # Refection: The previous assemble_object skill execution result was timeout. However, the execution summary indicates that the task was completed successfully and the red object is completely wrapped around the black pillar. This suggests a discrepancy between the skill's internal timeout mechanism and the actual state of the environment. The skill was successfully executed and the task is finished.
        # Plan:
        # ```json
        # {{
        #     "skill_sequence": [],
        #     "skill_params": [],
        #     "skill_monitoring_interval": 1.0,
        #     "analysis": "The previous skill 'assemble_object' reported a 'timeout', but the detailed execution summary clearly states that 'The red object is completely wrapped around the black pillar.' and 'The task was completed successfully'. This indicates that despite the timeout, the desired outcome of the task has been achieved. Therefore, no further actions are needed."
        # }}
        # ```
        # """
        return prompt

    def _format_system_prompt_for_monitoring(
        self,
        task: Task,
        current_skill: Dict[str, Any],
        observation: Optional[Any] = None,
    ) -> str:
        """Format a prompt for Qwen VL monitoring."""
        prompt = f"""The robot are try to orchestrate skills to accomplish a task.
You are monitoring robot's SKILL execution. Analyze the current situation and output the current execution result.
You should only consider the execution of CURRENT SKILL rather than the completion of tasks.

Current Skill: {current_skill["name"]}
Skill Description: {current_skill["description"]}
Skill Parameters: {current_skill["parameters"]}
Skill Criterion: {current_skill["criterion"]}

For each round of conversation, you will receive the current observation in Image format, You should analyze the image and response with a JSON object containing:
- result: The value for this field must be one of the options from `{current_skill["criterion"].keys()}`
- reason: Explanation for the result.
- current_scene_state: Description what happened in the scene, based on the observation
- confidence: Float between 0 and 1

Example response:
```json
{{
    "result": "failed",
    "reason": "The object was not grasped correctly",
    "current_scene_state", "The robot attempted to grasp the object but it slipped",
    "confidence": 0.9
}}
```

Next, Let's start first monitoring round.
"""
        # Available Skills: {chr(10).join(f"- {skill}" for skill in self.skill_registry.list_skills())}
        # Original Task: {task.description}
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
                return SkillPlan(
                    task_id=task.id,
                    skill_sequence=[],
                    skill_params=[],
                    skill_monitoring_interval=self.skill_monitoring_interval,
                    expected_duration=0.0,
                )

            try:
                sorted_plan_data = sorted(
                    plan_data, key=lambda item: item.get("step", sys.maxsize)
                )
            except TypeError:
                # This handles cases where 'step' is not a number, etc.
                print(
                    "[QwenVLBrain] Error sorting plan steps due to invalid 'step' values. Falling back to empty plan."
                )
                return SkillPlan(
                    task_id=task.id,
                    skill_sequence=[],
                    skill_params=[],
                    skill_monitoring_interval=self.skill_monitoring_interval,
                    expected_duration=0.0,
                )

            # 从对象列表中构建 skill_sequence 和 skill_params
            skill_sequence = []
            skill_params = []
            for skill_object in sorted_plan_data:
                if isinstance(skill_object, dict):
                    # Extract skill name, skip if 'method' key is missing
                    skill_name = skill_object.get("method")
                    if skill_name:
                        skill_sequence.append(skill_name)
                        # Extract params, defaulting to an empty dict if 'params' is missing or null
                        params = skill_object.get("params", {})
                        if (
                            params is None
                        ):  # Handle cases where model outputs "params": null
                            params = {}
                        skill_params.append(params)
                    else:
                        # It's good practice to warn about malformed steps
                        print(
                            f"[QwenVLBrain] Warning: A step object is missing the 'method' key and will be ignored: {skill_object}"
                        )
                else:
                    print(
                        f"[QwenVLBrain] Warning: Found a non-dictionary item in the plan data: {skill_object}"
                    )

            skill_monitoring_interval = self.skill_monitoring_interval

            return SkillPlan(
                task_id=task.id,
                skill_sequence=skill_sequence,
                skill_params=skill_params,
                skill_monitoring_interval=skill_monitoring_interval,
                expected_duration=len(skill_sequence) * 10.0,
            )

        except Exception as e:
            # 保持强大的异常处理
            print(f"[QwenVLBrain] Error parsing plan response: {e}")
            print(f"[QwenVLBrain] Original response was: {response_text[:500]}...")
            print("[QwenVLBrain] Falling back to empty plan")
            return SkillPlan(
                task_id=task.id,
                skill_sequence=[],
                skill_params=[],
                skill_monitoring_interval=self.skill_monitoring_interval,
                expected_duration=0.0,
            )

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
