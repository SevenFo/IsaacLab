"""
Brain component that uses Qwen VL for task planning and monitoring.
"""

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

try:
    import robot_brain_system.core.model_adapters

    MODEL_ADAPTERS_AVAILABLE = True
except ImportError:
    MODEL_ADAPTERS_AVAILABLE = False
    print(
        "[QwenVLBrain] Warning: Model adapters not available, using mock implementation"
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
        """Fetch the last N entries from the memory."""
        return self.history[0:1] + (
            self.history[-last_n * 2 :]  # n round
            if len(self.history) - 1 >= last_n * 2
            else self.history
        )

    def clear(self):
        """Clear the memory."""
        self.history = []
        print("[BrainMemory] Memory cleared.")


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

    def initialize(self):
        """Initialize the brain component."""
        print("[QwenVLBrain] Initializing...")
        # Initialize model adapter
        self.model_adapter = None
        if MODEL_ADAPTERS_AVAILABLE:
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
        else:
            self.adapter_type = "mock"
        self.current_skill_monitor_time = 0
        print(f"[QwenVLBrain] Initialized with adapter: {self.adapter_type}")

    def _initialize_model_adapter(self):
        """Initialize the appropriate model adapter based on configuration."""
        if self.adapter_type == "qwen_vl":
            if not self.model_path:
                raise ValueError("model_path is required for QwenVL adapter")
            from .model_adapters import QwenVLAdapter

            self.model_adapter = QwenVLAdapter(self.model_path, self.adapter_device)
            print(
                f"[QwenVLBrain] Initialized QwenVL adapter with model: {self.model_path}"
            )

        elif self.adapter_type == "openai":
            if not self.api_key:
                raise ValueError("api_key is required for OpenAI adapter")
            from .model_adapters import OpenAIAdapter

            model_name = self.model_name if self.model_name != "qwen-vl" else None
            self.model_adapter = OpenAIAdapter(
                api_key=str(self.api_key),
                base_url=str(self.base_url) if self.base_url else None,
                model_name=str(model_name) if model_name else None,
            )
            print(
                f"[QwenVLBrain] Initialized OpenAI adapter with model: {model_name or 'default'}"
            )
        elif self.adapter_type == "lmd":
            if not self.model_path:
                raise ValueError("model_path is required for LMD adapter")
            from .model_adapters import LMDAdapter

            self.model_adapter = LMDAdapter(
                model_path=self.model_path,
            )
            print(
                f"[QwenVLBrain] Initialized LMD adapter with model: {self.model_path}"
            )

        elif self.adapter_type == "mock":
            self.model_adapter = None
            print("[QwenVLBrain] Using mock implementation")

        else:
            raise ValueError(f"Unknown adapter type: {self.adapter_type}")

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

    def summary_skill_execution(self, skill_info: dict):
        # TODO 只提供image不提供text试试！
        self.monitor_memory.add_user_input(
            contents=[f"skill execution result: {skill_info['result']}"]
        )
        content = self._format_memory_content(memory=self.monitor_memory)
        content.append(
            (
                "The robot are try to orchestrate skills to accomplish a task.\n\n"
                "## Skill Execution Context Info:\n"
                f"Original Task: {self.state.current_task.description}\n"
                f"Current Skill: {skill_info['name']}\n"
                f"Skill Description: {skill_info['discription']}\n"
                f"Skill Parameters: {skill_info['parameters']}\n"
                f"Skill Criterion: {skill_info['criterion']}\n\n"
                "## Summary memory content as follows:\n"
                "Extract key information as bullet points.\n"
                "Your focus may include the following points: \n"
                "(timeout not means failed, timeout only means the skill is finished while whather successed or not is unkown! you need juedge by yourself from those Image, especially last Image)"
                "0. Did the task finished? Based on the Orignal Task and those scene image.\n"
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
        response_text, _ = self.model_adapter.generate_response(sum_memory.history)
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
                    observation.data["rgb_camera"]["inspector"][0].cpu().numpy()
                )
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
                self.replan_memory.add_user_input(
                    contents=[text_prompt, Image.fromarray(inspector_rgb)]
                )
                print(f"[QwenVLBrain] Replanning task: {task.description}")
                response_text, _ = self.model_adapter.generate_response(
                    input_data=self.replan_memory.fetch_history(last_n=3)
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
        assert skill_info is not None

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
        skill_criterion = skill_info["criterion"]

        return {
            "name": skill_name,
            "parameters": skill_params,
            "criterion": skill_criterion,
            "discription": skill_info["description"],
            "index": self.state.current_skill_index,
        }

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

    def should_monitor(self) -> bool:
        """Check if it's time to monitor the current skill execution."""
        if self.state.status != SystemStatus.EXECUTING or not self.state.current_plan:
            return False
        current_time = time.time()
        return (
            current_time - self.state.last_monitoring_time
        ) >= self.state.current_plan.skill_monitoring_interval

    def monitor_skill_execution(
        self, current_observation: list[Any] = []
    ) -> Dict[str, Any]:
        """
        Monitor current skill execution and make decisions. WE HAVED SUPPORTED MONITORING THE WHOLE TASK EXECUTION

        Args:
            current_observation: Current environment observation

        Returns:
            Dict with monitoring decision (continue, interrupt, retry, etc.)
        """
        try:
            self.state.last_monitoring_time = time.time()

            if not self.state.current_plan or not self.state.current_task:
                return {"action": "continue", "reason": "No active task"}

            if self.current_skill_monitor_time < self.monitor_waiting_times:
                self.current_skill_monitor_time += 1
                return {
                    "action": "continue",
                    "reason": f"current_skill_monitor_time: {self.current_skill_monitor_time}",
                }

            if len(current_observation) == 0:
                print("[QwenVLBrain] No observation data available for monitoring")
                return {"action": "continue", "reason": "No observation data"}

            # Get current skill info
            current_skill = self.get_next_skill()
            if not current_skill:
                return {"action": "complete", "reason": "All skills completed"}

            self.state.status = SystemStatus.MONITORING

            # Use Qwen VL to analyze current situation
            decision = self._query_qwen_for_monitoring(
                self.state.current_task, current_skill, current_observation
            )
            self.state.status = SystemStatus.EXECUTING
            return decision

        except Exception as e:
            self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            return {"action": "error", "reason": str(e)}

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

    def _format_memory_content(self, memory: BrainMemory) -> List:
        print(f"[QwenVLBrain] Formatting memory with {len(memory.history)} entries")
        chat_content = []
        chat_content.append("## memory content:\n\n")
        for item in memory.history:
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

    def _query_qwen_for_plan(self, task: Task, skill_descriptions: str) -> SkillPlan:
        """
        Query Qwen VL to create a skill execution plan.

        Uses the configured model adapter to generate a skill execution plan
        based on the task description and available skills.
        """
        if self.model_adapter is None:
            # Fallback to mock implementation
            return self._mock_plan_task(task, skill_descriptions)

        try:
            # Format prompt for planning
            prompt = self._format_prompt_for_planning(task, skill_descriptions)

            # Prepare input for the model adapter
            input_data = self.model_adapter.prepare_input(
                text=prompt,
                image=task.image if task.image else None,  # Pass None if no image
            )
            # Generate response
            print(
                f"[QwenVLBrain] Query model to generate plan for [{task.description}]"
            )
            response_text, _ = self.model_adapter.generate_response(
                input_data, max_tokens=self.max_tokens
            )

            # Parse the response to extract skill plan
            plan = self._parse_plan_response(response_text, task)

            print(
                f"[QwenVLBrain] Generated plan using {self.adapter_type}: {plan.skill_sequence}"
            )
            return plan

        except Exception as e:
            print(f"[QwenVLBrain] Error in planning: {e}")
            print("[QwenVLBrain] Falling back to mock implementation")
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

    def _query_qwen_for_monitoring(
        self,
        task: Task,
        current_skill: Dict[str, Any],
        observation: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Query Qwen VL to make monitoring decisions.

        Uses the configured model adapter to analyze the current situation
        and decide whether to continue, interrupt, retry, or modify the execution.
        """
        if self.model_adapter is None:
            # Fallback to mock implementation
            return self._mock_monitoring_decision(task, current_skill, observation)
        assert type(observation) is list
        try:
            if self.visualize:
                inspector_rgb = (
                    observation[-1].data["rgb_camera"]["inspector"][0].cpu().numpy()
                )
                import matplotlib.pyplot as plt

                plt.imshow(inspector_rgb)
                plt.axis("off")
                plt.savefig(
                    os.path.join(
                        self.log_path,
                        f"{len(self.system_state.plan_history)}_{self.state.current_skill_index}_monitor_{current_skill['name']}_input_{len(self.monitor_memory.history)}.png",
                    )
                )
            video_frames = []

            def calculate_indices(jump, total, available):
                if available >= total * jump + 1:
                    return list(range(-total * jump + 1, 0, jump))
                else:
                    return None

            def calculate_indicesv2(total, available, mini_available):
                if available < mini_available or total > available or total < 2:
                    return None

                step = (available - 1) / (total - 1)
                indices = [round(i * step) for i in range(total)]
                indices[-1] = available - 1  # Ensure the last index is correct
                return indices

            # 计算可用的观察帧数
            available_frames = len(observation)
            jump = 6
            total = 8
            indices = []
            if not (
                indices := calculate_indicesv2(total, available_frames, jump * total)
            ):
                return {
                    "action": "not enough",
                    "reason": "not enough availabel frames",
                }

            # 按索引顺序排序（从旧到新）
            indices.sort()
            # 提取帧
            for frame_index in indices:
                video_frames.append(
                    Image.fromarray(
                        observation[frame_index]
                        .data["rgb_camera"]["inspector"][0]
                        .cpu()
                        .numpy()
                    )
                )
            if self.visualize:
                gif_path = os.path.join(
                    self.log_path,
                    f"{len(self.system_state.plan_history)}_{self.state.current_skill_index}_monitor_{current_skill['name']}_input_{len(self.monitor_memory.history)}.gif",
                )
                video_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=video_frames[1:],
                    duration=500,
                    loop=0,
                    optimize=True,  # 优化文件大小
                    quality=85,  # 质量参数（0-100）
                )
            image_data = Image.fromarray(
                inspector_rgb
            )  # Prepare input for the model adapter
            self.monitor_memory.add_user_input(
                contents=[
                    "belowing is current scene video observation",
                    video_frames,
                ]
            )
            task.image = image_data  # Update task with image
            # Generate response
            print(
                f"[QwenVLBrain] Monitoring task: {task.description}, skill: {current_skill['name']}"
            )
            response_text, _ = self.model_adapter.generate_response(
                self.monitor_memory.fetch_history(last_n=3),
                max_tokens=self.max_tokens // 2,  # Use fewer tokens for monitoring
            )
            self.monitor_memory.add_assistant_output(response_text)
            # Parse the response to extract monitoring decision
            decision = self._parse_monitoring_response(response_text)
            print(
                f"[QwenVLBrain] Monitoring decision using {self.adapter_type}: {decision['action']}"
            )
            return decision

        except Exception as e:
            print(f"[QwenVLBrain] Error in monitoring: {e.__class__.__name__}: {e}")
            import traceback

            traceback.print_exc()
            print("[QwenVLBrain] Falling back to mock monitoring")
            return self._mock_monitoring_decision(task, current_skill, observation)

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

    def _parse_json_response(self, response_text: str) -> Dict | List:
        """
        从响应文本中提取并解析JSON内容.
        如果初次解析失败,会尝试请求模型生成有效的JSON再次解析.

        Args:
            response_text: 包含可能JSON内容的原始文本

        Returns:
            解析成功的字典或列表对象，失败返回None
        """
        parsed_data = extract_json_from_text(response_text)

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
            input_data = self.model_adapter.prepare_input(
                text=error_handling_prompt,
                image=None,  # No image for error handling
            )

            new_llm_response_text, _ = self.model_adapter.generate_response(
                input_data=input_data, max_tokens=self.max_tokens
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
        parsed_data_after_recovery = extract_json_from_text(new_llm_response_text)

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
                # 原有字段处理逻辑保持不变...
                action = decision_data.get("action", "continue")
                reason = decision_data.get("reason", "No reason provided")
                confidence = decision_data.get("confidence", 0.5)

                valid_actions = [
                    "continue",
                    "interrupt",
                    "successed",
                    "failed",
                    "retry",
                    "modify",
                    "complete",
                    "error",
                    "not determinable",
                ]
                if action not in valid_actions:
                    print(
                        f"[QwenVLBrain] Invalid action '{action}', defaulting to 'continue'"
                    )
                    action = "continue"

                return {
                    "action": action,
                    "reason": reason,
                    "confidence": confidence,
                }
            else:
                # 原有文本解析逻辑
                return self._parse_monitoring_text(response_text)

        except json.JSONDecodeError as e:
            print(f"[QwenVLBrain] Failed to parse monitoring JSON: {e}")
            return self._parse_monitoring_text(response_text)
        except Exception as e:
            print(f"[QwenVLBrain] Error parsing monitoring response: {e}")
            return {
                "action": "continue",
                "reason": "Error in parsing response",
            }

    def _parse_monitoring_text(self, response_text: str) -> Dict[str, Any]:
        """Parse monitoring decision from text response."""
        response_lower = response_text.lower()

        # Simple keyword-based parsing
        if any(
            word in response_lower for word in ["stop", "interrupt", "halt", "abort"]
        ):
            return {
                "action": "interrupt",
                "reason": "Model suggested interruption",
            }
        elif any(word in response_lower for word in ["retry", "again", "restart"]):
            return {"action": "retry", "reason": "Model suggested retry"}
        elif any(word in response_lower for word in ["complete", "done", "finished"]):
            return {
                "action": "complete",
                "reason": "Model indicated completion",
            }
        elif any(word in response_lower for word in ["error", "fail", "problem"]):
            return {"action": "error", "reason": "Model detected error"}
        else:
            return {
                "action": "continue",
                "reason": "Model suggested continuation",
            }

    def _format_prompt_for_planning(self, task: Task, skill_descriptions: str) -> str:
        """Format a prompt for Qwen VL planning."""
        prompt = f"""
You are a robot task planner. Given a task description and available skills, create an execution plan.

Task: {task.description}

Available Skills:
{skill_descriptions}

Please respond with a JSON array representing a sequence of skills to execute.
Each element in the array should be a JSON object with the following keys:
- "method": A string representing the name of the skill to be called.
- "params": A JSON object containing the parameters for that skill. If a skill has no parameters, provide an empty object {{}}.

Example response:
```json
[
    {{
        "method": "skill1",
        "params": {{
            "param1_of_skill1": "value",
            "param2_of_skill1": "value"
        }}
    }},
    {{
        "method": "skill2",
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
Skill Execution Result: {skill_info["result"] if skill_info["result"] != "timeout" else "unkown"}
Execution Summary: {skill_info.get("execution_summary", "No summary as it compelete success")}
"""
                for skill_info in skill_history
            ]
        )

        prompt = f"""
You are a robot task planner. Given a task description and available skills, create an execution plan.

Task: {task.description}

## Available Skills:
{self.skill_registry.get_skill_descriptions()}

## Last Planning Info:
Skill Sequence: {last_plan_info.skill_sequence}
Skill Params: {last_plan_info.skill_params}

## Skills Execution Info:
{skills_execution_info}

## Output
Based on the above information and the current scene images that will be provided to you next, please do reflection and plan the next plan
Please respond with a JSON array representing a sequence of skills to execute.
- "method": A string representing the name of the skill to be called.
- "params": A JSON object containing the parameters for that skill. If a skill has no parameters, provide an empty object {{}}.
If you determine the task is finished just output an empty sequence [].

### Output Format:
Refection: YOUR REFECTION CONTENT HERE
Analysis: YOUR ANALYSIS OF MAKEING THIS PLAN
Plan:
```json
[
    {{
        "method": "skill1",
        "params": {{
            "param1_of_skill1": "value",
            "param2_of_skill1": "value"
        }}
    }},
    {{
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
You are monitoring robot's SKILL execution. Analyze the current situation and decide the next action.
You should first consider the execution of skills rather than the completion of tasks, based on the skill description and criterion.

Original Task: {task.description}
Current Skill: {current_skill["name"]}
Skill Description: {current_skill["discription"]}
Skill Parameters: {current_skill["parameters"]}
Skill Criterion: {current_skill["criterion"]}
Available Skills: {chr(10).join(f"- {skill}" for skill in self.skill_registry.list_skills())}

For each round of conversation, you will receive the current observation in Image format, You should analyze the image and response with a JSON object containing:
- action: One of the key in Skill Criterion
- reason: Explanation for the decision, based on the rule described in Skill Criterion
- current_scene_state: Description what happened in the scene, based on the observation
- confidence: Float between 0 and 1

Example response:
```json
{{
    "action": "failed",
    "reason": "The object was not grasped correctly",
    "current_scene_state", "The robot attempted to grasp the object but it slipped",
    "confidence": 0.9
}}
```

Next, Let's start first monitoring round.
"""
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

            # 从对象列表中构建 skill_sequence 和 skill_params
            skill_sequence = []
            skill_params = []
            for skill_object in plan_data:
                if isinstance(skill_object, dict):
                    # 提取技能名称，如果找不到 'method' 键，可以给个默认值或跳过
                    skill_name = skill_object.get("method")
                    if skill_name:
                        skill_sequence.append(skill_name)
                        # 提取参数，如果找不到 'params' 键或为 None，则提供一个空字典
                        params = skill_object.get("params", {})
                        if params is None:  # 额外处理模型可能返回 null 的情况
                            params = {}
                        skill_params.append(params)
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
