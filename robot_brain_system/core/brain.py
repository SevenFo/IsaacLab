"""
Brain component that uses Qwen VL for task planning and monitoring.
"""

import time
import json
import os
from PIL import Image
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from robot_brain_system.core.types import Task, SkillPlan, SystemStatus
from robot_brain_system.core.skill_manager import SkillRegistry

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

    def add_system_prompt(self, prompt: str = "You are a helpful assistant."):
        """Add a system prompt to the memory."""

        self.history.append(
            {"role": "system", "content": [{"type": "text", "text": prompt}]}
        )

    def add_user_input(
        self, contents: List[str | Image.Image | list[Image.Image]]
    ):
        """Add user input to the memory."""
        item = {
            "role": "user",
            "content": [],
        }
        for content in contents:
            if isinstance(content, str):
                item["content"].append({"type": type, "text": content})
            elif isinstance(content, Image.Image):
                item["content"].append({"type": type, "image": content})
            elif isinstance(content, list) and all(
                isinstance(img, Image.Image) for img in content
            ):
                item["content"].append(
                    {"type": type, "video": content, "fps": 5}
                )
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
            self.history[-last_n:]
            if len(self.history) - 1 >= last_n
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
        self.max_tokens = self.qwen_config.get("max_tokens", 512)

        # Monitoring configuration
        self.monitoring_interval = config.get(
            "monitoring_interval", 1.0
        )  # seconds
        self.max_retries = config.get("max_retries", 3)
        self.visualize = config.get("visualize", False)
        self.log_path = config.get("log_path", "./logs")
        self.monitor_waiting_times = 3
        self.monitor_memory = BrainMemory()

    def initialize(self):
        """Initialize the brain component."""
        print("[QwenVLBrain] Initializing...")
        # Initialize model adapter
        self.model_adapter = None
        if MODEL_ADAPTERS_AVAILABLE:
            try:
                self._initialize_model_adapter()
            except Exception as e:
                print(f"[QwenVLBrain] Failed to initialize model adapter: {e}")
                print("[QwenVLBrain] Falling back to mock implementation")
                self.adapter_type = "mock"
        else:
            self.adapter_type = "mock"
        print(f"[QwenVLBrain] Initialized with adapter: {self.adapter_type}")

    def _initialize_model_adapter(self):
        """Initialize the appropriate model adapter based on configuration."""
        if self.adapter_type == "qwen_vl":
            if not self.model_path:
                raise ValueError("model_path is required for QwenVL adapter")
            from .model_adapters import QwenVLAdapter

            self.model_adapter = QwenVLAdapter(self.model_path)
            print(
                f"[QwenVLBrain] Initialized QwenVL adapter with model: {self.model_path}"
            )

        elif self.adapter_type == "openai":
            if not self.api_key:
                raise ValueError("api_key is required for OpenAI adapter")
            from .model_adapters import OpenAIAdapter

            model_name = (
                self.model_name if self.model_name != "qwen-vl" else None
            )
            self.model_adapter = OpenAIAdapter(
                api_key=str(self.api_key),
                base_url=str(self.base_url) if self.base_url else None,
                model_name=str(model_name) if model_name else None,
            )
            print(
                f"[QwenVLBrain] Initialized OpenAI adapter with model: {model_name or 'default'}"
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

    def parse_task(
        self, instruction: str, image_data: Optional[str] = None
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

            skill_info = self.skill_registry.get_skill_info(
                plan.skill_sequence[0]
            )

            assert skill_info is not None, (
                f"Skill {plan.skill_sequence[0]} not found in registry"
            )

            current_skill = {
                "name": plan.skill_sequence[0],
                "parameters": plan.skill_params[0],
                "criterion": skill_info["criterion"],
            }

            self.monitor_memory.add_system_prompt(
                self._format_system_prompt_for_monitoring(
                    self.state.current_task, current_skill, None
                )
            )

            self.current_monitor_time = 0

            print(f"[QwenVLBrain] Started executing task: {task.description}")
            return plan

        except Exception as e:
            self.state.status = SystemStatus.ERROR
            self.state.error_message = str(e)
            raise

    def get_next_skill(self) -> Optional[Dict[str, Any]]:
        """
        Get the next skill to execute in the current plan.

        Returns:
            Dict with skill name and parameters, or None if plan complete
        """
        if (
            not self.state.current_plan
            or self.state.current_skill_index
            >= len(self.state.current_plan.skill_sequence)
        ):
            return None

        skill_name = self.state.current_plan.skill_sequence[
            self.state.current_skill_index
        ]
        skill_params = self.state.current_plan.skill_params[
            self.state.current_skill_index
        ]

        return {
            "name": skill_name,
            "parameters": skill_params,
            "index": self.state.current_skill_index,
        }

    def advance_skill(self):
        """Advance to the next skill in the plan."""
        if self.state.current_plan:
            self.state.current_skill_index += 1
            self.current_monitor_time = 0
            if self.state.current_skill_index >= len(
                self.state.current_plan.skill_sequence
            ):
                # Plan completed
                self.state.status = SystemStatus.IDLE
                self.state.current_task = None
                self.state.current_plan = None
                self.state.current_skill_index = 0
                print("[QwenVLBrain] Task execution completed")

    def should_monitor(self) -> bool:
        """Check if it's time to monitor the current skill execution."""
        if (
            self.state.status != SystemStatus.EXECUTING 
            or not self.state.current_plan
        ):
            return False

        current_time = time.time()
        return (
            current_time - self.state.last_monitoring_time
        ) >= self.state.current_plan.monitoring_interval

    def monitor_execution(
        self, current_observation: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Monitor current skill execution and make decisions.

        Args:
            current_observation: Current environment observation

        Returns:
            Dict with monitoring decision (continue, interrupt, retry, etc.)
        """
        try:
            self.state.last_monitoring_time = time.time()

            if not self.state.current_plan or not self.state.current_task:
                return {"action": "continue", "reason": "No active task"}

            if self.current_monitor_time < self.monitor_waiting_times:
                self.current_monitor_time += 1
                return {
                    "action": "continue",
                    "reason": f"current_monitor_time: {self.current_monitor_time}",
                }

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

    def _query_qwen_for_plan(
        self, task: Task, skill_descriptions: str
    ) -> SkillPlan:
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
                image=task.image
                if task.image
                else None,  # Pass None if no image
            )

            # Generate response
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

    def _mock_plan_task(
        self, task: Task, skill_descriptions: str
    ) -> SkillPlan:
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
            skill_params = [
                {"target_position": [0.4, 0.0, 0.4], "tolerance": 0.01}
            ]
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
            monitoring_interval=self.monitoring_interval,
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
            return self._mock_monitoring_decision(
                task, current_skill, observation
            )
        assert type(observation) is list
        try:
            if self.visualize:
                inspector_rgb = (
                    observation[-1]
                    .data["rgb_camera"]["inspector"][0]
                    .cpu()
                    .numpy()
                )
                import matplotlib.pyplot as plt

                plt.imshow(inspector_rgb)
                plt.axis("off")
                plt.savefig(
                    os.path.join(
                        self.log_path,
                        f"monitor_{current_skill['name']}_input_{len(self.monitor_memory.history)}.png",
                    )
                )
            video_frames = []
            def calculate_indices(jump, total, available):
                if available >= total * jump + 1:
                    return list(range(-total*jump +1, 0,jump))
                else:
                    return None
            # 计算可用的观察帧数
            available_frames = len(observation)
            jump = 6
            total = 10
            indices = []
            if not (indices := calculate_indices(jump,total,available_frames)):
                return {"action": "not enough", "reason": "not enough availabel frames"}

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
                    f"monitor_{current_skill['name']}_input_{len(self.monitor_memory.history)}.gif",
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
            response_text, _ = self.model_adapter.generate_response(
                self.monitor_memory.fetch_history(last_n=3),
                max_tokens=self.max_tokens
                // 2,  # Use fewer tokens for monitoring
            )
            self.monitor_memory.add_assistant_output(response_text)
            # Parse the response to extract monitoring decision
            decision = self._parse_monitoring_response(response_text)

            print(
                f"[QwenVLBrain] Monitoring decision using {self.adapter_type}: {decision['action']}"
            )
            return decision

        except Exception as e:
            print(
                f"[QwenVLBrain] Error in monitoring: {e.__class__.__name__}: {e}"
            )
            import traceback

            traceback.print_exc()
            print("[QwenVLBrain] Falling back to mock monitoring")
            return self._mock_monitoring_decision(
                task, current_skill, observation
            )

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
            if "{" in response_text and "}" in response_text:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                json_str = response_text[start_idx:end_idx]

                decision_data = json.loads(json_str)

                # Extract required fields with defaults
                action = decision_data.get("action", "continue")
                reason = decision_data.get("reason", "No reason provided")
                confidence = decision_data.get("confidence", 0.5)

                # Validate action
                valid_actions = [
                    "continue",
                    "interrupt",
                    "successed",
                    "failed",
                    "retry",
                    "modify",
                    "complete",
                    "error",
                    "not determinable"
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
                # If no JSON found, try to parse from text
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
            word in response_lower
            for word in ["stop", "interrupt", "halt", "abort"]
        ):
            return {
                "action": "interrupt",
                "reason": "Model suggested interruption",
            }
        elif any(
            word in response_lower for word in ["retry", "again", "restart"]
        ):
            return {"action": "retry", "reason": "Model suggested retry"}
        elif any(
            word in response_lower for word in ["complete", "done", "finished"]
        ):
            return {
                "action": "complete",
                "reason": "Model indicated completion",
            }
        elif any(
            word in response_lower for word in ["error", "fail", "problem"]
        ):
            return {"action": "error", "reason": "Model detected error"}
        else:
            return {
                "action": "continue",
                "reason": "Model suggested continuation",
            }

    def _format_prompt_for_planning(
        self, task: Task, skill_descriptions: str
    ) -> str:
        """Format a prompt for Qwen VL planning."""
        prompt = f"""
You are a robot task planner. Given a task description and available skills, create an execution plan.

Task: {task.description}

Available Skills:
{skill_descriptions}

Please respond with a JSON object containing:
- skill_sequence: List of skill names to execute in order
- skill_params: List of parameter dictionaries for each skill
- monitoring_interval: How often to check progress (seconds)

Example response:
{{
    "skill_sequence": ["pick_and_place", "inspect_object"],
    "skill_params": [{{"target": "object1"}}, {{"target": "result"}}],
    "monitoring_interval": 1.0
}}
"""

        return prompt

    def _format_system_prompt_for_monitoring(
        self,
        task: Task,
        current_skill: Dict[str, Any],
        observation: Optional[Any] = None,
    ) -> str:
        """Format a prompt for Qwen VL monitoring."""
        prompt = f"""
You are monitoring robot task execution. Analyze the current situation and decide the next action.

Original Task: {task.description}
Current Skill: {current_skill["name"]}
Skill Parameters: {current_skill["parameters"]}
Skill Criterion: {current_skill["criterion"]}
Available Skills: {chr(10).join(f"- {skill}" for skill in self.skill_registry.list_skills())}

For each round of conversation, you will receive the current observation in Image format, You should analyze the image and response with a JSON object containing:
- action: One of the key in Skill Criterion
- reason: Explanation for the decision, based on the rule described in Skill Criterion
- current_scene_state: Description what happened in the scene, based on the observation
- confidence: Float between 0 and 1

Example response:
{{
    "action": "failed",
    "reason": "The object was not grasped correctly",
    "current_scene_state", "The robot attempted to grasp the object but it slipped",
    "confidence": 0.9
}}

Next, Let's start first monitoring round.
"""
        return prompt

    def _parse_plan_response(
        self, response_text: str, task: Task
    ) -> SkillPlan:
        """
        Parse the response from the model adapter to extract a skill plan.

        Args:
            response_text: The raw text response from the model
            task: The original task

        Returns:
            SkillPlan object
        """
        try:
            # Try to parse JSON response
            if "{" in response_text and "}" in response_text:
                # Extract JSON part from the response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                json_str = response_text[start_idx:end_idx]

                plan_data = json.loads(json_str)

                skill_sequence = plan_data.get("skill_sequence", [])
                skill_params = plan_data.get("skill_params", [])
                monitoring_interval = plan_data.get(
                    "monitoring_interval", self.monitoring_interval
                )

                # Validate that we have the same number of skills and parameters
                if len(skill_sequence) != len(skill_params):
                    print(
                        "[QwenVLBrain] Warning: Mismatch between skill sequence and parameters length"
                    )
                    # Pad with empty dicts if needed
                    while len(skill_params) < len(skill_sequence):
                        skill_params.append({})

                return SkillPlan(
                    task_id=task.id,
                    skill_sequence=skill_sequence,
                    skill_params=skill_params,
                    monitoring_interval=monitoring_interval,
                    expected_duration=len(skill_sequence) * 10.0,
                )

            else:
                # If no JSON found, try to extract skills from text
                print(
                    "[QwenVLBrain] No JSON found in response, trying text parsing"
                )
                return self._parse_text_response(response_text, task)

        except json.JSONDecodeError as e:
            print(f"[QwenVLBrain] Failed to parse JSON response: {e}")
            print(f"[QwenVLBrain] Response was: {response_text}")
            # Fallback to mock implementation
            return self._mock_plan_task(task, [])
        except Exception as e:
            print(f"[QwenVLBrain] Error parsing response: {e}")
            return self._mock_plan_task(task, [])

    def _parse_text_response(
        self, response_text: str, task: Task
    ) -> SkillPlan:
        """
        Parse a text response that doesn't contain JSON.

        This is a fallback method for when the model doesn't return structured JSON.
        """
        # Simple keyword-based parsing
        skill_sequence = []
        skill_params = []

        response_lower = response_text.lower()

        # Look for skill names in the response
        available_skills = (
            self.skill_registry.list_skills() if self.skill_registry else []
        )

        for skill_name in available_skills:
            if skill_name.lower() in response_lower:
                skill_sequence.append(skill_name)
                skill_params.append({})  # Default empty parameters

        # If no skills found, fallback to mock planning
        if not skill_sequence:
            print(
                "[QwenVLBrain] No skills found in text response, using mock planning"
            )
            return self._mock_plan_task(task, available_skills)

        return SkillPlan(
            task_id=task.id,
            skill_sequence=skill_sequence,
            skill_params=skill_params,
            monitoring_interval=self.monitoring_interval,
            expected_duration=len(skill_sequence) * 10.0,
        )
