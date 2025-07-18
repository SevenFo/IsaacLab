"""
Core types and enums for the robot brain system.
"""

from enum import Enum
import sys
from typing import Any, Dict, Optional, List, Union, Generator, Callable
from dataclasses import dataclass, field
import numpy as np
import torch


class SystemStatus(Enum):
    """System status enumeration."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class SkillStatus(Enum):
    """Skill execution status."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    IDLE = "idle"  # Skill is not currently executing
    TIMEOUT = "timeout"
    PENDING = "pending"  # Skill is waiting to be executed


@dataclass
class SkillStep:
    """Represents a single step in the skill plan."""

    name: str
    params: Dict[str, Any]
    status: SkillStatus = SkillStatus.PENDING

    def __str__(self):
        return f"[{self.status.name.upper()}] {self.name}({self.params})"


class SkillPlan:
    """
    An intelligent, mutable plan for task execution that supports
    operations like inserting, deleting, and modifying skills.
    """

    def __init__(
        self,
        task_id: str,
        skill_list: List[Dict[str, Any]],
        skill_monitoring_interval: float = 1.0,
    ):
        self.task_id = task_id
        self.skill_monitoring_interval = skill_monitoring_interval
        self.steps: List[SkillStep] = self._create_steps(skill_list)

    def _create_steps(self, skill_list: List[Dict[str, Any]]) -> List[SkillStep]:
        """Initializes SkillStep objects from the LLM's initial plan output."""
        # The initial plan is sorted by a 'step' key from the LLM
        sorted_list = sorted(skill_list, key=lambda item: item.get("step", sys.maxsize))
        return [
            SkillStep(name=item["method"], params=item.get("params", {}))
            for item in sorted_list
        ]

    def get_skill(self, index: int) -> Optional[SkillStep]:
        """Retrieves a skill step by its index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def insert_skill(self, index: int, name: str, params: Dict[str, Any]):
        """Inserts a new skill at a specific index."""
        if not (0 <= index <= len(self.steps)):
            raise IndexError("Insertion index is out of bounds.")

        new_step = SkillStep(name=name, params=params)
        self.steps.insert(index, new_step)
        print(f"[SkillPlan] Inserted at index {index}: {new_step.name}")

    def delete_skill(self, index: int):
        """Deletes a skill at a specific index."""
        if not (0 <= index < len(self.steps)):
            raise IndexError("Deletion index is out of bounds.")

        removed_step = self.steps.pop(index)
        print(f"[SkillPlan] Deleted skill at index {index}: {removed_step.name}")

    def modify_skill(
        self,
        index: int,
        new_name: Optional[str] = None,
        new_params: Optional[Dict[str, Any]] = None,
    ):
        """Modifies the name and/or parameters of an existing skill."""
        skill = self.get_skill(index)
        if not skill:
            raise IndexError("Modification index is out of bounds.")

        if new_name:
            print(
                f"[SkillPlan] Modified skill {index}: name changed from '{skill.name}' to '{new_name}'"
            )
            skill.name = new_name
        if new_params is not None:  # Allow empty dict to be set
            print(
                f"[SkillPlan] Modified skill {index}: params changed from '{skill.params}' to '{new_params}'"
            )
            skill.params = new_params
        skill.status = SkillStatus.PENDING  # Reset status after modification

    def mark_status(self, index: int, status: SkillStatus):
        """Marks a skill as completed, failed, etc."""
        skill = self.get_skill(index)
        if skill:
            print(
                f"[SkillPlan] Set status for skill {index} ({skill.name}) to {status.name}"
            )
            skill.status = status

    def get_next_pending_skill_with_index(self) -> Optional[tuple[int, SkillStep]]:
        """Finds the next skill that is not yet completed."""
        for i, step in enumerate(self.steps):
            if step.status == SkillStatus.PENDING:
                return i, step
        return None

    def is_complete(self) -> bool:
        """Checks if all steps in the plan are completed."""
        return all(step.status == SkillStatus.COMPLETED for step in self.steps)

    def pretty_print(self) -> str:
        """Returns a nicely formatted string representation of the plan."""
        header = f"--- Skill Plan for Task: {self.task_id} ---\n"
        plan_str = "\n".join(f"  Step {i}: {step}" for i, step in enumerate(self.steps))
        return header + plan_str


@dataclass
class Action:
    """Action representation for the environment."""

    data: torch.tensor
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def to_numpy(self) -> np.ndarray:
        """Convert action to numpy array format."""
        if isinstance(self.data, np.ndarray):
            return self.data
        elif isinstance(self.data, dict):
            # Flatten dict to array based on some convention
            return np.concatenate([np.array(v).flatten() for v in self.data.values()])
        else:
            return np.array(self.data)


@dataclass
class Observation:
    """Observation representation from the environment."""

    data: Union[np.ndarray, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get observation data by key."""
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default


@dataclass
class Task:
    """Task representation for the brain to process."""

    id: str
    description: str
    image: Optional[str] = None  # base64 encoded image
    priority: int = 1
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SkillPlan_v:
    """Plan generated by the brain for task execution."""

    task_id: str
    skill_sequence: List[str]
    skill_params: List[Dict[str, Any]]
    skill_monitoring_interval: float = 1.0  # seconds
    expected_duration: Optional[float] = None


class SkillType(Enum):
    """Types of skills."""

    FUNCTION = "function"  # Simple Python function (fixed logic)
    POLICY = "policy"  # Trained RL policy
    OBSERVATION = "observation"


class ExecutionMode(Enum):
    """Execution modes for skills."""

    DIRECT = "direct"  # Execute immediately without yielding
    STEPACTION = "stepaction"
    PREACTION = "preaction"


# Type aliases for better readability
PolicyFunction = object  # TODO
DirectFunction = Callable[[Dict[str, Any]], bool]


@dataclass
class SkillDefinition:
    """Definition of a skill."""

    name: str
    skill_type: SkillType
    execution_mode: ExecutionMode
    function: any  # TODO
    description: str = ""
    timeout: Optional[float] = None
    requires_env: bool = False
    criterion: Dict[str, str] | None = None  # 新增字段，用于描述技能状态判定条件
    enable_monitoring: bool = True  # 是否启用监控

    def __post_init__(self):
        if self.criterion is None:
            self.criterion = {
                "successed": "Skill completed successfully",
                "failed": "Skill failed to complete",
            }


@dataclass
class SkillExecution:
    """Current execution state of a skill."""

    skill_name: str
    status: SkillStatus
    generator: Optional[Generator[Action, Observation, Any]] = None
    start_time: Optional[float] = None
    last_action: Optional[Action] = None
    last_observation: Optional[Observation] = None
    error_message: Optional[str] = None


@dataclass
class SystemState:
    """Current state of the entire system."""

    status: SystemStatus = SystemStatus.IDLE
    is_running: bool = False
    # current_skill_generator: Optional[Generator] = None # Replaced by tracking status from IsaacSim
    current_task: Optional[Task] = (
        None  # To store the task being processed by the brain
    )
    last_observation: Optional[Observation] = None
    # last_action: Optional[Action] = None # Actions are now primarily handled in subprocess
    error_message: Optional[str] = None
    # New state for tracking skill execution in subprocess
    sub_skill_status: Dict[str, Any] = field(default_factory=dict)
    obs_history: list[Observation] = field(default_factory=list)
    plan_history: list[SkillPlan] = field(default_factory=list)
    skill_history: list[dict] = field(default_factory=list)
