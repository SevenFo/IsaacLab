"""
Robot Brain System - A simple and efficient robot control architecture.

Core Philosophy:
- Qwen VL as the brain for task parsing and skill orchestration
- Gym-based simulator running in subprocess
- Generator-based skill execution for non-blocking operation
- Real-time monitoring and intervention capabilities
- Simple, extensible, and fast iteration-friendly design
"""

__version__ = "1.0.0"
__author__ = "Robot Brain System Team"

from .core.brain import QwenVLBrain
from .core.isaac_simulator import IsaacSimulator
from .core.skill_manager import SkillRegistry, SkillExecutor
from .core.system import RobotBrainSystem
from .core.types import (
    Action,
    Observation,
    Task,
    SkillPlan,
    SystemStatus,
    SkillType,
    ExecutionMode,
)

__all__ = [
    "QwenVLBrain",
    "IsaacSimulator",
    "SkillRegistry",
    "SkillExecutor",
    "RobotBrainSystem",
    "Action",
    "Observation",
    "Task",
    "SkillPlan",
    "SystemStatus",
    "SkillType",
    "ExecutionMode",
]
