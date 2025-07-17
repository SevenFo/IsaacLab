"""
Skill registry and management system based on intelligent_robot_system design.
Skills execute directly in the Isaac subprocess with direct environment access.
"""

import importlib
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass

from .types import (
    SkillDefinition,
    SkillType,
    ExecutionMode,
    Action,
    SkillStatus,  # Make sure SkillStatus is defined in types.py
)


class SkillRegistry:
    """Registry for managing all available skills."""

    def __init__(self):
        self.skills: Dict[str, SkillDefinition] = {}
        # self.skill_instances: Dict[str, Any] = {} # Not typically used for function-based skills

    def register_skill(
        self,
        function: Callable,
        skill_type: SkillType,
        execution_mode: ExecutionMode,
        name: str | None = None,
        description: str = "",
        timeout: Optional[float] = None,
        requires_env: bool = False,
        criterion: Dict[str, str] | None = None,  # 新增参数
        enable_monitoring: bool = True,  # 是否启用监控
    ):
        """Register a skill in the registry."""
        name = name or function.__name__
        description = description or function.__doc__ or ""
        if name in self.skills:
            print(
                f"[SkillRegistry] Warning: Skill '{name}' already registered. Overwriting."
            )

        skill_def = SkillDefinition(
            name=name,
            skill_type=skill_type,
            execution_mode=execution_mode,
            function=function,
            description=description,
            timeout=timeout,
            requires_env=requires_env,
            criterion=criterion,
            enable_monitoring=enable_monitoring,  # 是否启用监控
        )

        self.skills[name] = skill_def
        print(
            f"[SkillRegistry] Registered skill: {name} ({skill_type.value}, {execution_mode.value})"
        )

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get a skill definition by name."""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """Get list of all registered skill names."""
        return list(self.skills.keys())

    def list_skills_by_type(self, skill_type: SkillType) -> List[str]:
        """Get list of skills by type."""
        return [
            name
            for name, skill in self.skills.items()
            if skill.skill_type == skill_type
        ]

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a skill."""
        skill = self.skills.get(name)
        if not skill:
            return None

        return {
            "name": skill.name,
            "type": skill.skill_type.value,
            "execution_mode": skill.execution_mode.value,
            "description": skill.description,
            "timeout": skill.timeout,
            "requires_env": skill.requires_env,
            "criterion": skill.criterion,
            "enable_monitoring": skill.enable_monitoring,
            "function_name": skill.function.__name__,  # function object itself is not easily serializable
        }

    def get_skill_descriptions(self) -> str:
        """
        Get a dictionary mapping skill names to their function docstrings.
        This is useful for providing detailed information about what each skill does,
        often used for generating prompts or help texts.
        The docstring is retrieved from the skill's underlying function.
        """

        formated_descriptions = [
            f"skill name: {name}: \n skill desc:\n{item.description}\n\n"
            for name, item in self.skills.items()
        ]

        return "".join(formated_descriptions)


    # _register_skills_from_module is effectively replaced by the decorator's direct registration

    def clear_registry(self):
        """Clear all registered skills."""
        self.skills.clear()
        # self.skill_instances.clear()
        print("[SkillRegistry] Cleared all registered skills")


# Global skill registry instance
_global_skill_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    global _global_skill_registry
    if _global_skill_registry is None:
        _global_skill_registry = SkillRegistry()
    return _global_skill_registry


def reset_skill_registry():
    """Reset the global skill registry."""
    global _global_skill_registry
    _global_skill_registry = None


# Skill registration decorator
def skill_register(
    skill_type: SkillType,
    execution_mode: ExecutionMode,
    name: str | None = None,
    description: str = "",
    timeout: Optional[float] = None,
    requires_env: bool = False,
    criterion: Dict[str, str] | None = None,  # 新增参数
    enable_monitoring: bool = True,  # 是否启用监控
):
    """Decorator to register a function as a skill."""

    def decorator(func):
        # Automatically register the skill with the global registry
        registry = get_skill_registry()
        registry.register_skill(
            skill_type=skill_type,
            execution_mode=execution_mode,
            name=name,
            function=func,
            description=description,
            timeout=timeout,
            requires_env=requires_env,
            criterion=criterion,
            enable_monitoring=enable_monitoring,  # 是否启用监控
        )
        return func

    return decorator


@dataclass
class SkillExecutionContext:  # Kept for potential future use, but SkillExecutor handles state now
    """Context for skill execution."""

    skill_name: str
    parameters: Dict[str, Any]
    simulator: Optional[Any] = (
        None  # In the new model, this is the 'env' passed to skill
    )
    timeout: Optional[float] = None
    status: SkillStatus = SkillStatus.NOT_STARTED
    result: Optional[Any] = None
    error: Optional[str] = None


class SkillExecutor:
    """Executes skills with direct environment interaction, typically within a simulator subprocess."""

    def __init__(self, skill_registry: SkillRegistry, env):
        self.registry = skill_registry
        self.current_skill: Optional[Callable] = None
        self.current_skill_name: Optional[str] = None
        self.current_skill_params: Optional[Dict[str, Any]] = None
        self.preaction_skills: list[Callable] = []
        self.status: SkillStatus = SkillStatus.IDLE  # Using SkillStatus enum
        self.env = env
        self.env_device = env.unwrapped.device
    
    def initialize_skill(
        self, skill_name: str, parameters: Dict[str, Any], policy_device: None,obs_dict:dict = {}
    ):
        if not policy_device:
            policy_device = self.env_device
        if self.is_running():
            print(
                f"[SkillExecutor] Another skill '{self.current_skill_name}' is already running. Terminating it."
            )
            self.terminate_current_skill()

        skill_def = self.registry.get_skill(skill_name)
        if not skill_def:
            print(
                f"[SkillExecutor] Skill '{skill_name}' not found, available skills: {self.registry.list_skills()}"
            )
            self.status = SkillStatus.FAILED
            return False

        try:
            if skill_def.execution_mode == ExecutionMode.STEPACTION:
                self.current_skill = skill_def.function(policy_device, **parameters)
                self.current_skill_name = skill_name
                self.current_skill_params = parameters
                self.status = SkillStatus.RUNNING
                print(f"[SkillExecutor] Started policy skill: {skill_name}")
                return True
            elif skill_def.execution_mode == ExecutionMode.PREACTION:
                self.preaction_skills.append(skill_def.function(policy_device, obs_dict, **parameters))
            elif (
                skill_def.execution_mode == ExecutionMode.DIRECT
            ):
                assert False, "Unsupported now!"
                # Direct execution is inherently blocking, so it completes immediately
                result = skill_def.function(*args_for_skill)
                print(
                    f"[SkillExecutor] Direct skill {skill_name} executed. Result: {result}"
                )
                self.status = SkillStatus.COMPLETED if result else SkillStatus.FAILED
                # No ongoing generator for direct skills
                self.current_skill_name = None
                self.current_skill_params = None
                return bool(result)
            else:
                print(f"[SkillExecutor] Unknown execution mode for skill {skill_name}")
                self.status = SkillStatus.FAILED
                return False
        except Exception as e:
            print(f"[SkillExecutor] Error initialize skill {skill_name}: {e}")
            import traceback

            traceback.print_exc()
            self.status = SkillStatus.FAILED
            return False

    def step(self, obs) -> Tuple:
        """
        Step the current non-blocking (generator) skill execution.
        Returns:
            state of each step
            None if no skill was running.
        """
        # preaction preprocess for obs
        for preaction_skill in self.preaction_skills:
            obs = preaction_skill(obs)
        
        action: Action = self.current_skill.select_action(obs)
        action_info = action.metadata["info"]
        if action_info == "error":
            step_result = (None, None, None, None, None)
            if action.data:
                action_data = action.data.to(self.env_device)
                step_result = self.env.step(action_data)
            print(f"[SkillExecutor] Error stepping skill {self.current_skill_name}")
            self.status = SkillStatus.FAILED
            self._reset_current_skill_state()
            return step_result
        elif action_info == "finished":
            print(
                f"[SkillExecutor] Skill {self.current_skill_name} finished successfully."
            )
            step_result = (None, None, None, None, None)
            if action.data:
                action_data = action.data.to(self.env_device)
                step_result = self.env.step(action_data)
            self.status = SkillStatus.COMPLETED
            self._reset_current_skill_state()
            return step_result
        elif action_info == "timeout":
            print(f"[SkillExecutor] Skill {self.current_skill_name} timed out.")
            self.status = SkillStatus.TIMEOUT
            self._reset_current_skill_state()
            return (None, None, None, None, None)
        else:
            action_data = action.data.to(self.env_device)
            step_result = self.env.step(action_data)
            # TODO 或许需要判断step后的情况？按道理来说应该是大模型来判断的！至少应该加一个超时判断
        return step_result

    def change_current_skill_status(
        self, skill_status: SkillStatus = SkillStatus.INTERRUPTED
    ) -> bool:
        """Terminate the current non-blocking skill execution."""
        if self.current_skill is not None and (self.status == SkillStatus.RUNNING or self.status == SkillStatus.PAUSED):
            self.status = skill_status
            print(
                f"[SkillExecutor] Change current skill status: {self.current_skill_name} with status: {skill_status}"
            )
            return True
        else:
            # this skill is already finished
            print(
                f"[SkillExecutor] Change current skill status no skill runing, unable to change"
            )
        return False

    def is_running(self) -> bool:
        """Check if a non-blocking skill is currently running."""
        return self.current_skill is not None and (self.status == SkillStatus.RUNNING) 

    def terminate_current_skill(
        self, skill_status: SkillStatus = SkillStatus.INTERRUPTED
    ) -> bool:
        """Terminate the current non-blocking skill execution."""
        if self.current_skill is not None:
            print(
                f"[SkillExecutor] Terminating skill: {self.current_skill_name} with status: {skill_status}"
            )
            self._reset_current_skill_state()
            self.status = skill_status
            return True
        else:
            # this skill is already finished
            print(
                f"[SkillExecutor] no skill runing, directly setting status to: {skill_status}"
            )
            self.status = skill_status
        return False

    def _reset_current_skill_state(self):
        self.current_skill = None
        self.current_skill_name = None
        self.current_skill_params = None

    def get_status_info(
        self,
    ) -> Dict[str, Any]:  # Renamed from get_status to avoid conflict if used elsewhere
        """Get current execution status information."""
        return {
            "status": self.status.value,
            "current_skill": self.current_skill_name,
            "skill_params": self.current_skill_params,
            "is_running": self.is_running(),
        }

    def reset_executor(self):  # Renamed from reset
        """Reset the executor to initial state."""
        self.terminate_current_skill()
        self.status = SkillStatus.IDLE
        print("[SkillExecutor] Executor reset.")


if __name__ == "__main__":
    print("Skill Exector Test")
